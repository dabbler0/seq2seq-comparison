function append_table(dest, inp)

  for i=1,#inp do
    table.insert(dest, inp[1])
  end

  return dest
end

function slice_table(src, index)
  local result = {}
  for i=index,#src do
    table.insert(result, src[i])
  end

  return result
end

function LRP_saliency(
    opt,
    alphabet,
    encoder_clones,
    normalizer,
    sentence)

  local first_hidden = {}
  for i=1,2*opt.num_layers do
    table.insert(first_hidden, torch.Tensor(1, opt.rnn_size):zero():cuda():expand(opt.rnn_size, opt.rnn_size))
  end

  local one_hots = {}
  for t=1,#sentence do
    one_hots[t] = torch.Tensor(1, #alphabet):zero():cuda()
    one_hots[t][1][sentence[t]] = 1
    one_hots[t] = one_hots[t]:expand(opt.rnn_size, #alphabet)
  end

  local rnn_state = first_hidden
  for t=1,#sentence do
    local encoder_input = {one_hots[t]}
    append_table(encoder_input, rnn_state)
    rnn_state = encoder_clones[t]:forward(encoder_input)
  end

  -- Relevance
  local relevances = {}
  for i=1,2*opt.num_layers-1 do
    table.insert(
      relevances,
      torch.Tensor(opt.rnn_size, opt.rnn_size):zero():cuda()
    )
  end

  -- Relevance at point x is f(x) / var(f(x))
  table.insert(
    relevances,
    torch.diag(torch.cdiv(rnn_state[#rnn_state][1], normalizer[2][1])):cuda()
  )

  local relevance_state = relevances
  local input_relevances = {}
  for t=#sentence,1,-1 do
    relevance_state = LRP(encoder_clones[t], relevance_state)
    input_relevances[t] = relevance_state[1]:view(opt.rnn_size) -- batch_size x alphabet_size, summed -> batch_size
    relevance_state = slice_table(relevance_state, 2)
  end

  return input_relevances
end

function LRP(gmodule, R)
  local relevances = {}

  -- Topological sort of nodes in the
  -- gmodule
  local toposorted = {}
  local visited = {}

  function dfs(node)
    if visited[node] then
      return
    end
    visited[node] = true

    for dependency, t in pairs(node.data.reverseMap) do
      dfs(dependency)
    end

    table.insert(toposorted, node.data)
  end

  dfs(gmodule.innode)

  -- PROPAGATION
  relevances[gmodule.outnode.data] = R

  for i=1,#toposorted do
    local node = toposorted[i]
    local relevance = relevances[node]

    -- Propagate input relevance
    local clock = os.clock()
    local input_relevance = relevance_propagate(node, relevance)
    local final = os.clock() - clock
    if node.module then
      print('Elapsed propagation time total:', final, torch.typename(node.module))
    else
      print('Elapsed propagation time total:', final, '(nil)')
    end

    if #node.mapindex == 1 then
      -- Case 1: Select node
      if node.selectindex then
        -- Initialize the selection table
        if relevances[node.mapindex[1]] == nil then
          relevances[node.mapindex[1]] = {}
        end

        if relevances[node.mapindex[1]][node.selectindex] == nil then
          relevances[node.mapindex[1]][node.selectindex] = input_relevance
        else
          relevances[node.mapindex[1]][node.selectindex] = relevances[node.mapindex[1]][node.selectindex] + input_relevance
        end

      -- Case 2: Not select node
      else

        if relevances[node.mapindex[1]] == nil then
          relevances[node.mapindex[1]] = input_relevance
        else
          relevances[node.mapindex[1]] = relevances[node.mapindex[1]] + input_relevance
        end

      end

    else

      -- Table uses information from several input nodes
      for j=1,#node.mapindex do
        if relevances[node.mapindex[j]] == nil then
          relevances[node.mapindex[j]] = input_relevance[j]
        else
          relevances[node.mapindex[j]] = relevances[node.mapindex[j]] + input_relevance[j]
        end
      end

    end

  end

  return relevances[gmodule.innode.data]
end

function relevance_propagate(node, R)
  -- For nodes without modules (like select nodes),
  -- pass through
  if node.module == nil then return R end

  local I = node.input
  local module = node.module

  -- Unpack single-element inputs
  if #I == 1 then I = I[1] end

  -- MulTable: pass-through on non-gate inputs
  if torch.typename(module) == 'nn.CMulTable' then
    local input_nodes = node.mapindex

    local result_table = {}
    -- Identify the gates as the Sigmoid nodes
    for i=1,#input_nodes do
      if torch.typename(input_nodes[i].module) == 'nn.Sigmoid' then
        table.insert(result_table, R:clone():zero())
      else
        table.insert(result_table, R)
      end
    end

    return result_table
  end

  if torch.typename(module) == 'nn.CAddTable' then
    return lrp_add(I, R)
  end

  if torch.typename(module) == 'nn.Linear' then
    -- For the closest-to-input layer, don't try to do the linear
    -- propagation because it's suuuper memory intensive
    if I:size(2) > 50000 then
      return R:sum(2)
    end

    return lrp_linear(I, module, R)
  end

  if torch.typename(module) == 'nn.Reshape' then
    return R:clone():viewAs(I)
  end

  if torch.typename(module) == 'nn.SplitTable' then
    return torch.cat(R, module.dimension)
  end

  if torch.typename(module) == 'nn.LookupTable' then
    -- Batch mode, so sum over second (embedding) dimension
    return R:sum(2)
  end

  -- All other cases: pass-through
  return R
end


-- Morrored as closely as possible from Arras's LRP_for_lstm
local eps = 0.001
function lrp_linear(hin, module, relevance)
  local iclock = os.clock()
  local w = module.weight:t():clone():contiguous()
  local b = module.bias
  local bias_factor = 1

  hin = hin:clone():contiguous()

  -- No bias means zero bias
  if b == nil then b = relevance[1]:clone():zero() end
  b = b:clone():contiguous()

  local D = hin:size(2) -- Input has size batch x D
  local M = relevance:size(2) -- Output has size batch x M
  local batch = hin:size(1)

  -- Compute output. Linear layers always transform one-to-one,
  -- so this is indeed the output.

  local hout = torch.mm(hin, w) + b:view(1, M):expand(batch, M)
  local bias_nb_units = D

  -- Take sign, positive when zero
  local sign_out = torch.sign(hout) --:clone():fill(1)
  sign_out:add(torch.eq(hout, 0):cuda())

  local numer = w:view(1, D, M):expand(batch, D, M):clone()
  numer:cmul(hin:view(batch, D, 1):expand(batch, D, M))
  numer:add((bias_factor * b:view(1, 1, M) / bias_nb_units):expand(batch, D, M))
  numer:add((eps * sign_out:view(batch, 1, M) / bias_nb_units):expand(batch, D, M))

  local denom = (hout + eps * sign_out):view(batch, 1, M):expand(batch, D, M) -- Size b x D x M

  local message = torch.cmul(torch.cdiv(numer, denom), relevance:view(batch, 1, M):expand(batch, D, M)) -- Size b x D x M

  local Rin = message:sum(3) -- Size b x D

  return Rin
end

function lrp_add(inputs, R)
  local iclock = os.clock()

  sum_inputs = inputs[1]:clone()
  for i=2,#inputs do
    sum_inputs:add(inputs[i])
  end

  print('Summed', os.clock() - iclock)

  sign_sum = torch.sign(sum_inputs)
  sign_sum:add(torch.eq(sum_inputs, 0):cuda())
  print('Signed', os.clock() - iclock)

  sum_inputs:add(eps * sign_sum)
  print('Stabilized', os.clock() - iclock)

  local factors = {}
  for i=1,#inputs do
    table.insert(factors, torch.cdiv(inputs[i], sum_inputs))
  end
  print('Sliced', os.clock() - iclock)

  local relevances = {}
  for i=1,#inputs do
    table.insert(relevances, torch.cmul(R, factors[i]))
  end
  print('Scaled', os.clock() - iclock)

  return relevances
end
