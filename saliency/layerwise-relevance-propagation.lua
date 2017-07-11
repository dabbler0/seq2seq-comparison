-- Importance measured by LRP as described in this paper:
-- https://arxiv.org/pdf/1706.07206.pdf
--
-- Doesn't seem to work; might be flaw in the model when used with Uni-LSTMs,
-- might be flaw in the implementation

function append_table(dest, inp)
  for i=1,#inp do
    table.insert(dest, inp[i])
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
    neuron,
    sentence)

  -- Construct beginning hidden state
  local first_hidden = {}
  for i=1,2*opt.num_layers do
    table.insert(first_hidden, torch.Tensor(1, opt.rnn_size):zero():cuda())
  end

  -- Create one-hot encoding for each sentence
  local one_hots = {}
  for t=1,#sentence do
    one_hots[t] = torch.Tensor(1, #alphabet):zero():cuda()
    one_hots[t][1][sentence[t]] = 1 -- one-hot encoding
  end

  -- Forward pass
  local rnn_state = first_hidden
  for t=1,#sentence do
    local encoder_input = {one_hots[t]}
    append_table(encoder_input, rnn_state)
    rnn_state = encoder_clones[t]:forward(encoder_input)
    if testNan(rnn_state) then
      print('An rnn_state was NaN, we\'re doomed.')
      print('This occured at time', t)
      for i=1,#rnn_state do
        print(rnn_state[i])
      end
      return nil
    end
  end

  -- Construct relevance with desired neuron
  local relevances = {}
  for i=1,2*opt.num_layers do
    table.insert(relevances, torch.Tensor(1, opt.rnn_size):zero():cuda())
  end

  -- Inject relevance at desired neuron
  relevances[#relevances][1][neuron] = (rnn_state[#rnn_state][1][neuron] - normalizer[1][1][neuron]) / normalizer[2][1][neuron] --/ normalizer[2][1][neuron] -- stdev for normalization

  -- Propagate relevance
  local relevance_state = relevances
  local input_relevances = {}
  for t=#sentence,1,-1 do
    relevance_state = LRP(encoder_clones[t], relevance_state)
    input_relevances[t] = relevance_state[1][1][sentence[t]]
    relevance_state = slice_table(relevance_state, 2)
  end

  return input_relevances
end

-- Recursively test for nan
function testNan(el)
  if torch.type(el) == 'table' then
    for k,v in pairs(el) do
      if testNan(v) then
        return true
      end
      return false
    end
  else
    local sum = el:sum()
    return sum ~= sum -- True only for nan
  end
end

-- Layer-wise relevance propagation on a network.
-- To be called immediately after a forward pass has been done.
function LRP(gmodule, R)
  local relevances = {}
  local agenda = {}

  -- Topological sort the graph nodes by dfs
  local toposorted = {}
  local visited = {}

  function dfs(node)
    if visited[node] then
      return
    end
    visited[node] = true

    for dependency, t in pairs(node.data.reverseMap) do
      -- Add everyone we depend on
      dfs(dependency)
    end

    -- Add us
    table.insert(toposorted, node.data)
  end

  -- Initialize the output relevance
  dfs(gmodule.innode) --outnode)
  relevances[gmodule.outnode.data] = R

  -- We are now guaranteed that if we need X to compute Y,
  -- then X comes before Y in the toposort. So we can
  -- iterate through the toposort and compute relevances.
  for i=1,#toposorted do
    local node = toposorted[i]
    local relevance = relevances[node]

    -- Propagate input relevance
    local input_relevance = relPropagate(node, relevance)

    if input_relevance == nil then
      print('we failed.')
      print('typename', torch.typename(node.module))
      print('relevance', #relevance)
      return nil
    else
      if testNan(input_relevance) then
        print('we failed (there was a nan)')
        print('typename', torch.typename(node.module))
        print('rel', relevance)
        if torch.type(input_relevance) == 'table' then
          for k,v in pairs(input_relevance) do
            print('inp_rel', k, v)
          end
        else
          print('inp_rel', input_relevance)
        end
        return nil
      end
    end

    -- Accumulate.
    if #node.mapindex == 1 then

      -- Special case for when there is only one
      -- input, because then the input_relevance is not a table

      -- In the case that we are a select node,
      -- fill in just the part of the input table that we selected.
      if node.selectindex then
        if relevances[node.mapindex[1]] == nil then
          relevances[node.mapindex[1]] = {}
        end

        if relevances[node.mapindex[1]][node.selectindex] == nil then
          relevances[node.mapindex[1]][node.selectindex] = input_relevance
        else
          relevances[node.mapindex[1]][node.selectindex] = relevances[node.mapindex[1]] + input_relevance
        end

      -- Otherwise, in the normal case, this transfers one tensor to one tensor
      else
        if relevances[node.mapindex[1]] == nil then
          relevances[node.mapindex[1]] = input_relevance
        else
          relevances[node.mapindex[1]] = relevances[node.mapindex[1]] + input_relevance
        end
      end

    else

      -- Otherwise use input_relevance as a table
      for j=1,#node.mapindex do
        if relevances[node.mapindex[j]] == nil then
          relevances[node.mapindex[j]] = input_relevance[j]
        else
          relevances[node.mapindex[j]] = relevances[node.mapindex[j]] + input_relevance[j]
        end
      end

    end

  end

  -- Return the input relevance
  return relevances[gmodule.innode.data]
end

-- Propagate relevance through a single
-- gModule node.
--
-- Supports all the node types that appear in the
-- encoder LSTM.
function relPropagate(node_data, R)
  -- For nodes with no module, pass through.
  if node_data.module == nil then
    return R
  end

  local I = node_data.input
  local module = node_data.module

  if #I == 1 then I = I[1] end

  -- For CMulTable, determine which one is the "gate",
  -- and propagate back only through the non-gate.
  if torch.typename(module) == 'nn.CMulTable' then
    local input_nodes = node_data.mapindex

    -- We currently don't support MulTable for more than 2
    -- factors
    if #input_nodes > 2 then
      print('Too many input nodes:', #input_nodes)
      return nil
    end

    local result_table = {}
    for i=1,#input_nodes do
      if torch.typename(input_nodes[i].module) == 'nn.Sigmoid' then
        -- Zeros
        table.insert(result_table, R:clone():zero())
      else
        -- Pass-through
        table.insert(result_table, R)
      end
    end

    return result_table

  -- For CAddTable, use relPropagateAdd
  elseif torch.typename(module) == 'nn.CAddTable' then
    return relPropagateAdd(I, R)

  -- For Linear, use relPropagateLinear
  elseif torch.typename(module) == 'nn.Linear' then
    return relPropagateLinear(I, module.weight, module.bias, R)

  -- Reshape and split table should
  -- rearrange the relevance exactly backwards
  -- as the inputs were rearranged forward
  elseif torch.typename(module) == 'nn.Reshape' then
    return R:clone():viewAs(I)
  elseif torch.typename(module) == 'nn.SplitTable' then
    return torch.cat(R, module.dimension)

  -- For LookupTable, the single input receives all
  -- of the relevance
  elseif torch.typename(module) == 'nn.LookupTable' then
    return R:sum()

  -- Dropout; pass through
  elseif torch.typename(module) == 'nn.Dropout' then
    return R

  -- Identity; pass through
  elseif torch.typename(module) == 'nn.Identity' then
    return R

  -- Nonlinearities; pass through.
  elseif torch.typename(module) == 'nn.Tanh' then
    return R
  elseif torch.typename(module) == 'nn.Sigmoid' then
    return R
  end
end

-- Here's how it works.
-- Suppose there's a Linear layer with input V and weight W.
-- Then for each output index i, take Norm1(V x W[i]) (that's the hadamard product)
-- and add R_v += Norm1(V x W[i]) * R_i
--
-- In other words, with a Linear layer with input V and weight W and output relevance R,
-- we want:
local epsilon = 0.001
function relPropagateLinear(V, W, B, R)
  local rs, vs = R:size(2), V:size(2)

  -- Get contributions
  local contributions = torch.cmul(V:view(1, vs):expandAs(W), W)
  contributions = contributions + epsilon * torch.sign(contributions) / vs -- Add stabilizer

  -- Make conservative. Bias is going to be an (rs) vector,
  -- which we distribute evenly among the contributions
  if B ~= nil then
    contributions = contributions + B:view(rs, 1):expandAs(W) / vs
  end

  contributions[contributions:eq(0)] = epsilon / vs -- Make positive when zero

  -- Normalize
  local normalizingFactor = contributions:sum(2):view(rs, 1)
  normalizingFactor = normalizingFactor + epsilon * torch.sign(normalizingFactor) -- Add stabilizer

  normalizingFactor[normalizingFactor:eq(0)] = epsilon -- Make positive when zero

  contributions:cdiv(normalizingFactor:expandAs(contributions))

  -- Now each row is the contribution that each input gave to the
  -- output corresponding to that row index.
  --
  -- Multiplying will give relevance propagations back to V.
  return torch.mm(R, contributions)
end

function relPropagateAdd(T, R)
  -- Get total contributions
  sumT = T[1]:clone()
  for i=1,#T do
    sumT:add(T[i])
  end

  sumT:add(epsilon * torch.sign(sumT))
  sumT[sumT:eq(0)] = epsilon -- Add stabilizer (make positive when zero)

  -- Normalize to sum to 1
  local normT = {}
  for i=1,#T do
    table.insert(normT, torch.cdiv(T[i], sumT))
  end

  -- Multiply to get proportional propagation
  local relevances = {}
  for i=1,#T do
    table.insert(relevances, torch.cmul(R, normT[i]))
  end

  return relevances
end
