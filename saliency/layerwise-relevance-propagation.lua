-- Layer-wise relevance propagation on a network.
-- To be called immediately after a forward pass has been done.
function LRP(gmodule, R)
  local relevancies = {}
  local agenda = {}

  -- Topological sort the graph nodes by dfs
  local toposorted = {}
  local visited = {}

  function dfs(node)
    if visited[node] ~= nil then return end
    visited[node] = true

    for dependency, t in ipairs(node.data.reverseMap) do
      -- Add everyone we depend on
      dfs(dependency)
    end

    -- Add us
    table.insert(toposorted, node.data)
  end

  -- Initialize the output relevance
  dfs(gmodule.outnode)
  relevancies[gmodule.outnode.data] = R

  -- We are now guaranteed that if we need X to compute Y,
  -- then X comes before Y in the toposort. So we can
  -- iterate through the toposort and compute relevancies.
  for i=1,#toposorted do
    local node = toposorted[i]
    local relevance = relevancies[node]

    -- Propagate input relevance
    local input_relevance = relPropagate(node, relevance)

    -- Accumulate.
    if #node.mapindex == 1 then

      -- Special case for when there is only one
      -- input, because then the input_relevance is not a table

      if relevancies[node.mapindex[0]] == nil then
        relevancies[node.mapindex[0]] = input_relevance
      else
        relevancies[node.mapindex[0]] = relevancies[node.mapindex[0]] + input_relevance
      end

    else

      -- Otherwise use input_relevance as a table
      for j=1,#node.mapindex do
        if relevancies[node.mapindex[j]] == nil then
          relevancies[node.mapindex[j]] = input_relevance[j]
        else
          relevancies[node.mapindex[j]] = relevancies[node.mapindex[j]] + input_relevance[j]
        end
      end

    end

  end

  -- Return the input relevance
  return relevancies[gmodule.innode.data]
end

-- Propagate relevance through a single
-- gModule node.
--
-- Supports all the node types that appear in the
-- encoder LSTM.
function relPropagate(node_data, R)
  local I = node_data.module.input
  local module = node_data.module

  -- For CMulTable, determine which one is the "gate",
  -- and propagate back only through the non-gate.
  if torch.typename(module) == 'nn.CMulTable' then
    local input_nodes = node_data.mapindex

    -- We currently don't support MulTable for more than 2
    -- factors
    if #input_nodes > 2 then
      return nil
    end

    local result_table = {}
    for i=1,#input_nodes do
      if torch.typename(input_nodes[i].module) == 'nn.Sigmoid' then
        table.insert(result_table, 0)
      else
        table.insert(result_table, 1)
      end
    end

  -- For CAddTable, use relPropagateAdd
  elseif torch.typename(module) == 'nn.CAddTable' then
    return relPropagateAdd(I, R)

  -- For Linear, use relPropagateLinear
  elseif torch.typename(module) == 'nn.Linear' then
    return relPropagateLinear(I, module.weight, R)

  -- Reshape and split table should
  -- rearrange the relevance exactly backwards
  -- as the inputs were rearranged forward
  elseif torch.typename(module) == 'nn.Reshape' then
    return torch.viewAs(R, I)
  elseif torch.typename(module) == 'nn.SplitTable' then
    return torch.concatenate(R, module.dimension)

  -- For LookupTable, pass through.
  elseif torch.typename(module) == 'nn.LookupTable' then
    return R

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
local epsilon = 1e-6
function relPropagateLinear(V, W, R)
  local rs, vs = R:size(1), V:size(1)

  -- Get contributions
  local contributions = (V:view(1, vs):expandAs(rs, vs) * W)
  contributions = contributions + epsilon * torch.sign(contributions) / vs -- Add stabilizer

  -- Normalize
  local normalizingFactor = contributions:mean(2):view(rs)
  normalizingFactor = normalizingFactor + epsilon * torch.sign(normalizingFactor) -- Add stabilizer
  contributions:cdiv(normalizingFactor:expandAs(rs, vs))

  -- Now each row is the contribution that each input gave to the
  -- output corresponding to that row index.
  --
  -- Multiplying will give relevance propagations back to V.
  return torch.mm(R, contributions)
end

function relPropagateAdd(T, R)
  -- Get total contributions
  sumT = T[1]:clone():zero()
  for i=1,#T do
    sumT:cadd(T[i])
  end

  -- Normalize to sum to 1
  local normT = {}
  for i=1,#T do
    table.insert(normT, torch.cdiv(T[i], sumT))
  end

  -- Multiply to get proportional propagation
  local relevances = {}
  for i=1,#T do
    table.insert(relevances, R * normT[i])
  end

  return relevances
end
