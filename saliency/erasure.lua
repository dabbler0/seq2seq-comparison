-- Importance measured by word erasure:
-- implemented from https://arxiv.org/pdf/1612.08220.pdf

function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

function slice_table(t, index)
  result = {}
  for i=index,#t do
    table.insert(result, t[i])
  end
  return result
end

function erasure(
    opt,
    alphabet,
    encoder_clones,
    normalizer,
    sentence)

  local alphabet_size = #alphabet
  local length = #sentence

  local mean, stdev = normalizer[1], normalizer[2]

  local all_params = torch.Tensor(length * alphabet_size):zero():cuda() --uniform()

  local current_source = {}

  for t=1,length do
    table.insert(
      current_source,
      all_params:narrow(1, (t-1)*alphabet_size + 1, alphabet_size):view(1, alphabet_size)
    )
  end

  local source_gradients = {}

  -- Construct beginning hidden state
  local first_hidden = {}
  for i=1,2*opt.num_layers do
    table.insert(
      first_hidden,
      torch.Tensor(1, opt.rnn_size):zero():cuda()
    )
  end

  -- Gradient-retrieval function
  function run_forward(all_params, length, skip)
    -- Forward pass
    local rnn_state = first_hidden
    for t=1,length-1 do
      -- Skip the given token
      if t >= skip then t = t + 1 end

      local encoder_input = {current_source[t]}
      append_table(encoder_input, rnn_state)
      rnn_state = encoder_clones[1]:forward(encoder_input)
    end

    -- Return entire output vector
    -- as a 1d tensor
    return (rnn_state[#rnn_state][1] - mean[1]):cdiv(stdev[1])
  end

  local lime_data_inputs = {}
  local lime_data_outputs = {}

  all_params:zero()

  -- Start at a given sentence
  for t=1,length do
    current_source[t][1][sentence[t]] = 1
  end

  -- Do several perturbations
  local length = #sentence
  local results = {}
  for t=1,length+1 do
    results[t] = run_forward(all_params, length, t):clone()
  end

  local reference = results[#results]

  -- Get affinity for each token in the sentence
  local affinity = {}
  for t=1,length do
    table.insert(affinity, (results[t] - reference):cdiv(reference))
  end

  return affinity
end
