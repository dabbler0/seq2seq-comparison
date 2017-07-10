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

function smooth_grad(
    opt,
    alphabet,
    encoder_clones,
    normalizer,
    neuron,
    sentence,
    num_perturbations,
    perturbation_size)


  print(normalizer)

  -- Default arguments
  if num_perturbations == nil then num_perturbations = 3 end
  if perturbation_size == nil then perturbation_size = 11 end

  local alphabet_size = #alphabet
  local length = #sentence

  local mean, stdev = normalizer[1], normalizer[2]

  local all_params = torch.Tensor(length * alphabet_size):zero():cuda() --uniform()
  local grad_params = torch.Tensor(length * alphabet_size):zero():cuda()
  local cumulative_gradients = torch.Tensor(length * alphabet_size):zero():cuda()

  local current_source = {}

  for t=1,length do
    table.insert(
      current_source,
      all_params:narrow(1, (t-1)*alphabet_size + 1, alphabet_size):view(1, alphabet_size)
    )
  end

  local source_gradients = {}

  for t=1,length do
    table.insert(
      source_gradients,
      cumulative_gradients:narrow(1, (t-1)*alphabet_size + 1, alphabet_size):view(1, alphabet_size)
    )
  end

  -- Construct beginning hidden state
  local first_hidden = {}
  for i=1,2*opt.num_layers do
    table.insert(
      first_hidden,
      --all_params:narrow(1, 1 + length*alphabet_size + opt.rnn_size*(i-1), opt.rnn_size):view(1, opt.rnn_size)
      torch.Tensor(1, opt.rnn_size):zero():cuda()
    )
  end

  -- Gradient-retrieval function
  function get_gradient(all_params, length, perfect)
    if perfect == nil then perfect = false end

    grad_params:zero()

    local softmax = {}
    for i=1,length do
      local layer = nn.Sequential()
      -- Softmax layer (currently unused in favor of Normalize)
      layer:add(nn.SoftMax())
      --layer:add(nn.Normalize(1))

      table.insert(softmax, layer:cuda())
    end

    -- Forward pass
    local rnn_state = first_hidden
    local encoder_inputs = {}
    for t=1,length do
      local encoder_input = nil
      if perfect then
        encoder_input = {current_source[t]}
      else
        encoder_input = {softmax[t]:forward(current_source[t])}
      end
      append_table(encoder_input, rnn_state)
      encoder_inputs[t] = encoder_input
      rnn_state = encoder_clones[t]:forward(encoder_input)
    end

    -- Compute normalized loss
    local loss = (
      rnn_state[#rnn_state][1][neuron] - mean[1][neuron]
    ) / stdev[1][neuron]

    -- Backward pass

    -- Construct final gradient
    local last_hidden = {}
    for i=1,2*opt.num_layers do
      table.insert(
        last_hidden,
        torch.zeros(1, opt.rnn_size):cuda()
      )
    end

    -- Trying to maximize exactly this neuron
    last_hidden[#last_hidden][1][neuron] = 1 / stdev[1][neuron]

    -- Initialize.
    local rnn_state_gradients = {}
    rnn_state_gradients[length] = last_hidden

    for t=length,1,-1 do
      local encoder_input_gradient = encoder_clones[t]:backward(encoder_inputs[t], rnn_state_gradients[t])
      -- Get source gradients and copy into gradient array
      if perfect then
        grad_params:narrow(1, 1 + (t-1)*alphabet_size, alphabet_size):copy(
          encoder_input_gradient[1]
        )
      else
        grad_params:narrow(1, 1 + (t-1)*alphabet_size, alphabet_size):copy(
          softmax[t]:backward(current_source[t], encoder_input_gradient[1])
        )
      end
      -- Get RNN state gradients
      rnn_state_gradients[t-1] = slice_table(encoder_input_gradient, 2)
    end

    return loss --grad_params
  end

  local saliency_maps = {}

  all_params:zero()
  grad_params:zero()
  cumulative_gradients:zero()

  local affinities = {}
  local activations = {}

  -- Do several perturbations
  local length = #sentence
  for i=1,num_perturbations do
    -- Give all the other parameters a little bit of probability
    all_params:uniform() --:mul(perturbation_size / alphabet_size)
    -- all_params:zero() -- Zero it for softmax
    -- all_params:uniform()

    -- Start at a given sentence
    for t=1,length do
      current_source[t][1][sentence[t]] = perturbation_size
      -- e^x will be just a little bit more than all the other probabilities combined
    end

    get_gradient(all_params, length)
    cumulative_gradients:add(grad_params)
  end

  -- Average
  cumulative_gradients:div(num_perturbations)

  -- Get affinity for each token in the sentence
  local affinity = {}
  for t=1,length do
    table.insert(affinity, source_gradients[t][1][sentence[t]])
  end

  return affinity
end

