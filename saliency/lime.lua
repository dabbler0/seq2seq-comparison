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

function lime(
    opt,
    alphabet,
    encoder_clones,
    normalizer,
    neuron,
    sentence,
    num_perturbations,
    perturbation_size)

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
      --all_params:narrow(1, 1 + length*alphabet_size + opt.rnn_size*(i-1), opt.rnn_size):view(1, opt.rnn_size)
      torch.Tensor(1, opt.rnn_size):zero():cuda()
    )
  end

  -- Softmax layer
  local softmax = {}
  for i=1,length do
    local layer = nn.Sequential()
    -- Softmax layer (currently unused in favor of Normalize)
    layer:add(nn.SoftMax())
    --layer:add(nn.Normalize(1))

    table.insert(softmax, layer:cuda())
  end

  -- Gradient-retrieval function
  function run_forward(all_params, length)
    -- Forward pass
    local rnn_state = first_hidden
    local perturbed_encodings = {}
    for t=1,length do
      local encoder_input = {softmax[t]:forward(current_source[t])}
      table.insert(perturbed_encodings, encoder_input[1])
      append_table(encoder_input, rnn_state)
      rnn_state = encoder_clones[t]:forward(encoder_input)
    end

    -- Compute normalized loss
    local loss = (
      rnn_state[#rnn_state][1][neuron] - mean[1][neuron]
    ) / stdev[1][neuron]

    return perturbed_encodings, loss --grad_params
  end

  local lime_data_inputs = {}
  local lime_data_outputs = {}

  -- Do several perturbations
  local length = #sentence
  for i=1,num_perturbations do
    -- Give all the other parameters a little bit of probability
    all_params:uniform() --:mul(perturbation_size / alphabet_size)
    -- all_params:zero() -- Zero it for softmax
    -- all_params:uniform()

    -- Start at a given sentence
    for t=1,length do
      current_source[t][1][sentence[t]] = perturbation_size -- perturbation_size
      -- e^x will be just a little bit more than all the other probabilities combined
    end

    -- Create the data point for LIME to regress from
    local perturbed_encodings, loss = run_forward(all_params, length)
    local input_data = torch.Tensor(length)
    for t=1,length do
      input_data[t] = perturbed_encodings[t][1][sentence[t]]
    end

    table.insert(lime_data_outputs, loss)
    table.insert(lime_data_inputs, input_data)
  end

  -- Put data points into the model for regression
  local input_matrix = torch.Tensor(num_perturbations, length)
  local output_matrix = torch.Tensor(num_perturbations)
  for t=1,num_perturbations do
    input_matrix[t] = lime_data_inputs[t]
    output_matrix[t] = lime_data_outputs[t]
  end

  -- Create the local linear model
  -- Projection should be length x 1
  local projection = torch.gels(output_matrix, input_matrix)

  print('regressed projection.')
  print(projection)

  -- Get affinity for each token in the sentence
  local affinity = {}
  for t=1,length do
    table.insert(affinity, projection[t][1])
  end

  return affinity
end
