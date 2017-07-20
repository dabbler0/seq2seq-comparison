-- Importance as measured by gradients at slightly permuted positions, as described here:
-- https://arxiv.org/pdf/1706.03825.pdf

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

function sensitivity_analysis(
    alphabet,
    model,
    normalizer,
    sentence)

  local opt, encoder_clones, lookup_table = model.opt, model.clones, model.lookup

  local alphabet_size = #alphabet
  local length = #sentence

  local mean, stdev = normalizer[1], normalizer[2]

  local affinities = torch.Tensor(length, opt.rnn_size):zero():cuda()

  local source_gradients = {}

  -- Construct beginning hidden state
  local first_hidden = {}
  for i=1,2*opt.num_layers do
    table.insert(
      first_hidden,
      --all_params:narrow(1, 1 + length*alphabet_size + opt.rnn_size*(i-1), opt.rnn_size):view(1, opt.rnn_size)
      torch.Tensor(1, opt.rnn_size):zero():cuda():expand(opt.rnn_size, opt.rnn_size)
    )
  end

  -- Forward pass
  local rnn_state = first_hidden
  local encoder_inputs = {}
  for t=1,length do
    local encoder_input = {
      lookup_table:forward(
        torch.Tensor{sentence[t]}:expand(opt.rnn_size)
      )
    }
    append_table(encoder_input, rnn_state)
    encoder_inputs[t] = encoder_input
    rnn_state = encoder_clones[t]:forward(encoder_input)
  end

  -- Backward pass

  -- Construct final gradient
  local last_hidden = {}
  for i=1,2*opt.num_layers-1 do
    table.insert(
      last_hidden,
      torch.zeros(opt.rnn_size, opt.rnn_size):cuda()
    )
  end

  -- Diagonal matrix of normalizers. This represents 500 batches, one for each dimensions,
  -- where dimension x is backpropagating relevance for the xth output
  table.insert(
    last_hidden,
    torch.diag(torch.cinv(stdev[1])):cuda()
  )

  -- Initialize.
  local rnn_state_gradients = {}
  rnn_state_gradients[length] = last_hidden

  for t=length,1,-1 do
    local encoder_input_gradient = encoder_clones[t]:backward(encoder_inputs[t], rnn_state_gradients[t])
    -- Get source gradients and copy into gradient array
    local final_gradient = encoder_input_gradient[1]

    affinities[t]:copy(final_gradient:pow(2):sum(2))

    -- Get RNN state gradients
    rnn_state_gradients[t-1] = slice_table(encoder_input_gradient, 2)
  end

  -- Get affinity for each token in the sentence
  local result_affinity = {}
  for t=1,length do
    table.insert(result_affinity, affinities[t])
  end

  return result_affinity
end

