-- Importance measured by local linear approximation
-- a simplified version of LIME: https://arxiv.org/pdf/1602.04938v1.pdf

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

local source = torch.CudaTensor()
local input_matrix = torch.CudaTensor()
local output_matrix = torch.CudaTensor()

function lime(
    alphabet,
    model,
    normalizer,
    sentence,
    num_perturbations,
    perturbation_size)

  local opt, encoder_clones, linear = model.opt, model.clones, model.linear

  local mean, stdev = normalizer[1], normalizer[2]

  source:resize(1, #alphabet)

  -- Construct beginning hidden state
  local first_hidden = {}
  for i=1,2*opt.num_layers do
    table.insert(
      first_hidden,
      torch.Tensor(1, opt.rnn_size):zero():cuda()
    )
  end

  -- Softmax layer
  local softmax = nn.Sequential()
  softmax:add(nn.SoftMax())
  softmax:cuda()

  -- Gradient-retrieval function
  function run_forward(input_vector, output_vector)
    -- Forward pass
    local rnn_state = first_hidden
    local perturbed_encodings
    for t=1,#sentence do
      source:uniform()
      source[1][sentence[t]] = perturbation_size * 2 * torch.uniform()

      local softmaxed = softmax:forward(source)

      local encoder_input = {
        linear:forward(
          softmax:forward(source)
        )
      }

      -- Record amount of perturbation
      input_vector[t] = softmaxed[1][sentence[t]]

      append_table(encoder_input, rnn_state)
      rnn_state = encoder_clones[t]:forward(encoder_input)
    end

    -- Compute and record normalized loss
    output_vector:copy(rnn_state[#rnn_state][1]):csub(mean[1]):cdiv(stdev[1])

    return perturbed_encodings, loss
  end

  local lime_data_inputs = {}
  local lime_data_outputs = {}

  -- Do several perturbations
  input_matrix:resize(num_perturbations, #sentence)
  output_matrix:resize(num_perturbations, opt.rnn_size)

  for i=1,num_perturbations do
    -- Create the data point for LIME to regress from
    run_forward(input_matrix[i], output_matrix[i])
  end

  -- Create the local linear model
  -- Projection should be length x 1
  local projection = torch.inverse(input_matrix:t() * input_matrix) * input_matrix:t() * output_matrix

  -- Get affinity for each token in the sentence
  local affinity = {}
  for t=1,#sentence do
    table.insert(affinity, projection[t])
  end

  return affinity
end
