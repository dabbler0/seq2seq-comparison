--[[

Compute all of the linear regressions between the given descriptions and their MSEs.

]]
require 'cutorch'
require 'nn'
require 'cunn'
require 'json'
require 'optim'

-- Normalize a sample to have mean 0
function normalize_mean(X)
  local N = X:clone()

  local w, h = N:size(1), N:size(2)

  -- First, get units normalized to mean 0 and standard deviation 1
  local M = N:sum(1):mul(1 / w)
  N:csub(
    M:view(1, h):expand(w, h)
  )

  return N
end

-- Normalize a sample to have standard deviation 1
function normalize_stdev(N)
  local Q = N:clone()

  local w, h = N:size(1), N:size(2)

  local S = Q:clone():pow(2):sum(1):mul(1 / w):sqrt()
  Q:cdiv(
    S:view(1, h):expand(w, h)
  )

  return Q
end

function train_nonlinear_predictor(A, hiddens, B, batch_size, epochs)
  -- Construct simple one-layer neural network
  local network = nn.Sequential()
  network:add(nn.Linear(A:size(1), hiddens))
  network:add(nn.Tanh())
  network:add(nn.Linear(hiddens, B:size(1)))

  -- MSE criterion
  local criterion = nn.MSECriterion()

  -- Get parameters
  local params, grad_params = network.getParameters()

  -- Do several SGD passes
  local full_epochs = 0
  local order = torch.randperm(A:size(1))
  local index = 1
  function optim_function(params)
    network:zeroGradParameters()

    -- Start new epoch if necessary
    if index + batch_size > #order then
      -- Put items in a new random order
      -- and start indexing over.
      order = torch.randperm(A:size(1))
      index = 1
      full_epochs = full_epochs + 1
    end

    -- Assemble the current minibatch
    local batch_input = torch.Tensor(batch_size, A:size(2))
    local batch_output = torch.Tensor(batch_size, B:size(2))

    for i=1,batch_size do
      batch_input[i] = A[order[index]]
      batch_output[i] = B[order[index]]
      index = index + 1
    end

    -- Feed the minibatch (forward)
    local prediction = network:forward(batch_input)
    local loss = criterion:forward(prediction, batch_output)

    -- (backward)
    local output_grad = criterion:backward(prediction, batch_output)
    local input_grad = network:backward(batch_input, loss_grad)

    -- Return loss
    return loss, grad_params
  end

  local optim_state = {
    learning_rate = 0.05
  }
  while full_epochs < epochs do
    optim.adam(optim_function, optim_state)
  end

  -- Do one forward pass to get error
  local total_prediction = network:forward(A)

  (total_prediction - B):pow(2):mean(1):sqrt()

  return loss
end

function covariance_matrix(A, B) -- samples x sizeA, samples x sizeB
  -- Assume these are normalized, so return:
  return torch.mm(A:t(), B):mul(1 / A:size(1)) -- sizeA x sizeB
end

function main()
  cmd = torch.CmdLine()

  cmd:option('-desc_list', 'desc_list', 'File containing a list of description files (one per line)')
  cmd:option('-out_file', 'out_file', 'Output file')

  local opt = cmd:parse(arg)

  assert(path.exists(opt.desc_list), 'Description file list does not exist.')

  print('Reading out description file.')

  -- Read out lines of this file
  local filenames = {}
  local desc_file = io.open(opt.desc_list)
  while true do
    local line = desc_file:read("*line")
    if line == nil then break end
    table.insert(filenames, line)
  end
  io.close()

  print('Done.')

  -- For each file, get its described encoding,
  -- and extract sentence encodings, and normalize them
  -- to have standard deviation 1 and mean 0.
  local encodings = {}
  local indices_used = {}
  for i=1,#filenames do
    local filename = filenames[i]
    print('Reading out:', filename)
    local data = torch.load(filename)
    local all_embeddings = data['encodings']
    local sample_length = data['sample_length']

    print('Done.')
    print('Normalizing:', filename)

    --local sample_length = #all_embeddings
    local rnn_size = all_embeddings[1]:size(2)

    local new_encoding = torch.Tensor(sample_length, rnn_size):zero()

    local k = 1
    for i=1,#all_embeddings do
      local vector = all_embeddings[i]

      for j=1,vector:size(1) do
        local element = vector[j]
        new_encoding[k] = nn.utils.recursiveType(element, 'torch.DoubleTensor')
        k = k + 1
      end
    end

    -- Normalize
    local normalized_new_encoding = normalize_stdev(normalize_mean(new_encoding))

    encodings[filename] = normalized_new_encoding
  end

  -- Now create the entire LSLR MSE table.
  comparison_table = {}
  for name_A, encoding_A in pairs(encodings) do
    mapping_table = {}
    for name_B, encoding_B in pairs(encodings) do
      if name_A ~= name_B then
        print('COMPARING:', name_A, name_B)

        -- Train nonlinear predictor
        local mse = train_nonlinear_predictor(
          encoding_A,
          500, -- Usually equal to the number of dimensions
          encoding_B,
          50,
          13
        )

        -- Extract MSE into a Lua table
        -- and put it into the big mapping table
        mapping_table[name_B] = mse:totable()
      end
    end
    comparison_table[name_A] = mapping_table
  end

  -- Save the measurements as JSON, for
  -- visualization by a JS frontend.
  print('Writing...')
  local out_file = io.open(opt.out_file, 'w')
  out_file:write(json.encode(comparison_table))
  out_file:close()
  print('Done.')

end

main()
