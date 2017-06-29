require 'nn'
require 'nngraph'
require 'hdf5'
require 'optim'

require 'cutorch'
require 'cunn'

require 's2sa.data'
require 's2sa.models'
require 's2sa.model_utils'

torch.manualSeed(0)

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

function head_table(t, index)
  result = {}
  for i=1,index do
    table.insert(result, t[i])
  end
  return result
end

function generate_ideal_sentence(checkpoint, alphabet, starting_sentence, neuron, epochs, starting_counts)
  local model, opt = checkpoint[1], checkpoint[2]
  local encoder = model[1]

  local length = #starting_sentence

  print(length)

  encoder:replace(function(module)
    -- Replace instances of nn.LookupTable with similarly-sized Linear layer,
    -- so that we can backprop across it continuously
    if torch.typename(module) == 'nn.LookupTable' then
      local weight = module.weight
      local layer = nn.Linear(weight:size(1), weight:size(2), false)

      layer.weight = weight:t()

      return layer
    else
      return module
    end
  end)

  encoder = encoder:cuda()

  local alphabet_size = #alphabet

  print('Creating clones.')

  local encoder_clones = clone_many_times(encoder, length)

  print('Done.')

  --[[
  local all_params = torch.Tensor(length * alphabet_size + opt.rnn_size * 2 * opt.num_layers):uniform()
  local grad_params = torch.Tensor(length * alphabet_size + opt.rnn_size * 2 * opt.num_layers):zero()
  ]]
  local all_params = torch.Tensor(length * alphabet_size):zero():cuda() --uniform()
  local grad_params = torch.Tensor(length * alphabet_size):zero():cuda()

  local current_source = {}

  for t=1,length do
    table.insert(
      current_source,
      all_params:narrow(1, (t-1)*alphabet_size + 1, alphabet_size):view(1, alphabet_size)
    )
  end

  local token_probability_mask = torch.Tensor(alphabet_size)

  for i=1,alphabet_size do
    token_probability_mask[i] = starting_counts[i] or 0
  end

  token_probability_mask = token_probability_mask / token_probability_mask:sum()
  token_probability_mask = token_probability_mask:cuda()

  -- Start at a given sentence
  for t=1,length do
    current_source[t][1][starting_sentence[t]] = 10
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

  function opfunc(all_params)
    grad_params:zero()

    local softmax = {}
    for i=1,length do
      local layer = nn.Sequential()
      layer:add(nn.SoftMax())

      -- A Bayesian update on the MLE counts from the validation corpus
      local cmul = nn.CMul(alphabet_size)
      cmul.weight = token_probability_mask
      layer:add(cmul)

      layer:add(nn.Normalize(1))

      table.insert(softmax, layer:cuda())
    end

    -- Forward pass
    local rnn_state = first_hidden
    local encoder_inputs = {}
    for t=1,length do
      local encoder_input = {softmax[t]:forward(current_source[t])}
      append_table(encoder_input, rnn_state)
      encoder_inputs[t] = encoder_input
      rnn_state = encoder_clones[t]:forward(encoder_input)
    end

    local loss = -rnn_state[#rnn_state][1][neuron] -- Trying to maximize exactly this neuron

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
    last_hidden[#last_hidden][1][neuron] = -1

    -- Initialize.
    local rnn_state_gradients = {}
    rnn_state_gradients[length] = last_hidden

    for t=length,1,-1 do
      local encoder_input_gradient = encoder_clones[t]:backward(encoder_inputs[t], rnn_state_gradients[t])
      -- Get source gradients and copy into gradient array
      grad_params:narrow(1, 1 + (t-1)*alphabet_size, alphabet_size):copy(
        softmax[t]:backward(current_source[t], encoder_input_gradient[1])
      )
      -- Get RNN state gradients
      rnn_state_gradients[t-1] = slice_table(encoder_input_gradient, 2)
    end

    -- Copy initial RNN state gradients into gradient array
    --[[
    for i=1,2*opt.num_layers do
      grad_params:narrow(1, 1 + length*alphabet_size + (i-1)*opt.rnn_size, opt.rnn_size):copy(
        rnn_state_gradients[0][i]
      )
    end
    ]]

    print('Loss', loss)

    -- Get all the tokens in this sentence
    local str = ''
    for t=1,length do
      -- Get max
      n, index = torch.topk(softmax[t]:forward(current_source[t]), 1, true)
      print(n[1][1], alphabet[index[1][1]])
      -- Get word associated with max
      str = str .. ' ' .. alphabet[index[1][1]]
    end

    print('Current sentence:')
    print(str)

    return loss, grad_params
  end

  local optim_state = {learningRate = 0.1}
  for i=1,epochs do
    print('Epoch', i, 'of', epochs)
    optim.adam(opfunc, all_params, optim_state)
  end

  -- Get all the tokens in this sentence
  local tokens = {}
  for t=1,length do
    -- Get max
    n, index = torch.topk(current_source[t], 1, true)
    -- Get word associated with max
    table.insert(tokens, alphabet[index])
  end

  return tokens
end

function idx2key(file)
  local f = io.open(file,'r')
  local t = {}
  for line in f:lines() do
    local c = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(c, w)
    end
    t[tonumber(c[2])] = c[1]
  end
  return t
end

function invert_table(t)
  r = {}
  for k, v in ipairs(t) do
    r[v] = k
  end
  return r
end

function tokenize(line, inverse_alphabet)
  -- Tokenize the start line
  local tokens = {
    inverse_alphabet['<s>']
  }
  local k = 0
  for entry in line:gmatch'([^%s]+)' do
    table.insert(tokens,
      inverse_alphabet[entry] or inverse_alphabet['<unk>']
    )
  end
  table.insert(tokens, inverse_alphabet['end'])

  return tokens
end

function main()
  cmd = torch.CmdLine()

  cmd:option('-model', 'model.t7', 'Path to model .t7 file')
  cmd:option('-src_dict', '', 'Source dictionary')
  cmd:option('-neuron', 0, 'Output neuron to maximize')
  cmd:option('-length', 10, 'Token length of sequence to generate')
  cmd:option('-epochs', 100, 'Epochs to optimize for')
  cmd:option('-start_line', 0, 'Line from val/en.tok to start from')

  local opt = cmd:parse(arg)

  local alphabet = idx2key(opt.src_dict)
  local inverse_alphabet = invert_table(alphabet)
  local checkpoint = torch.load(opt.model)

  -- Get a starting sentence from the validation file
  local lines_file = io.open('/data/sls/scratch/abau/seq2seq-comparison/input-data/un/val/en.tok')

  local lines = {}
  while true do
    local line = lines_file:read("*line")
    if line == nil then break end
    table.insert(lines, tokenize(line, inverse_alphabet))
  end

  local counts = {}
  for i=1,#lines do
    for j=1,#lines[i] do
      counts[lines[i][j]] = (counts[lines[i][j]] or 0) + 1
    end
  end

  local tokens = generate_ideal_sentence(checkpoint, alphabet, head_table(lines[opt.start_line], opt.length), opt.neuron, opt.epochs, counts)

  local str = ''
  for i=1,#tokens do
    str = str .. ' ' .. tokens[i]
  end

  print(str)
end

main()
