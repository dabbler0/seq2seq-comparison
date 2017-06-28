require 'nn'
require 'nngraph'
require 'hdf5'
require 'optim'

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

function generate_ideal_sentence(checkpoint, alphabet, neuron, length, epochs)
  local model, opt = checkpoint[1], checkpoint[2]
  local encoder = model[1]

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

  local alphabet_size = #alphabet

  local encoder_clones = clone_many_times(encoder, length)

  --[[
  local all_params = torch.Tensor(length * alphabet_size + opt.rnn_size * 2 * opt.num_layers):uniform()
  local grad_params = torch.Tensor(length * alphabet_size + opt.rnn_size * 2 * opt.num_layers):zero()
  ]]
  local all_params = torch.Tensor(length * alphabet_size + 0 * opt.rnn_size * 2 * opt.num_layers):zero() --uniform()
  local grad_params = torch.Tensor(length * alphabet_size + 0 * opt.rnn_size * 2 * opt.num_layers):zero()

  local current_source = {}

  for i=1,length do
    table.insert(
      current_source,
      all_params:narrow(1, (i-1)*alphabet_size + 1, alphabet_size):view(1, alphabet_size)
    )
  end

  -- Construct beginning hidden state
  local first_hidden = {}
  for i=1,2*opt.num_layers do
    table.insert(
      first_hidden,
      --all_params:narrow(1, 1 + length*alphabet_size + opt.rnn_size*(i-1), opt.rnn_size):view(1, opt.rnn_size)
      torch.Tensor(1, opt.rnn_size):zero()
    )
  end

  function opfunc(all_params)
    grad_params:zero()

    local softmax = {}
    for i=1,length do
      table.insert(softmax, nn.SoftMax())
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
        torch.zeros(1, opt.rnn_size)
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

  local optim_state = {learningRate = 0.05}
  for i=1,epochs do
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

function main()
  cmd = torch.CmdLine()

  cmd:option('-model', 'model.t7', 'Path to model .t7 file')
  cmd:option('-src_dict', '', 'Source dictionary')
  cmd:option('-neuron', 0, 'Output neuron to maximize')
  cmd:option('-length', 10, 'Token length of sequence to generate')
  cmd:option('-epochs', 100, 'Epochs to optimize for')

  local opt = cmd:parse(arg)

  local alphabet = idx2key(opt.src_dict)
  local checkpoint = torch.load(opt.model)

  local tokens = generate_ideal_sentence(checkpoint, alphabet, opt.neuron, opt.length, opt.epochs)

  local str = ''
  for i=1,#tokens do
    str = str .. ' ' .. tokens[i]
  end

  print(str)
end

main()
