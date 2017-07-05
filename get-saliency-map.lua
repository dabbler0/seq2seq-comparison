require 'nn'
require 'nngraph'
require 'hdf5'
require 'optim'

require 'cutorch'
require 'cunn'

require 's2sa.data'
require 's2sa.models'
require 's2sa.model_utils'

require 'json'

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

function create_encoder_clones(checkpoint, max_len)
  local encoder, opt = checkpoint[1][1], checkpoint[2]
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

  return clone_many_times(encoder, max_len), opt
end

function get_all_saliencies(encoder_clones, normalizer, opt, alphabet, neuron, sentence, num_perturbations, perturbation_size)
  local alphabet_size = #alphabet
  local length = #sentence

  local mean, stdev = normalizer[1], normalizer[2]

  --[[
  local all_params = torch.Tensor(length * alphabet_size + opt.rnn_size * 2 * opt.num_layers):uniform()
  local grad_params = torch.Tensor(length * alphabet_size + opt.rnn_size * 2 * opt.num_layers):zero()
  ]]
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
      --layer:add(nn.SoftMax())
      layer:add(nn.Normalize(1))

      -- A Bayesian update on the MLE counts from the validation corpus
      -- (Currently unused)
      --[[
      local cmul = nn.CMul(alphabet_size)
      cmul.weight = token_probability_mask
      layer:add(cmul)

      layer:add(nn.Normalize(1))
      ]]

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
  for length=1,#sentence do
    io.stderr:write(string.format('Processing length %d of %d\n', length, #sentence))

    for i=1,num_perturbations do
      -- Give all the other parameters a little bit of probability
      all_params:uniform():mul(perturbation_size / alphabet_size)
      -- all_params:zero() -- Zero it for softmax
      -- all_params:uniform()

      -- Start at a given sentence
      for t=1,length do
        current_source[t][1][sentence[t]] = 1 -- perturbation_size
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

    -- Do a "perfect" run to get true activation
    all_params:zero()
    for t=1,length do
      current_source[t][1][sentence[t]] = 1
    end
    local loss = get_gradient(all_params, length, true)
    table.insert(activations, loss)

    affinities[length] = affinity
  end

  return activations, affinities
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
  table.insert(tokens, inverse_alphabet['</s>'])

  return tokens
end

function main()
  cmd = torch.CmdLine()

  cmd:option('-model_list', '', 'List of models with alphabets (alternating lines)')
  cmd:option('-max_len', 30, 'Maximum length')
  cmd:option('-num_perturbations', 3, 'Number of perturbations over which to average')

  local opt = cmd:parse(arg)

  local models = {}
  local opts = {}
  local alphabets = {}
  local inverse_alphabets = {}
  local normalizers = {}

  io.stderr:write('Opening model list ' .. opt.model_list .. '\n')
  local model_file = io.open(opt.model_list)
  while true do
    local model_name = model_file:read("*line")
    if model_name == nil then break end
    local model_key = model_name:match("%a%a%-%a%a%-%d")
    io.stderr:write('Loading ' .. model_name .. ' as ' .. model_key .. '\n')
    models[model_key], opts[model_key] = create_encoder_clones(torch.load(model_name), opt.max_len)

    local dict_name = model_file:read("*line")
    if dict_name == nil then break end
    alphabets[model_key] = idx2key(dict_name)
    inverse_alphabets[model_key] = invert_table(alphabets[model_key])

    io.stderr:write('Computing normalizer...\n')
    local desc_name = model_file:read("*line")
    if desc_name == nil then break end

    -- Collect encodings
    local encodings = torch.load(desc_name)['encodings']
    local concatenated = torch.cat(encodings, 1):cuda()

    -- Get mean
    local mean = concatenated:mean(1)
    concatenated:csub(mean:view(1, concatenated:size(2)):expandAs(concatenated))

    -- Get stdev
    local stdev = concatenated:pow(2):mean(1):sqrt()
    normalizers[model_key] = {mean, stdev}
  end
  model_file:close()

  io.stderr:write('Loaded all models.\n')

  while true do
    local network = io.read()
    local neuron = tonumber(io.read()) + 1
    local perturbation_size = tonumber(io.read())
    local sentence = tokenize(io.read(), inverse_alphabets[network])

    local backward_tokens = {}

    for i=1,#sentence do
      table.insert(backward_tokens, alphabets[network][sentence[i]])
    end

    local activations, saliencies =
      get_all_saliencies(models[network], normalizers[network], opts[network], alphabets[network], neuron, sentence, opt.num_perturbations, perturbation_size)

    print(json.encode({
      ['activations'] = activations,
      ['saliencies'] = saliencies,
      ['tokens'] = backward_tokens
    }))
  end
end

main()
