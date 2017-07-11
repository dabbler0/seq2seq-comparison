-- Compute saliencies by four methods:
--  - Word erasure
--  - Layer-wise Relevance Propagation
--  - SmoothGrad
--  - LIME

require 'nn'
require 'nngraph'
require 'hdf5'
require 'optim'

require 'cutorch'
require 'cunn'

require 's2sa.data'
require 's2sa.models'
require 's2sa.model_utils'

require 'smooth-grad'
require 'layerwise-relevance-propagation'
require 'lime'
require 'erasure'

require 'json'

torch.manualSeed(0)

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
  encoder:evaluate()

  return clone_many_times(encoder, max_len), opt
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

  -- Don't insert the end-of-sentence token,
  -- as we're trying to examine the activation right here

  --table.insert(tokens, inverse_alphabet['</s>'])

  return tokens
end

function get_all_saliencies(
    opt,
    alphabet,
    encoder_clones,
    normalizer,
    neuron,
    sentence,
    num_perturbations,
    perturbation_size)

  -- Find raw activations.

  -- inputs
  local one_hot_inputs = {}
  for t=1,#sentence do
    one_hot_inputs[t] = torch.Tensor(1, #alphabet):zero():cuda()
    one_hot_inputs[t][1][sentence[t]] = 1
  end

  -- rnn state
  local rnn_state = {}
  for i=1,2*opt.num_layers do
    table.insert(rnn_state, torch.Tensor(1, opt.rnn_size):zero():cuda())
  end

  local activations = {}
  for t=1,#sentence do
    local inp = {one_hot_inputs[t]}
    append_table(inp, rnn_state)
    rnn_state = encoder_clones[t]:forward(inp)
    print('mean', normalizer[1][1][neuron])
    print('stdev', normalizer[2][1][neuron])
    print('activation', rnn_state[#rnn_state][1][neuron])
    activations[t] = (rnn_state[#rnn_state][1][neuron] - normalizer[1][1][neuron]) / normalizer[2][1][neuron]
  end

  -- SmoothGrad saliency
  local smooth_grad_saliency = smooth_grad(
      opt,
      alphabet,
      encoder_clones,
      normalizer,
      neuron,
      sentence,
      num_perturbations,
      perturbation_size)

  -- LRP saliency
  local layerwise_relevance_saliency = LRP_saliency(opt,
      alphabet,
      encoder_clones,
      normalizer,
      neuron,
      sentence)

  local lime_saliency = lime(opt,
      alphabet,
      encoder_clones,
      normalizer,
      neuron,
      sentence,
      num_perturbations * 10, -- Lime requires many more perturbations than SmoothGrad
      perturbation_size
  )

  local erasure_saliency = erasure(opt,
      alphabet,
      encoder_clones,
      normalizer,
      neuron,
      sentence
  )

  return {
    ['sgrad'] = smooth_grad_saliency,
    ['lrp'] = layerwise_relevance_saliency,
    ['lime'] = lime_saliency,
    ['erasure'] = erasure_saliency,
    ['activat'] = activations
  }

end

function main()
  cmd = torch.CmdLine()

  cmd:option('-model_list', '', 'List of models with alphabets (alternating lines)')
  cmd:option('-max_len', 30, 'Maximum length')
  cmd:option('-num_perturbations', 3, 'Number of perturbations over which to average')
  cmd:option('-perturbation_size', 11, 'Number of perturbations over which to average')

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
    local sentence = tokenize(io.read(), inverse_alphabets[network])

    local backward_tokens = {}

    for i=1,#sentence do
      table.insert(backward_tokens, alphabets[network][sentence[i]])
    end

    local saliencies = get_all_saliencies(
      opts[network],
      alphabets[network],
      models[network],
      normalizers[network],
      neuron,
      sentence,
      opt.num_perturbations,
      opt.perturbation_size
    )

    for k,p in pairs(saliencies) do
      str = k
      for i=1,#p do
        str = str .. '\t' .. string.format('%.3f', p[i])
      end
      print(str)
    end

    --print(json.encode(saliencies))
  end
end

main()
