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

function get_all_saliencies(neuron, checkpoint, alphabet, sentences, num_perturbations, starting_counts)
  local model, opt = checkpoint[1], checkpoint[2]
  local encoder = model[1]

  local max_len = 0
  for s_idx, sentence in ipairs(sentences) do
    if #sentence > max_len then max_len = #sentence end
  end

  print(max_len)

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

  local encoder_clones = clone_many_times(encoder, max_len)

  print('Done.')

  --[[
  local all_params = torch.Tensor(length * alphabet_size + opt.rnn_size * 2 * opt.num_layers):uniform()
  local grad_params = torch.Tensor(length * alphabet_size + opt.rnn_size * 2 * opt.num_layers):zero()
  ]]
  local all_params = torch.Tensor(max_len * alphabet_size):zero():cuda() --uniform()
  local grad_params = torch.Tensor(max_len * alphabet_size):zero():cuda()
  local cumulative_gradients = torch.Tensor(max_len * alphabet_size):zero():cuda()

  local current_source = {}

  for t=1,max_len do
    table.insert(
      current_source,
      all_params:narrow(1, (t-1)*alphabet_size + 1, alphabet_size):view(1, alphabet_size)
    )
  end

  local source_gradients = {}

  for t=1,max_len do
    table.insert(
      source_gradients,
      cumulative_gradients:narrow(1, (t-1)*alphabet_size + 1, alphabet_size):view(1, alphabet_size)
    )
  end

  -- Token probability mask (currently unused)
  local token_probability_mask = torch.Tensor(alphabet_size)

  for i=1,alphabet_size do
    token_probability_mask[i] = starting_counts[i] or 0
  end

  token_probability_mask = token_probability_mask / token_probability_mask:sum()
  token_probability_mask = token_probability_mask:cuda()

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
  function get_gradient(all_params, length)
    grad_params:zero()

    local softmax = {}
    for i=1,max_len do
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
      local encoder_input = {softmax[t]:forward(current_source[t])}
      append_table(encoder_input, rnn_state)
      encoder_inputs[t] = encoder_input
      rnn_state = encoder_clones[t]:forward(encoder_input)
    end

    local loss = rnn_state[#rnn_state][1][neuron] -- Trying to maximize exactly this neuron

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
    last_hidden[#last_hidden][1][neuron] = 1

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

    return loss --grad_params
  end

  local saliency_maps = {}

  for s_idx, sentence in ipairs(sentences) do
    print('Processing sentence', s_idx)
    -- Zero out everything
    all_params:zero()
    grad_params:zero()
    cumulative_gradients:zero()

    local affinities = {}
    local activations = {}

    -- Do several perturbations
    for length=1,#sentence do
      print('Processing length', length, 'of', #sentence)
      for i=1,num_perturbations do
        -- Give all the other parameters a little bit of probability
        all_params:uniform():mul(1 / alphabet_size)

        -- Start at a given sentence
        for t=1,length do
          current_source[t][1][sentence[t]] = 1
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
      local loss = get_gradient(all_params, length)
      table.insert(activations, loss)
      print(loss)

      for t, a in ipairs(affinity) do
        print(alphabet[sentence[t]], a, activations[t])
      end

      affinities[length] = affinity
    end

    saliency_maps[s_idx] = affinities
  end

  return saliency_maps
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
  cmd:option('-length', 10, 'Number of lines to analyze')
  cmd:option('-num_perturbations', 50, 'Number of perturbations over which to average')

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
    local tokens = tokenize(line, inverse_alphabet)
    if #tokens < 250 then
      table.insert(lines, tokens)
    end
  end

  local counts = {}
  for i=1,#lines do
    for j=1,#lines[i] do
      counts[lines[i][j]] = (counts[lines[i][j]] or 0) + 1
    end
  end

  local saliencies = get_all_saliencies(opt.neuron + 1, checkpoint, alphabet, {lines[opt.length]}, opt.num_perturbations, counts)

  print('Writing saliency map...')
  f = io.open('saliency-map.json', 'w')
  f:write(json.encode(saliencies))
  f:close()
end

main()
