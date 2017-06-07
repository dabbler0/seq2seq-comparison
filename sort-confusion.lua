-- Compute several statistics given two encoding distribution samples.
-- Currently computes:
--  - Covariance
--  - Correlation
require 'cutorch'
require 'nn'
require 'cunn'

-- Confusion mse
function main()
  cmd = torch.CmdLine()

  cmd:option('-src_file', 'src_file', 'Source file')
  cmd:option('-desc_file', 'desc_file', 'Description file (output of describe.lua)')
  cmd:option('-comp_file', 'comp_file', 'Comparison file (output of compare.lua)')
  cmd:option('-out_file', 'out_file', 'Output file')

  local opt = cmd:parse(arg)

  local encodings = torch.load(opt.desc_file)
  local comparisons = torch.load(opt.comp_file)

  -- Sort mses
  local mses, confusing_indices = torch.sort(comparisons['backward_mse'], 1, true)

  -- Get most confusing index

  local all_outputs = {}
  for k=1,confusing_indices:size(1) do
    local confusing_index = confusing_indices[k]

    print('Sorting by confusing index', confusing_indices[0], mses[0])

    -- Construct the sentence encodings
    local activations_to_sort = torch.Tensor(indices:size(1))
    for i=1,#encodings do
      local line_encodings = encodings[i]
      local sentence_encoding = line_encodings[line_encodings:size[1]]
      activations_to_sort[i] = sentence_encoding[confusing_index]
    end

    -- Sort activations and indices
    local activations, indices = torch.sort(activations_to_sort[i], 1, true)
    local output = {}
    for i=1,indices:size(1) do
      table.insert(output, {
        ['index']=indices[i],
        ['activation']=activations[i]
      })
    end

    all_outputs[confusing_index] = {
      ['confusion']=mses[k]
      ['sorting']=output
    })
  end

  confusing_indices[0]

  -- Log some confusing sentences:
  lines = {}
  io.open()
  while true do
    local line = io.read("*line")
    if line == nil then
      break
    else
      table.insert(lines, line)
    end
  end

  -- Top five activated sentences:
  print('Confusing categories: HIGH:')
  for i=1,5 do
    print(output[i]['activation'], lines[output[i]['index']])
  end

  print('Confusing categories: LOW:')
  for i=#output,#output-5,-1 do
    print(output[i]['activation'], lines[output[i]['index']])
  end

  torch.save(opt.out_file, output)
end

main()
