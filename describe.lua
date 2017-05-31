-- Takes the same arguments as (evaluate), but instead of generating a translation
-- simply generate a table of encodings.
--
-- Run this on the same input file for multiple models in order to do a representation
-- comparison.
require 'cutorch'
local beam = require 's2sa.beam'

function main()
  beam.init(arg)
  local opt = beam.getOptions()

  assert(path.exists(opt.src_file), 'src_file does not exist')

  local file = io.open(opt.src_file, "r")

  local encodings = {}
  local total_token_length = 0

  -- Encode each line in the input sample file
  for line in file:lines() do
    encoding = beam.encode(line)
    table.insert(encodings, encoding[1]:cuda()) -- encoding[1] should be size_l x rnn_size
    total_token_length = total_token_length + encoding:size()[2]
  end

  -- Get the average
  mean = torch.Tensor(encodings[1]:size()[2]):zero():cuda()
  for i=1,#encodings do
    encoding = encodings[i]
    mean:add(1 / total_token_length, torch.sum(encoding, 1))
  end

  -- Get the stdev
  stdev = torch.Tensor(encodings[1]:size()[2]):zero():cuda()
  for i=1,#encodings do
    encoding = encodings[i]
    for j=1,encoding:size()[1] do
      stdev:add(1 / total_token_length, torch.pow(torch.add(encoding[j], -1, mean), 2))
    end
  end

  stdev:sqrt()

  -- Save the encodings
  torch.save(opt.output_file, {
    ['encodings'] = encodings,
    ['mean'] = mean,
    ['stdev'] = stdev,
    ['sample_length'] = total_token_length
  })
end

main()