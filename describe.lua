-- Takes the same arguments as (evaluate), but instead of generating a translation
-- simply generate a table of encodings.
--
-- Run this on the same input file for multiple models in order to do a representation
-- comparison.
require 'cutorch'

require 'nn'

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
    if encoding ~= nil then
      table.insert(encodings, nn.utils.recursiveType(encoding[1], 'torch.DoubleTensor')) -- encoding[1] should be size_l x rnn_size
      total_token_length = total_token_length + encoding:size()[2]
    else
      print('Skipping line because it is too long:')
      print(line)
    end
  end

  -- Save the encodings
  torch.save(opt.output_file, {
    ['encodings'] = encodings,
    ['sample_length'] = total_token_length
  })
end

main()
