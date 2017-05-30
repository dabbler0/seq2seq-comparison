-- Compute several statistics given two encoding distribution samples.
-- Currently computes:
--  - Correlation

function main()
  local opt = cmd:parse(arg)

  assert(path.exists(opt.A), 'model description A does not exist')
  assert(path.exists(opt.B), 'model description B does not exist')

  local A, B = torch.load(opt.A), torch.load(opt.B)
  local encodings_A, encodings_B = A['encodings'], B['encodings']
  local mean_A, mean_B = A['mean'], B['mean']
  local sample_length = A['sample_length'] -- should also == B['sample_length']

  local E_AB = 

  -- For each line in the input sample,
  for i=1,#encodings_A do
    -- For each token in this line
    vectors_A, vectors_B = encodings_A[i], encodings_B[i]
    for j=1,#vectors_A do
      E_AB:add(1 / sample_length, )
    end

    -- Take the product and add it to our average
  end

  -- Save the encodings
  torch.save(encodings, out_file)
end

main()
