-- Compute several statistics given two encoding distribution samples.
-- Currently computes:
--  - Covariance
--  - Correlation

function main()
  local opt = cmd:parse(arg)

  assert(path.exists(opt.A), 'model description A does not exist')
  assert(path.exists(opt.B), 'model description B does not exist')

  local A, B = torch.load(opt.A), torch.load(opt.B)
  local encodings_A, encodings_B = A['encodings'], B['encodings']
  local mean_A, mean_B = A['mean'], B['mean']
  local sigma_A, sigma_B = A['stdev'], B['stdev']
  local sample_length = A['sample_length'] -- should also == B['sample_length']

  -- Measure the rnn size by looking at the size of an
  -- encoding
  local rnn_size_A = encodings_A[1]:size()[2]
  local rnn_size_B = encodings_B[1]:size()[2]

  -- Initialize
  local E_AB = torch.Tensor(rnn_size_A, rnn_size_B):zero():cuda()

  -- For each line in the input sample,
  for i=1,#encodings_A do
    -- For each token in this line
    vectors_A, vectors_B = encodings_A[i]:cuda(), encodings_B[i]:cuda() -- 1 x rnn_size
    for j=1,#vectors_A do
      -- rnn_size x 1 X 1 x rnn_size = rnn_size x rnn_size
      -- Take the product and add it to our average
      E_AB:add(1 / sample_length, torch.mm(vectors_A[j]:transpose(1, 2), vectors_B[j]))
    end
  end

  -- Also take the product of the means
  local EA_EB = torch.mm(mean_A:transpose(1, 2), mean_B)

  local sigma = torch.mm(sigma_A:transpose(1, 2), sigma_B)

  local covariance = torch.csub(E_AB, EA_EB)
  local correlation = corch.cdiv(covariance, sigma)

  -- Save the encodings
  torch.save({
    ['correlation'] = correlation,
    ['covariance'] = covariance
  }, opt.out_file)
end

main()
