-- Compute several statistics given two encoding distribution samples.
-- Currently computes:
--  - Covariance
--  - Correlation
require 'cutorch'

function main()
  cmd = torch.CmdLine()

  cmd:option('-A', 'A', 'model A')
  cmd:option('-B', 'B', 'model B')
  cmd:option('-out_file', 'out_file', 'output file')

  local opt = cmd:parse(arg)

  assert(path.exists(opt.A), 'model description A does not exist')
  assert(path.exists(opt.B), 'model description B does not exist')

  -- 1. Correlation and covariance matrices
  --
  local A, B = torch.load(opt.A), torch.load(opt.B)
  local encodings_A_raw, encodings_B_raw = A['encodings'], B['encodings']
  local mean_A, mean_B = A['mean'], B['mean']
  local sigma_A, sigma_B = A['stdev'], B['stdev']
  local sample_length = A['sample_length'] -- should also == B['sample_length']

  -- Measure the rnn size by looking at the size of an
  -- encoding
  local rnn_size_A = encodings_A_raw[1]:size()[2]
  local rnn_size_B = encodings_B_raw[1]:size()[2]

  -- Collapse the encodings of A and encodings of B into a single vector
  local encodings_A, encodings_B = torch.Tensor(sample_length, rnn_size_A):zero():cuda(), torch.Tensor(sample_length, rnn_size_B):zero():cuda()

  local k = 1
  for i=1,#encodings_A_raw do
    local vectors_A, vectors_B = encodings_A_raw[i], encodings_B_raw[i]
    local l = vectors_A:size(1)
    encodings_A:narrow(1, k, l):set(vectors_A)
    encodings_B:narrow(1, k, l):set(vectors_B)
    k = k + l
  end

  -- encodings_A is sample_size x rnn_size_A
  -- encodings_B is sample_size x rnn_size_B
  --
  -- We want a matrix of rnn_size_A x rnn_size_B.
  print('computing E_AB')
  local E_AB = torch.mm(encodings_A:transpose(1, 2), encodings_B):mul(1 / sample_length)

  -- Also take the product of the means
  print('computing EA_EB')
  local EA_EB = torch.mm(mean_A:view(-1, 1), mean_B:view(1, -1))

  -- And the product of the standard deviations
  print('computing sigma')
  local sigma = torch.mm(sigma_A:view(-1, 1), sigma_B:view(1, -1))

  -- This gives us all the correlation matrices we want.
  print('computing covariance')
  local covariance = torch.csub(E_AB, EA_EB)
  print('computing correlation')
  local correlation = torch.cdiv(covariance, sigma)
  --[[
  print('computing cca')
  local cca = torch.svd(correlation)

  -- Compute the basis change
  print('computing basis change')
  local basis_change_forward = torch.gels(encodings_A, encodings_B)
  local basis_change_backward = torch.gels(encodings_B, encodings_A)

  -- Mean squared error:
  print('computing mse')
  local forward_mse = torch.mm(basis_change_forward, encodings_A):csub(encodings_B)
    :pow(2):sum():mul(1 / (sample_length * rnn_size_B))
  local backward_mse = torch.mm(basis_change, encodings_B):csub(encodings_A)
    :pow(2):sum():mul(1 / (sample_length * rnn_size_A))
  ]]

  -- Save the measurements
  torch.save(opt.out_file, {
    ['correlation'] = correlation,
    ['covariance'] = covariance --[[,
    ['cca'] = cca,
    ['basis_change_forward'] = basis_change_forward,
    ['basis_change_backward'] = basis_change_backward,
    ['forward_mse'] = forward_mse,
    ['backward_mse'] = backward_mse ]]
  })
end

main()
