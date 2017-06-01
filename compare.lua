-- Compute several statistics given two encoding distribution samples.
-- Currently computes:
--  - Covariance
--  - Correlation
require 'cutorch'
require 'nn'
require 'cunn'

function normalize_mean(X)
  local N = X:clone()

  local w, h = N:size(1), N:size(2)

  -- First, get units normalized to mean 0 and standard deviation 1
  local M = N:sum(1):mul(1 / w)
  N:csub(
    M:view(1, h):expand(w, h)
  )

  print('Mean should be zero:')
  M = N:sum(1):mul(1 / w)
  print(M)

  -- Normalize to standard deviation 1
  --[[
  local S = N:clone():pow(2):sum(1):mul(1 / w):sqrt()
  N:cdiv(
    S:view(1, h):expand(w, h)
  )
  ]]

  return N
end

function covariance_matrix(A, B) -- samples x sizeA, samples x sizeB
  -- Assume these are normalized, so return:
  return torch.mm(A:t(), B):mul(1 / A:size(1)) -- sizeA x sizeB
end

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
  local sample_length = #encodings_A_raw -- should also = #encodings_B_raw
  -- (not using per-word embedding) A['sample_length'] -- should also == B['sample_length']

  -- Measure the rnn size by looking at the size of an
  -- encoding
  local rnn_size_A = encodings_A_raw[1]:size()[2]
  local rnn_size_B = encodings_B_raw[1]:size()[2]

  -- Collapse the encodings of A and encodings of B into a single vector
  local encodings_A, encodings_B = torch.Tensor(sample_length, rnn_size_A):zero(), torch.Tensor(sample_length, rnn_size_B):zero()

  --[[
  local k = 1
  for i=1,#encodings_A_raw do
    local vectors_A, vectors_B = encodings_A_raw[i], encodings_B_raw[i]
    vectors_A = nn.utils.recursiveType(vectors_A, 'torch.DoubleTensor')
    vectors_B = nn.utils.recursiveType(vectors_B, 'torch.DoubleTensor')
    for j=1,vectors_A:size(1) do
      encodings_A[k] = vectors_A[j]
      encodings_B[k] = vectors_B[j]
      k = k + 1
    end
  end
  ]]

  -- Take sentence embeddings
  -- (this also reduces space size for
  -- regression purposes)
  for i=1,sample_length do
    local vector_A = encodings_A_raw[i]
    local last_element_A = vector_A[vector_A:size(1)]
    encodings_A[i] = nn.utils.recursiveType(last_element_A, 'torch.DoubleTensor')

    local vector_B = encodings_B_raw[i]
    local last_element_B = vector_B[vector_B:size(1)]
    encodings_B[i] = nn.utils.recursiveType(last_element_B, 'torch.DoubleTensor')
  end

  -- Two normalized samples
  local nA = normalize_mean(encodings_A)
  local nB = normalize_mean(encodings_B)

  -- All covariance matrices
  local c_AA = covariance_matrix(nA, nA)
  local c_AB = covariance_matrix(nA, nB)
  local c_BA = covariance_matrix(nB, nA)
  local c_BB = covariance_matrix(nB, nB)

  local ic_AA = torch.inverse(c_AA)
  local ic_BB = torch.inverse(c_BB)

  local cca_matrix = ic_AA * c_AB * ic_BB * c_BA

  local e = torch.eig(cca_matrix)

  -- Compute the basis change
  print('computing basis change (forward)')
  local basis_change_forward, forward_residual = torch.gels(encodings_B, encodings_A)
  print('computing basis change (backward)')
  local basis_change_backward, backward_residual = torch.gels(encodings_A, encodings_B)

  -- Mean squared error:
  print('computing mse')
  local forward_mse = torch.sum(torch.pow(
      torch.csub(torch.mm(basis_change_forward, encodings_A:t()), encodings_B:t()), -- residuals
      2
    ), 2):mul(1 / sample_length)
  local backward_mse = torch.sum(torch.pow(
      torch.csub(torch.mm(basis_change_backward, encodings_B:t()), encodings_A:t()), -- residuals
      2
    ), 2):mul(1 / sample_length)

  -- Save the measurements
  torch.save(opt.out_file, {
    ['correlation_AA'] = c_AA,
    ['correlation_AB'] = c_AB,
    ['correlation_BA'] = c_BA,
    ['correlation_BB'] = c_BB,
    ['cca_magnitudes'] = e:narrow(2, 1, 1),
    ['basis_change_forward'] = basis_change_forward,
    ['basis_change_backward'] = basis_change_backward,
    ['forward_mse'] = forward_mse,
    ['backward_mse'] = backward_mse
  })
end

main()
