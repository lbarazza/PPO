import torch

def sample_batch(x, batch_size, n_samples):
  for _ in range(n_samples):
    i = torch.randperm(batch_size)[:n_samples]
    yield (j[i] for j in x), i
