import torch

def sample_batch(x, batch_size, n_samples):
  for _ in range(n_samples):
    i = torch.randperm(batch_size)[:n_samples]
    yield (j[i] for j in x), i

@torch.no_grad()
def gae_rtg(x, gam, lam):
    R, v, T = x
    j = 0
    gae = torch.empty(T, 1)
    rtg = torch.empty(T, 1)
    for r in R:
        for i in reversed(range(len(r))):
            t = j + i

            # calculate rewards to go
            rtg[t] = r[i] + gam * (rtg[t+1] if i != len(r)-1 else 0)

            # calculate GAE
            delta_t = - v[t] + r[i] + gam * (v[t+1] if i != len(r)-1 else 0)
            gae[t] = delta_t + gam * lam * (gae[t+1] if i != len(r)-1 else 0)

        j += len(r)

    return gae, rtg
