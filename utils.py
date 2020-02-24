import torch

def sample_batch(x, batch_size, n_samples):
    """
    Samples a subset of batch_size elements at the same indeces across all values
    in tuple x. It returns an interator that does this for n_samples times.

    Args:
        x (tuple): The tuple of elements to sample from. The elements of the tuple
                   must be of type torch.tensor
        batch_size (int): The batch size
        n_samples (int): Number of iterations for which the iterator will sample

    Returns:
        (iterator): Iterator that runs for n_samples times

    Yields:
        (tuple): Samples for all elements of x
        i (torch.tensor): Indices used to sample
    """

    for _ in range(n_samples):
        i = torch.randperm(batch_size)[:n_samples]
        yield (j[i] for j in x), i

@torch.no_grad()
def gae_rtg(x, gam, lam):
    """
    This function calculates the rewards to go and at the same time the
    advantages with GAE. It works even if the length of the episodes
    is not the same for all of them.

    Args:
        x (tuple): Stores the rewards (R) as a list, the value estimates from
                   the critic (V) as a torch tensor and the total number of
                   timesteps stored (T) as an int.
        gam (float): Value of gamma
        lam (float): Value of lamda

    Returns:
        gae (torch.tensor): The advantages calculated with GAE
        rtg (torch.tensor): The rewards to go
    """

    R, V, T = x
    j = 0
    gae = torch.empty(T, 1)
    rtg = torch.empty(T, 1)
    for r in R:
        for i in reversed(range(len(r))):
            t = j + i # keep track of where we are at in global coordinates

            # calculate GAE
            delta_t = - V[t] + r[i] + gam * (V[t+1] if i != len(r)-1 else 0)
            gae[t] = delta_t + gam * lam * (gae[t+1] if i != len(r)-1 else 0)

            # calculate rewards to go
            rtg[t] = r[i] + gam * (rtg[t+1] if i != len(r)-1 else 0)

        j += len(r)

    return gae, rtg
