import torch
from torch.distributions import Normal
from models import *
from memory import Memory
import copy
import utils


class PPOAgentOneNet:
    """
    Agent using Proximal Policy Optimization by clipping

    Args:
        nS (int): Dimension of observation space
        nA (int): Dimension of action space
        lr_policy (float): Learning rate of the policy
        lr_critic (float): Learning rate of the critic
        gam (float): Value of gamma
        lam (float): Value of lambda (for GAE)
        eps (float): Value of epsilon
        batch_size (int): batch size for policy and critic updates
        policy_updates (int): number of ppo training iterations per call to
                              the "update" method
        v_updates (int): number of critic training epochs per call to "update"
                         method
        update_freq (int): Update agent every update_freq episodes

    Attributes:
        memory (Memory): Memory where all the trajectories' information get stored
        policy (Actor): Neural network for the policy
        policy_optimizer (torch.optim): Optimizer for the policy
        critic (Critic): Neural network for the critic
        critic_optimizer (torch.optim): Optimizer for the critic
        n_tau (int): number of current trajectory
        nS, nA, lr_policy, lr_critic, gam, lam, eps, batch_size, policy_updates,
        v_updates, update_freq : see "Args"
    """

    def __init__(self, nS, nA, lr, gam, lam, eps, c1, c2, batch_size, policy_updates, update_freq):
        self.memory = Memory()
        self.policy = ActorCritic(nS, nA)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gam = gam
        self.lam = lam
        self.eps = eps
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.policy_updates = policy_updates
        self.update_freq = update_freq
        self.n_tau = 1

    @torch.no_grad()
    def choose_action(self, state):
        """
        Returns an action from the observed state.
        """

        return self.policy(state)[0].sample()

    def step(self, state, action, reward, done):
        """
        Central Unit controlling everything. It adds to the memory and
        calls the update method when needed.

        Returns:
            (torch.tensor): Critic loss if done, otherwise None
        """

        self.memory.add(state, action, reward, done)
        if done and self.n_tau % self.update_freq == 0:
            self.n_tau += 1
            return self.update()
        return None

    def update(self):
        """
        Where policy and critic updates get computed.

        Returns:
            (torch.tensor): Critic loss
        """

        # get states, actions, rewards and total timesteps from memory
        states, actions, R, T = self.memory.get()

        # compute value estimates for the states
        dist, v = self.policy(states)

        # compute advantages (using GAE) and rewards to go
        A, rtg = utils.gae_rtg((R, v, T), self.gam, self.lam)

        # store the initial version of both the policy and the log probs of the
        # actions for later comparison with the future versions (needed for PPO)
        policy_old = copy.deepcopy(self.policy)
        log_probs_old = torch.sum(policy_old(states)[0].log_prob(actions), dim=1, keepdim=True)

        # sample from a batch of experiences
        # ("_" subscript indicates "sampled from")
        for (v_, A_, rtg_, log_probs_old_, actions_), i in utils.sample_batch((v, A, rtg, log_probs_old, actions), self.batch_size, self.policy_updates):
            dist_ , v_ = self.policy(states[i])
            log_probs_ = torch.sum(dist_.log_prob(actions_), dim=1, keepdim=True)

            # estimate ratio between the new log probs and the old ones
            r_ = torch.exp(log_probs_ - log_probs_old_)

            l_1 = r_ * A_
            l_2 = torch.clamp(r_, 1-self.eps, 1+self.eps) * A_

            # TODO: implement entropy
            # TODO: merge policy and critic

            S = -torch.sum(log_probs_ * torch.exp(log_probs_))

            # surragate loss function for PPO
            l_clip = -torch.mean(torch.min(l_1, l_2), dim=0) + self.c1 * F.mse_loss(v_, rtg_) - self.c2 * S

            # update the policy
            self.policy_optimizer.zero_grad()
            l_clip.backward(retain_graph=True)
            self.policy_optimizer.step()


        # clear the memory. PPO is an On-Policy method so we don't need these
        # memories anymore
        self.memory.clear()

        # return the loss of the value function for display
        return F.mse_loss(v, rtg)
