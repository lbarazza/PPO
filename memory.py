import torch


class Memory:
    """
    Memory that stores the states, actions and rewards of all trajectories.

    Attributes:
            states (list containing numpy arrays): stores all the states of all
                                                   episodes
            actions (list containing numpy arrays): stores all the actions of all
                                                    episodes
            rewards (list of list): a list containing a list (for every episode)
                                    which contains all the rewards specific to an
                                    episode
            episode_rewards (list): stores all the rewards of the current episode
            T (int): number of timesteps across all episodes stored
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_rewards = []
        self.T = 0

    def clear(self):
        """
        It resets all of the memory's attributes.
        """

        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_rewards = []
        self.T = 0

    def add(self, state, action, reward, done):
        """
        It adds one timestep worth of states, actions and rewards to the memory.
        """

        self.T += 1
        self.states.append(state)
        self.actions.append(action)
        self.episode_rewards.append(reward)
        if done:
            self.rewards.append(self.episode_rewards)
            self.episode_rewards = []

    def get(self):
        """
        Returns the memory's content.

        Returns:
            states (torch.tensor): preprocessed states
            actions (torch.tensor): preprocessed actions
            rewards (list of list): not preprocessed rewards to facilitate
                                    calculation of advantages and rewards to go
            T (int): number of timesteps across all episodes stored
        """

        states = torch.tensor(self.states).float()
        actions = torch.tensor(self.actions).float()
        return states, actions, self.rewards, self.T
