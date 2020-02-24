import gym
import torch
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ppo import PPOAgent
from datetime import datetime

# create environment
env = gym.make("Pendulum-v0")

# set random seeds
seed = 123456
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)

# set hyperparameters
lr_policy=0.001
lr_critic=0.005
gam=0.96
lam=0.93
eps=0.20
batch_size=100
policy_updates=80
v_updates=25
update_freq=1

# initialize tensorboard
comment = datetime.now().strftime('%b%d_%H-%M-%S') + f' lr_policy={lr_policy} lr_critic={lr_critic} gamma={gam} lambda={lam} epsilon={eps} batch_size={batch_size} policy_updates={policy_updates} v_updates={v_updates} update_freq={update_freq}'
tb = SummaryWriter("runs/" + comment)

# initilize PPO agent
agent = PPOAgent(
    nS = env.observation_space.shape[0],
    nA = env.action_space.shape[0],
    lr_policy=lr_policy,
    lr_critic=lr_critic,
    gam=gam,
    lam=lam,
    eps=eps,
    batch_size=batch_size,
    policy_updates=policy_updates,
    v_updates=v_updates,
    update_freq=update_freq
)

# set counter for number of episodes and list to store last 100 returns
n_episode = 0
returns = deque(maxlen=100)

while True:
    n_episode += 1

    # reset environment and reward for the current episode
    episode_reward = 0
    state = env.reset()

    while True:
        # render environment
        env.render()

        # action needs to be a list since this accepts Box Actions
        # the reward is of the same type as the action that we pass in

        # choose action
        action = [agent.choose_action(torch.tensor(state).unsqueeze(0).float()).item()]

        # apply the action in the environment and store the outcomes
        next_state, reward, done, info = env.step(action)

        # pass the outcomes of the action to the agent so that it can learn from it
        critic_loss = agent.step(state, action, reward, done)

        # update current state and return
        state = next_state
        episode_reward += reward

        # when the episode is over
        if done:
            # log stats on tensorboard
            tb.add_scalar("Rewards", episode_reward, n_episode)
            if not None: tb.add_scalar("Critic Loss", critic_loss, n_episode)

            # add to the list of last 100 rewards
            returns.append(episode_reward)

            # print some basic stats in the terminal
            print("Episode n. {:6d}   Return: {:9.2f}   Avg. Return: {:9.2f}".format(n_episode, episode_reward, np.mean(returns)))
            break

# close the environment and tensorboard
env.close()
tb.close()
