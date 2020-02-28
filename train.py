import gym
import torch
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ppo import PPOAgent
from ppo_onenet import PPOAgentOneNet
from pathlib import Path
from datetime import datetime
import utils


# create environment
env = gym.make("LunarLanderContinuous-v2")

# set random seeds
seed = 123456
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)

# set hyperparameters
lr=0.005
gam=0.99#0.99
lam=0.98#0.96
eps=0.20
c1=0.001
c2=0.01
batch_size=50#150
policy_updates=10#100
v_updates=50
update_freq=1

# initialize tensorboard
comment = datetime.now().strftime('%b%d_%H-%M-%S') + f' lr={lr} gamma={gam} lambda={lam} epsilon={eps} c1={c1} c2={c2} batch_size={batch_size} policy_updates={policy_updates} v_updates={v_updates} update_freq={update_freq}'
tb = SummaryWriter("runs/" + comment)

# initilize PPO agent
agent = PPOAgentOneNet(
    nS = env.observation_space.shape[0],
    nA = env.action_space.shape[0],
    lr=lr,
    gam=gam,
    lam=lam,
    eps=eps,
    c1=c1,
    c2=c2,
    batch_size=batch_size,
    policy_updates=policy_updates,
    update_freq=update_freq
)

# set counter for number of episodes and list to store last 100 returns
n_episode = 0
returns = deque(maxlen=100)

# initilize parameters for saving training
save_freq = 10
run_name = "test37"
checkpoint_path = 'checkpoints/' + run_name + '.tar'
checkpoint_path_best = 'checkpoints/' + run_name + '_best' + '.tar'
checkpoint_file = Path(checkpoint_path)

# load already existing agent if possible
if checkpoint_file.is_file():
    agent, n_episode = utils.load_agent_onenet(agent, checkpoint_path)


best_avg = -float("inf")
while True:
    n_episode += 1

    # reset environment and reward for the current episode
    episode_reward = 0
    state = env.reset()
    z = 0
    while True:
        z+=1
        # render environment

        ## TEST ##
        #env.render()

        # action needs to be a list since this accepts Box Actions
        # the reward is of the same type as the action that we pass in

        # choose action
        action = agent.choose_action(torch.tensor(state).unsqueeze(0).float()).squeeze(0).numpy()

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

            # calculate average return
            avg = np.mean(returns)

            # print some basic stats in the terminal
            print("Episode n. {:6d}   Return: {:9.2f}   Avg. Return: {:9.2f}".format(n_episode, episode_reward, avg), "     ", z)

            # save best agent so far
            if avg > best_avg:
                best_avg = avg
                utils.save_agent_onenet(agent, n_episode, checkpoint_path_best)

            # save current agent
            if n_episode % save_freq == 0:
                utils.save_agent_onenet(agent, n_episode, checkpoint_path)

            break

# close the environment and tensorboard
env.close()
tb.close()
