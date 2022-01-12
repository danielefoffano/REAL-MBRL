from Dynamics_model import Dynamics_model
from QAdversary import QAdversary
from QAgent import QAgent
from FakeLake import FakeLake
import gym
import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from frozen_lake import FrozenLakeEnv
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os

cmap_4 = [
"SFFF",
"FFHF",
"FFFF",
"FHFG"]

size = 4
env = FrozenLakeEnv(None, None, True, size, 0.8)
n_observations = env.observation_space.n
n_actions = env.action_space.n

#number of episode we will run
n_episodes = 5000
#maximum of iteration per episode
max_iter_episode = 100
n_ensembles = 3

use_adv = True

rewards_per_episode = list()
agent = QAgent(n_observations,n_actions)

adversary = None
epsilon = 0.9
if use_adv:
    adversary = QAdversary(n_observations, n_actions, n_ensembles, epsilon)

env = FakeLake(size, cmap_4)
env.env.render()

# Run until we collect 10k samples
while env.tot_samples < 10000:

    if True:

        ret_true = env.eval_agent(agent)
        ret_learn = env.eval_agent_learned(agent,adversary)
        sim = env.model_similarity()
        print("After {} samples: {:.3f} (true env) | {:.3f} (learned env) | {:.3f} (similarity)".format(env.tot_samples,ret_true, ret_learn, sim))

        env.file.write('{},{},{},{}\n'.format(env.tot_samples, ret_true, ret_learn,sim))
        env.file.flush()

        env.all_rews = []

    env.improve_model(agent, adversary)

    for e in range(n_episodes):
        
        current_state = env.reset()
        done = False
        
        #sum the rewards that the agent gets from the environment
        total_episode_reward = 0
        
        for i in range(max_iter_episode): 

            # Take action
            action = agent.take_action(current_state)

            adv_model = None
            
            if use_adv:
                adv_model = adversary.take_action(current_state, action)
            
            # Sample from environment
            next_state, reward, done, _ = env.step(action, adv_model)

            # Check next action to update adversary
            # For the adversary, the future state is (next_state, next_action)

            temp_action = agent.take_action(next_state)
            
            # Update main player Q Table
            agent.improve_q_table(current_state, action, next_state, reward)

            # Update adversary Q Table, use negative reward to minimise
            if use_adv:
                adversary.improve_q_table(current_state, action, adv_model, next_state, temp_action, -reward)

            total_episode_reward = total_episode_reward + reward

            current_state = next_state
            # If the episode is finished, we leave the for loop
            if done:
                break