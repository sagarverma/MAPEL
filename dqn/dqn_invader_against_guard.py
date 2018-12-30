import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random

import visdom 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

sys.path.append('../environments')
from env import *
from agent import *

sys.path.append('../utils')
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from dataloader import action_to_loc, valid_loc, loc_to_action

from models import DQN

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

BATCH_SIZE = 1024
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 500000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGERT_UPDATE_FREQ = 1000
LEARNING_RATE = 0.001
ALPHA = 0.99
EPS = 0.2
IMG_H = 32
IMG_W = 32
IMG_C = 3
NUM_ACTIONS = 9

agent = 'invader'

# Build Environment
invader = Invader(speed=1)
guard = Guard2Targets(speed=1)
target = Target(speed=0)

env = Environment2Targets([32,32], guard, invader, target)

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}


optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

exploration_schedule = LinearSchedule(100000, 0.1)

# Construct an epilson greedy policy with given exploration schedule
def select_epilson_greedy_action(model, obs, t):
    sample = random.random()
    eps_threshold = exploration_schedule.value(t)
    if sample > eps_threshold:
        obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
        # Use volatile = True if variable is only used in inference mode, i.e. don't save the history
        return model(Variable(obs)).data.max(1)[1].cpu()
    else:
        return torch.IntTensor([[random.randrange(NUM_ACTIONS)]])
    


# vis = visdom.Visdom(port=8124)

# Initialize target q function and q function
Q = DQN(IMG_C, FRAME_HISTORY_LEN, NUM_ACTIONS).type(dtype)
target_Q = DQN(IMG_C, FRAME_HISTORY_LEN, NUM_ACTIONS).type(dtype)

# Construct Q network optimizer function
optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

# Construct the replay buffer
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN)

###############
# RUN ENV     #
###############
num_param_updates = 0
mean_episode_reward = -float('nan')
best_mean_episode_reward = -float('inf')
last_obs = env.reset()
episodes_rewards = []

for t in count():
    ### Step the env and store the transition
    # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
    last_idx = replay_buffer.store_frame(last_obs)
    print (last_idx, last_obs.shape)
    # encode_recent_observation will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    recent_observations = replay_buffer.encode_recent_observation()
    print (recent_observations.shape)

    guard_action, invader_action = env.act()
    # Choose random action if not yet start learning
    if t > LEARNING_STARTS:
        action = select_epilson_greedy_action(Q, recent_observations, t).item()
        print (action)
    else:
        action = random.randrange(NUM_ACTIONS)
        print (action)
    
    
    print (guard_action, invader_action)
    # Advance one step
    obs, reward, done, _ = env.step(guard_action, invader_action)
    print (reward)
    # clip rewards between -1 and 1
    reward = -1 * reward
#     reward = max(-1.0, min(reward, 1.0))
    # Store other info in replay memory
    replay_buffer.store_effect(last_idx, action, reward, done)
    # Resets the environment when reaching an episode boundary.
    if done:
        episodes_rewards.append(reward)
        if len(episodes_rewards) % 100 == 0:
            print (np.mean(episodes_rewards), t)
            episodes_rewards = []
            
            torch.save(Q, '../weights/Q_dqn_invader.pt')
            torch.save(target_Q, '../weights/target_Q_dqn_invader.pt')
                 
        obs = env.reset()
        
    last_obs = obs
    exit()
    ### Perform experience replay and train the network.
    # Note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (t > LEARNING_STARTS and
            t % LEARNING_FREQ == 0 and
            replay_buffer.can_sample(BATCH_SIZE)):
        # Use the replay buffer to sample a batch of transitions
        # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(BATCH_SIZE)
        # Convert numpy nd_array to torch variables for calculation
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
        act_batch = Variable(torch.from_numpy(act_batch).long())
        rew_batch = Variable(torch.from_numpy(rew_batch))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

        if USE_CUDA:
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()

        # Compute current Q value, q_func takes only state and output value for every state-action pair
        # We choose Q based on action taken.
        current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).view(-1)
        # Compute next Q value based on which action gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = rew_batch + (GAMMA * next_Q_values)
        # Compute Bellman error
        bellman_error = target_Q_values - current_Q_values
        # clip the bellman error between [-1 , 1]
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        # Note: clipped_bellman_delta * -1 will be right gradient
        d_error = clipped_bellman_error * -1.0
        # Clear previous gradients before backward pass
        optimizer.zero_grad()
        # run backward pass
        current_Q_values.backward(d_error.data)

        # Perfom the update
        optimizer.step()
        num_param_updates += 1

        # Periodically update the target network by Q network to target Q network
        if num_param_updates % TARGERT_UPDATE_FREQ == 0:
            target_Q.load_state_dict(Q.state_dict())

print (np.mean(episodes_rewards))