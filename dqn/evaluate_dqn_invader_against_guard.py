import sys
sys.path.append('../environments')
from env import Environment
from agent import Invader, Guard, Target, Agent

sys.path.append('../utils')
from dataloader import *
from graph import NxGraph

from models import DQN

import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os

<<<<<<< HEAD

=======
>>>>>>> a481c368a11700233855b30b7721d7b568796767

class NEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super(NEnvironment, self).__init__(*args, **kwargs)

    def act(self):
        """
        Invader and guard act inside environment. Invader and guard new positions are updated.
        """
        invader_action = self.invader.act(self._featurize(), self.grid)
        guard_action = self.guard.act(self.grid, self.invader, self.target)

        return guard_action, invader_action

class NGuard(Guard):
    def __init__(self, *args, **kwargs):
        super(NGuard, self).__init__(*args, **kwargs)

    def act(self, environment, target1, target2):
        """
        Guard moves to target or invader based on who is closest.
        """
        graph = NxGraph()
        graph.grid_to_graph(environment)
        shortest_path1 = graph.shortest_path((self.loc[0], self.loc[1]), (target1.loc[0], target1.loc[1]))
        shortest_path2 = graph.shortest_path((self.loc[0], self.loc[1]), (target2.loc[0], target2.loc[1]))

        if len(shortest_path1) <= len(shortest_path2):
            if len(shortest_path1) <= self.speed:
                return [shortest_path1[-1][0], shortest_path1[-1][1]]
            else:
                return [shortest_path1[self.speed][0], shortest_path1[self.speed][1]]
        else:
            if len(shortest_path2) <= self.speed:
                return [shortest_path2[-1][0], shortest_path2[-1][1]]
            else:
                return [shortest_path2[self.speed][0], shortest_path2[self.speed][1]]


class NInvader(Invader):
    def __init__(self, model, *args, **kwargs):
        super(NInvader, self).__init__(*args, **kwargs)
        self.model = model

    def act(self, obs, grid):
        """
        Guard moves to target or invader based on who is closest.
        """
        current_loc = self.loc
        obs = torch.from_numpy(np.asarray([obs.transpose(2,0,1) / 255.])).float().cuda()
        action_logits = self.model(obs).data.cpu().numpy()[0]
        actions_high_to_low = np.flip(np.argsort(action_logits))

        for action in actions_high_to_low:
            action_loc = action_to_loc(current_loc, action)
            if valid_loc(grid, action_loc):
                return action_loc

use_gpu = torch.cuda.is_available()
DEVICE = 0

#Data statistics
num_actions = 9
history_length = 1
num_channels = 3

agent = 'invader'
dataset_name = 'Q_dqn'
# dataset_name = 'dummies_bfs'

weights_dir = '../weights/'

#Create model and initialize/freeze weights
model_conv = DQN(num_channels, history_length, num_actions)
model_conv = torch.load(weights_dir + dataset_name + '_' + agent + '.pt')

if use_gpu:
    model_conv = model_conv.cuda(DEVICE)

invader_wins = 0
for i in range(100):

    invader = NInvader(model_conv, speed=1)
    guard = NGuard(speed=1)
    target = Target(speed=0)

    env = NEnvironment([32,32], guard, invader, target)

    done = False

    # j = 1
    while not done:
        guard_current_loc = env.guard.loc
        invader_current_loc = env.invader.loc
        current_obs = env.grid
        guard_action, invader_action = env.act()
        obs, reward, done, info = env.step(guard_action, invader_action)
        env.render()

    if env.wins == 'invader':
        invader_wins += 1

print (invader_wins)
