import sys
sys.path.append('../environments')
from env import Environment
from agent import Invader, Guard, Target, Agent

sys.path.append('../utils')
from dataloader import *
from graph import NxGraph

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

class SimpleNet(nn.Module):
    def __init__(self, num_channels, history_length, num_actions):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels*history_length, 256, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*20*20, 2048)
        self.fc2 = nn.Linear(2048, num_actions)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
#         print (x.size())
        x = x.view(x.size()[0], -1)
        x = F.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class NEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super(NEnvironment, self).__init__(*args, **kwargs)

    def act(self):
        """
        Invader and guard act inside environment. Invader and guard new positions are updated.
        """
        invader_action = self.invader.act(self.grid, self.target)
        guard_action = self.guard.act(self._featurize(), self.grid)

        return guard_action, invader_action

class NGuard(Guard):
    def __init__(self, model, *args, **kwargs):
        super(NGuard, self).__init__(*args, **kwargs)
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

agent = 'guard'
dataset_name = 'intelligent_bfs'
# dataset_name = 'dummies_bfs'

weights_dir = '../weights/'

#Create model and initialize/freeze weights
model_conv = SimpleNet(num_channels, history_length, num_actions)
model_conv = torch.load(weights_dir + dataset_name + '_' + agent + '_sim.pt')

if use_gpu:
    model_conv = model_conv.cuda(DEVICE)
    
guard_wins = 0
for i in range(100):

    invader = Invader(speed=1)
    guard = NGuard(model_conv, speed=1)
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
#         env.render()

    if env.wins == 'guard':
        guard_wins += 1

print (guard_wins)