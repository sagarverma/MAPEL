import argparse
import multiprocessing
from queue import Empty
import numpy as np
import time
import random
import cv2
import csv
from graph import NxGraph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os

from utils.folder import featurize
from env import Invader, Guard, Target, Environment, Agent

NUM_AGENTS = 3
NUM_ACTIONS = 9

use_gpu = torch.cuda.is_available()
DEVICE = 0

#Data statistics
num_classes = 9

#dataset_name = 'intelligent_bfs'
dataset_name = 'dummies_bfs'
weights_dir = 'weights/'
model_file = weights_dir + dataset_name + '_sim.pt'

def action_to_loc(current_loc, action):
    return get_adjacent_locs(current_loc)[action]

def valid_loc(grid, loc):
    if loc[0] > 0 and loc[1] > 0 and loc[0] < grid.shape[0] and loc[1] < grid.shape[1]:
        if grid[loc[0], loc[1]] != 1:
            return True

    return False

def get_adjacent_locs(current_loc):
    adjacent_locs = [[current_loc[0] - 1, current_loc[1] - 1],
                    [current_loc[0] - 1, current_loc[1]],
                    [current_loc[0] - 1, current_loc[1] + 1],
                    [current_loc[0], current_loc[1] - 1],
                    current_loc,
                    [current_loc[0], current_loc[1] + 1],
                    [current_loc[0] + 1, current_loc[1] - 1],
                    [current_loc[0] + 1, current_loc[1]],
                    [current_loc[0] + 1, current_loc[1] + 1]]
    return adjacent_locs

def loc_to_action(current_loc, next_loc):
    adjacent_locs = get_adjacent_locs(current_loc)

    for i in range(len(adjacent_locs)):
        if adjacent_locs[i][0] == next_loc[0] and adjacent_locs[i][1] == next_loc[1]:
            return i

def loc_online_inbetween(loc, source, target):
    dxc = loc[0] - source[0]
    dyc = loc[1] - source[1]

    dxl = target[0] - source[0]
    dyl = target[1] - source[1]

    cross = dxc * dyl - dyc * dxl

    if cross == 0:
        if abs(dxl) >= abs(dyl):
            if dxl > 0:
                return (source[0] <= loc[0]) & (loc[0] <= target[0])
            else:
                return (target[0] <= loc[0]) & (loc[0] <= source[0])
        else:
            if dyl > 0 :
                return (source[1] <= loc[1]) & (loc[1] <= target[1])
            else:
                return (target[1] <= loc[1]) & (loc[1] <= source[1])
    else:
        return False

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=7)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        #print(x.size())
        x = x.view(-1, 256)
        x = F.relu(self.fc2(x))
        return x

class NEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super(NEnvironment, self).__init__(*args, **kwargs)

    def act(self):
        """
        Invader and guard act inside environment. Invader and guard new positions are updated.
        """
        invader_action = self.invader.act(self.grid, self.target)
        guard_action = self.guard.act(self.grid, self.target, self.invader)

        return guard_action, invader_action

# class DummyInvader(Invader):
#     def __init__(self, *args, **kwargs):
#         super(DummyInvader, self).__init__(*args, **kwargs)
#
#     def act(self, environment, target):
#         adjacent_locs = get_adjacent_locs(self.loc)
#         empty_adjacent_locs = [adjacent_loc for adjacent_loc in adjacent_locs if valid_loc(environment, adjacent_loc)]
#
#         new_loc = None
#         for adjacent_loc in empty_adjacent_locs:
#             if adjacent_loc[0] != self.loc[0] and adjacent_loc[1] != self.loc[1]:
#                 if loc_online_inbetween(adjacent_loc, self.loc, target.loc):
#                     new_loc = adjacent_loc
#                     break
#
#         if not new_loc:
#             graph = NxGraph()
#             graph.grid_to_graph(environment)
#             shortest_path = graph.shortest_path((self.loc[0], self.loc[1]), (target.loc[0], target.loc[1]))
#             if len(shortest_path) <= self.speed:
#                 return [shortest_path[-1][0], shortest_path[-1][1]]
#             else:
#                 return [shortest_path[self.speed][0], shortest_path[self.speed][1]]
#
#         return new_loc

class DummyInvader(Invader):
    def __init__(self, *args, **kwargs):
        super(DummyInvader, self).__init__(*args, **kwargs)

    def act(self, environment, target):
        adjacent_locs = get_adjacent_locs(self.loc)
        empty_adjacent_locs = [adjacent_loc for adjacent_loc in adjacent_locs if valid_loc(environment, adjacent_loc)]

        new_loc = None
        for adjacent_loc in empty_adjacent_locs:
            if adjacent_loc[0] != self.loc[0] and adjacent_loc[1] != self.loc[1]:
                if loc_online_inbetween(adjacent_loc, self.loc, target.loc):
                    new_loc = adjacent_loc
                    break

        if not new_loc:
            new_loc = random.choice(empty_adjacent_locs)

        return new_loc

class ImmitationGuard(Guard):
    def __init__(self, model_file, *args, **kwargs):
        super(ImmitationGuard, self).__init__(*args, **kwargs)
        self.model = torch.load(model_file)

    def act(self, environment, target, invader):
        img = np.zeros((32,32,3))
        img[environment == 1] = [255, 255, 255]
        img[environment == 7] = [255, 0, 0]
        img[environment == 8] = [0, 255, 0]
        img[environment == 9] = [0, 0, 255]
        features = featurize(img)
        features = Variable(torch.from_numpy(features).type(torch.FloatTensor).cuda(DEVICE))
        _, actions = torch.topk(self.model(features), k=num_classes)

        for action in actions.data.cpu().numpy()[0]:
            new_loc = action_to_loc(self.loc, action)
            if valid_loc(environment, new_loc) and new_loc[0] != self.loc[0] and new_loc[1] != self.loc[1]:
                return new_loc

        return self.loc


guard_wins = 0
for i in range(10):
    # os.mkdir('dataset/sims/imitation_dummy_invader/' + str(i).zfill(2))

    invader = DummyInvader(speed=1)
    guard = ImmitationGuard(model_file, speed=1)
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

        # img = np.zeros((32,32,3))
        # img[obs == 1] = [255, 255, 255]
        # img[obs == 7] = [255, 0, 0]
        # img[obs == 8] = [0, 255, 0]
        # img[obs == 9] = [0, 0, 255]
        # cv2.imwrite('dataset/sims/imitation_dummy_invader/' + str(i).zfill(2) + '/' + str(j).zfill(10) + '.png', cv2.resize(img, (500,500), interpolation=cv2.INTER_AREA))
        # #env.render()
        # j += 1

    # print (env.wins)
    if env.wins == 'guard':
        guard_wins += 1

print (guard_wins)
