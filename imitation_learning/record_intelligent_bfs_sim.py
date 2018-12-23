import sys
sys.path.append('../environments')
from env import Environment
from agent import Invader, Guard, Target, Agent

sys.path.append('../utils')
from dataloader import *
from graph import NxGraph

from multiprocessing import Queue
import cv2
import random
from graph import NxGraph
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import csv

def action_to_loc(current_loc, action):
    return get_adjacent_locs(current_loc)[action]

def loc_to_action(current_loc, next_loc):
    adjacent_locs = get_adjacent_locs(current_loc)

    for i in range(len(adjacent_locs)):
        if adjacent_locs[i][0] == next_loc[0] and adjacent_locs[i][1] == next_loc[1]:
            return i

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

class NEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super(NEnvironment, self).__init__(*args, **kwargs)

    def act(self):
        """
        Invader and guard act inside environment. Invader and guard new positions are updated.
        """
        invader_action = self.invader.act(self.grid, self.target)
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

def featurize(obs):
    maps = []

    nmap = np.zeros(obs.shape)
    nmap[obs == 1] = 1
    maps.append(nmap)

    nmap = np.zeros(obs.shape)
    nmap[obs == 7] = 1
    maps.append(nmap)

    nmap = np.zeros(obs.shape)
    nmap[obs == 8] = 1
    maps.append(nmap)

    nmap = np.zeros(obs.shape)
    nmap[obs == 9] = 1
    maps.append(nmap)

    return np.stack(maps, axis=2)

# observations = []
# actions = []
# rewards = []

for phase in ['train','test']:
    fout = open('../dataset/intelligent_bfs/' + phase + '.csv', 'w')
    w = csv.writer(fout)

    obs_no = 1
    for i in range(1):
        invader = Invader(speed=1)
        guard = NGuard(speed=1)
        target = Target(speed=0)

        env = NEnvironment([32,32], guard, invader, target)

        done = False

        while not done:
            guard_current_loc = env.guard.loc
            invader_current_loc = env.invader.loc
            current_obs = env.grid
            guard_action, invader_action = env.act()
            obs, reward, done, info = env.step(guard_action, invader_action)
            # observations.append(featurize(current_obs))
            # actions.append(loc_to_action(guard_current_loc, guard_action))
            # rewards.append(reward)
            img = np.zeros((32,32,3))
            img[obs == 1] = [255, 255, 255]
            img[obs == 7] = [255, 0, 0]
            img[obs == 8] = [0, 255, 0]
            img[obs == 9] = [0, 0, 255]
            cv2.imwrite('../dataset/intelligent_bfs/' + phase + '/' + str(obs_no).zfill(10) + '.png', img)
            w.writerow([str(obs_no).zfill(10) + '.png', loc_to_action(guard_current_loc, guard_action)])
            obs_no += 1
    # fout = open('dataset/intelligent_bfs/sim.pkl', "wb")
    # pickle.dump({'observations':observations, 'actions':actions, 'rewards':rewards}, fout)
    fout.close()
