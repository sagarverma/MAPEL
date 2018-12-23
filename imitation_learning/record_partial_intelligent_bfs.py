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

    def get_distance(self, agent):
        return ((self.loc[0] - agent.loc[0])**2 + (self.loc[1] - agent.loc[1])**2)**0.5

    def _action_on_out_of_los(self, environment, target1, target2):
        adjacent_locs = self._get_adjacent_locs(self.loc)
        empty_adjacent_locs = [adjacent_loc for adjacent_loc in adjacent_locs if self._valid_loc(environment, adjacent_loc)]
        empty_adjacent_locs = self._get_ordered_empty_adjacent_locs(empty_adjacent_locs, target.loc)

        nearest_target = target1.loc
        if self.get_distance(target2) < self.get_distance(target1):
            nearest_target = target2.loc

        new_loc = None
        for adjacent_loc in empty_adjacent_locs:
            if adjacent_loc[0] != self.loc[0] and adjacent_loc[1] != self.loc[1]:
                if self._loc_online_inbetween(adjacent_loc, self.loc, nearest_target):
                    new_loc = adjacent_loc
                    break

        if not new_loc:
            new_loc = random.choice(empty_adjacent_locs)

        return new_loc

    def act(self, environment, target1, target2):
        """
        Guard moves to target or invader based on who is closest.
        """
        observerd_environment = self.get_observed_environment(environment)

        if not self.obs_size:
            return self._action_on_los(observerd_environment, target1, target2)
        else:
            if self._target_inside_obs_space(target1):
                return self._action_on_los(observerd_environment, target1)
            elif self._target_inside_obs_space(target2):
                return self._action_on_los(observerd_environment, target2)
            else:
                return self._action_on_out_of_los(observerd_environment, target1, target2)


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
    fout = open('../dataset/partial_intelligent_bfs/' + phase + '.csv', 'w')
    w = csv.writer(fout)

    obs_no = 1
    for i in range(1):
        invader = Invader(speed=1, obs_size=5)
        guard = NGuard(speed=1, obs_size=5)
        target = Target(speed=0)

        env = NEnvironment([32,32], guard, invader, target, sim_speed=0.0001)

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

            if guard.obs_size:
                guard_observation = guard.get_observed_environment(obs)
                img[guard_observation == 0] = [5,128,5]
            if invader.obs_size:
                invader_observation = invader.get_observed_environment(obs)
                img[invader_observation == 0] = [128,5,5]

            cv2.imwrite('../dataset/partial_intelligent_bfs/' + phase + '/' + str(obs_no).zfill(10) + '.png', img)
            w.writerow([str(obs_no).zfill(10) + '.png', loc_to_action(guard_current_loc, guard_action)])
            obs_no += 1
    # fout = open('dataset/intelligent_bfs/sim.pkl', "wb")
    # pickle.dump({'observations':observations, 'actions':actions, 'rewards':rewards}, fout)
    fout.close()
