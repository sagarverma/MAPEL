import argparse
import multiprocessing
from queue import Empty
import numpy as np
import time
import random
import cv2
import csv
from graph import NxGraph

from env import Invader, Guard, Target, Environment, Agent

NUM_AGENTS = 3
NUM_ACTIONS = 9

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
            graph = NxGraph()
            graph.grid_to_graph(environment)
            shortest_path = graph.shortest_path((self.loc[0], self.loc[1]), (target.loc[0], target.loc[1]))
            if len(shortest_path) <= self.speed:
                return [shortest_path[-1][0], shortest_path[-1][1]]
            else:
                return [shortest_path[self.speed][0], shortest_path[self.speed][1]]

        return new_loc

class DummyGuard(Guard):
    def __init__(self, *args, **kwargs):
        super(DummyGuard, self).__init__(*args, **kwargs)

    def act(self, environment, target, invader):
        guard_invader_los = ((self.loc[0]-invader.loc[0])**2 + (self.loc[1]-invader.loc[1])**2)**0.5
        guard_target_los = ((self.loc[0]-target.loc[0])**2 + (self.loc[1]-target.loc[1])**2)**0.5

        adjacent_locs = get_adjacent_locs(self.loc)
        empty_adjacent_locs = [adjacent_loc for adjacent_loc in adjacent_locs if valid_loc(environment, adjacent_loc)]

        new_loc = None
        for adjacent_loc in empty_adjacent_locs:
            if adjacent_loc[0] != self.loc[0] and adjacent_loc[1] != self.loc[1]:
                if guard_invader_los < guard_target_los:
                    if loc_online_inbetween(adjacent_loc, self.loc, invader.loc):
                        new_loc = adjacent_loc
                        break
                else:
                    if loc_online_inbetween(adjacent_loc, self.loc, target.loc):
                        new_loc = adjacent_loc
                        break

        if not new_loc:
            if guard_invader_los < guard_target_los:
                graph = NxGraph()
                graph.grid_to_graph(environment)
                shortest_path = graph.shortest_path((self.loc[0], self.loc[1]), (invader.loc[0], invader.loc[1]))
                if len(shortest_path) <= self.speed:
                    new_loc = [shortest_path[-1][0], shortest_path[-1][1]]
                else:
                    new_loc = [shortest_path[self.speed][0], shortest_path[self.speed][1]]
            else:
                graph = NxGraph()
                graph.grid_to_graph(environment)
                shortest_path = graph.shortest_path((self.loc[0], self.loc[1]), (target.loc[0], target.loc[1]))
                if len(shortest_path) <= self.speed:
                    new_loc = [shortest_path[-1][0], shortest_path[-1][1]]
                else:
                    new_loc = [shortest_path[self.speed][0], shortest_path[self.speed][1]]

        return new_loc

for phase in ['train','test']:
    fout = open('dataset/dummies_bfs/' + phase + '.csv', 'wb')
    w = csv.writer(fout)

    obs_no = 1
    for i in range(5000):
        invader = DummyInvader(speed=1)
        guard = DummyGuard(speed=1)
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
            cv2.imwrite('dataset/dummies_bfs/' + phase + '/' + str(obs_no).zfill(10) + '.png', img)
            w.writerow([str(obs_no).zfill(10) + '.png', loc_to_action(guard_current_loc, guard_action)])
            obs_no += 1
    # fout = open('dataset/intelligent_bfs/sim.pkl', "wb")
    # pickle.dump({'observations':observations, 'actions':actions, 'rewards':rewards}, fout)
    fout.close()
