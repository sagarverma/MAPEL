import sys
sys.path.append('../environments')
from env import Environment
from agent import Agent, Invader, Target, Guard

from graph import NxGraph

from multiprocessing import Queue
import cv2
import random

import numpy as np
import time
import matplotlib.pyplot as plt
import os

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



guard_wins = 0
for i in range(1):
    # os.mkdir('dataset/sims/partial_intelligent_bfs/' + str(i).zfill(2))

    invader = Invader(speed=1, obs_size=3)
    guard = NGuard(speed=1, obs_size=3)
    target = Target(speed=0)

    env = NEnvironment([32,32], guard, invader, target, sim_speed=0.01)

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

        # if guard.obs_size:
        #     guard_observation = guard.get_observed_environment(obs)
        #     img[guard_observation == 0] = [5,128,5]
        # if invader.obs_size:
        #     invader_observation = invader.get_observed_environment(obs)
        #     img[invader_observation == 0] = [128,5,5]
        #
        # cv2.imwrite('dataset/sims/partial_intelligent_bfs/' + str(i).zfill(2) + '/' + str(j).zfill(10) + '.png', cv2.resize(img, (500,500), interpolation=cv2.INTER_AREA))
        env.render()
        # j += 1

    print (env.wins)
    if env.wins == 'guard':
        guard_wins += 1

print (guard_wins)
