import numpy as np
import random
import time
import sys

import matplotlib.pyplot as plt
import cv2

from shapely.geometry import Point
from shapely.geometry import box

sys.path.append('../utils/')
from dungeonGenerator import dungeonGenerator
from graph import Graph, NxGraph

class Environment(object):

    def __init__(self, grid_size, guard, invader, target, fixed_grid=False, sim_speed=1):
        self.grid_size = grid_size
        self.fixed_grid = fixed_grid
        self.sim_speed = sim_speed

        if self.fixed_grid:
            self.grid = np.load('fixed_grid.npy')
        else:
            dungeon = dungeonGenerator(*self.grid_size)
            dungeon.placeRandomRooms(2, 10, 5, 2)
            self.grid = np.asarray(dungeon.grid)

        self.guard = guard
        self.invader = invader
        self.target = target

        self.step_count = 0
        self.done = False
        self.wins = None

        self._spawn_agents()


        self._init_game_state = self.get_json_info()


    def _spawn_agents(self):
        """
        Place agents in the environment
        """
        guard_loc = self.guard.set_position(self.grid)
        self.grid[guard_loc[0], guard_loc[1]] = self.guard.id

        invader_loc = self.invader.set_position(self.grid)
        self.grid[invader_loc[0], invader_loc[1]] = self.invader.id

        target_loc = self.target.set_position(self.grid)
        self.grid[target_loc[0], target_loc[1]] = self.target.id

    def reset(self):

        if self._init_game_state is not None:
            self.set_json_info()
        else:
            if self.fixed_grid:
                self.grid = np.load('fixed_grid.npy')
            else:
                dungeon = dungeonGenerator(*self.grid_size)
                dungeon.placeRandomRooms(2, 10, 5, 2)
                self.grid = np.asarray(dungeon.grid)

            self._spawn_agents()
            self.step_count = 0
            self.done = False
            self.wins = None

        return self.grid


    def get_json_info(self):
        info = {'guard': {'id': self.guard.id, 'loc': self.guard.loc},
            'invader': {'id': self.invader.id, 'loc': self.invader.loc},
            'target': {'id': self.target.id, 'loc': self.target.loc},
            'grid': self.grid,
            'grid_size': self.grid.shape,
            'step_count': self.step_count,
            'done': self.done,
            'wins': self.wins}

        return info

    def set_json_info(self):
        self.grid = np.copy(self._init_game_state['grid'])

        self.guard.loc = self._init_game_state['guard']['loc']
        self.invader.loc = self._init_game_state['invader']['loc']
        self.targetloc = self._init_game_state['target']['loc']

        self.step_count = self._init_game_state['step_count']
        self.done = self._init_game_state['done']
        self.wins = self._init_game_state['wins']

    def _get_rewards(self):
        """
         Reward for guard
        """
        if self.done:
            if self.wins == "guard":
                return 1
            if self.wins == "invader":
                return -1

        return 0

    def act(self):
        """
        Invader and guard act inside environment. Invader and guard new positions are updated.
        """
        invader_action = self.invader.act(self.grid, self.target)
        guard_action = self.guard.act(self.grid, self.invader)

        return guard_action, invader_action

    def step(self, guard_action, invader_action):
        """
        Given that our environment is static throught out the game, only things that this function has to caclulate
        are rewards for invader and guard and game done or not
        """
        if guard_action[0] == self.invader.loc[0] and guard_action[1] == self.invader.loc[1]:
            self.wins = "guard"
            self.done = True
        if guard_action[0] == self.target.loc[0] and guard_action[1] == self.target.loc[1]:
            self.wins = "guard"
            self.done = True
        if invader_action[0] == self.target.loc[0] and invader_action[1] == self.target.loc[1]:
            self.wins = "invader"
            self.done = True

        if self.done and self.wins == "guard":
            self.grid[self.guard.loc[0], self.guard.loc[1]] = 0
            self.grid[guard_action[0], guard_action[1]] = self.guard.id
            self.guard.loc = guard_action

        if self.done and self.wins == "invader":
            self.grid[self.invader.loc[0], self.invader.loc[1]] = 0
            self.grid[invader_action[0], invader_action[1]] = self.invader.id
            self.invader.loc = invader_action

        if not self.done:
            self.grid[self.guard.loc[0], self.guard.loc[1]] = 0
            self.grid[guard_action[0], guard_action[1]] = self.guard.id
            self.guard.loc = guard_action

            self.grid[self.invader.loc[0], self.invader.loc[1]] = 0
            self.grid[invader_action[0], invader_action[1]] = self.invader.id
            self.invader.loc = invader_action

        self.step_count += 1

        return self.grid, self._get_rewards(), self.done, self.get_json_info()

    def render(self):
        img = np.zeros((self.grid.shape[0], self.grid.shape[1], 3))
        img[self.grid == 1] = [255, 255, 255]
        img[self.grid == 7] = [255, 0, 0]
        img[self.grid == 8] = [0, 255, 0]
        img[self.grid == 9] = [0, 0, 255]

        if self.guard.obs_size:
            guard_observation = self.guard.get_observed_environment(self.grid)
            img[guard_observation == 0] = [5,128,5]
        if self.invader.obs_size:
            invader_observation = self.invader.get_observed_environment(self.grid)
            img[invader_observation == 0] = [128,5,5]

        plt.imshow(img.astype(np.uint8))
        plt.pause(self.sim_speed)
        plt.draw()
