import sys
import numpy as np
import random
import time
import copy

import matplotlib.pyplot as plt
import cv2

from shapely.geometry import Point
from shapely.geometry import box

sys.path.append('../utils/')
from dungeonGenerator import dungeonGenerator
from graph import Graph, NxGraph

class Environment(object):

    def __init__(self, grid_size, guards, invaders, target, fixed_grid=False, sim_speed=1):
        self.grid_size = grid_size
        self.fixed_grid = fixed_grid
        self.sim_speed = sim_speed
        self.guards_reward = 0
        self.invaders_reward = 0

        if self.fixed_grid:
            self.grid = np.load('fixed_grid.npy')
        else:
            dungeon = dungeonGenerator(*self.grid_size)
            dungeon.placeRandomRooms(2, 10, 5, 2)
            self.grid = np.asarray(dungeon.grid)

        self.guards = {guard.id: guard for guard in guards}
        self.invaders = {invader.id: invader for invader in invaders}
        self.target = target

        self.step_count = 0
        self.done = False
        self.wins = None

        self._spawn_agents()


        #self._init_game_state = self.get_json_info()


    def _spawn_agents(self):
        """
        Place agents in the environment
        """

        for guard in self.guards.values():
            guard.speed = 1
            guard_loc = guard.set_position(self.grid)
            self.grid[guard_loc[0], guard_loc[1]] = guard.id

        for invader in self.invaders.values():
            invader.speed = 1
            invader_loc = invader.set_position(self.grid)
            self.grid[invader_loc[0], invader_loc[1]] = invader.id

        target_loc = self.target.set_position(self.grid)
        self.grid[target_loc[0], target_loc[1]] = self.target.id

    def reset(self):

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
    #
    #
    # def get_json_info(self):
    #     info = {'guard': {'id': self.guard.id, 'loc': self.guard.loc},
    #         'invader': {'id': self.invader.id, 'loc': self.invader.loc},
    #         'target': {'id': self.target.id, 'loc': self.target.loc},
    #         'grid': self.grid,
    #         'grid_size': self.grid.shape,
    #         'step_count': self.step_count,
    #         'done': self.done,
    #         'wins': self.wins}
    #
    #     return info
    #
    # def set_json_info(self):
    #     self.grid = np.copy(self._init_game_state['grid'])
    #
    #     self.guard.loc = self._init_game_state['guard']['loc']
    #     self.invader.loc = self._init_game_state['invader']['loc']
    #     self.targetloc = self._init_game_state['target']['loc']
    #
    #     self.step_count = self._init_game_state['step_count']
    #     self.done = self._init_game_state['done']
    #     self.wins = self._init_game_state['wins']

    def _get_rewards(self):
        """
         Reward for guards and invaders
        """

        if self.done:
            if self.wins == 1:
                #invader(s) catch the target
                guards_rewards = {}
                for guard in self.guards.values():
                    guards_rewards[guard.id] = 0 * guard.steps + (-1.0 / len(self.guards))

                invaders_rewards = {}
                for invader in self.invaders.values():
                    invaders_rewards[invader.id] = 0 * invader.steps + (1.0 / len(self.invaders))

                return guards_rewards, invaders_rewards

            if self.wins == 2:
                #guard(s) catch the target
                guards_rewards = {}
                for guard in self.guards.values():
                    guards_rewards[guard.id] = 0 * guard.steps + (1.0 / len(self.guards))

                invaders_rewards = {}
                for invader in self.invaders.values():
                    invaders_rewards[invader.id] = 0 * invader.steps + (-1.0 / len(self.invaders))

                return guards_rewards, invaders_rewards

            if self.wins == 3:
                #guards catch all the invaders
                guards_rewards = {}
                for guard in self.guards.values():
                    guards_rewards[guard.id] = 0 * guard.steps + (2.0 / len(self.guards))

                invaders_rewards = {}
                for invader in self.invaders.values():
                    invaders_rewards[invader.id] = 0 * invader.steps + (-2.0 / len(self.invaders))

                return guards_rewards, invaders_rewards

        guards_rewards = {}
        for guard in self.guards.values():
            guards_rewards[guard.id] = 0 * guard.steps

        invaders_rewards = {}
        for invader in self.invaders.values():
            invaders_rewards[invader.id] = 0 * invader.steps

        # if self.step_count == self.grid.shape[0] * self.grid.shape[1]:
        #     self.done = True
        #     self.wins = 0
        #     guards_rewards = {}
        #     for guard in self.guards.values():
        #         guards_rewards[guard.id] = -1 * guard.steps
        #
        #     invaders_rewards = {}
        #     for invader in self.invaders.values():
        #         invaders_rewards[invader.id] = -1 * invader.steps

        return guards_rewards, invaders_rewards

    def act(self):
        """
        Invader and guard act inside environment. Invader and guard new positions are updated.
        """
        invaders_actions = {}
        for invader in self.invaders.values():
            invaders_actions[invader.id] = invader.act(self)

        guards_actions = {}
        for guard in self.guards.values():
            guards_actions[guard.id] = guard.act(self)

        return guards_actions, invaders_actions

    def step(self, guards_actions, invaders_actions):
        """
        Given that our environment is static throught out the game, only things that this function has to caclulate
        are rewards for invaders and guards and game done or not
        """

        #check if any invader catches the target
        invader_caught_target = False
        for invader in invaders_actions.keys():
            if invaders_actions[invader][0] == self.target.loc[0] and invaders_actions[invader][1] == self.target.loc[1]:
                self.wins = 1
                self.done = True
                break

        #check if any guard catches the target
        for guard in guards_actions.keys():
            if guards_actions[guard][0] == self.target.loc[0] and guards_actions[guard][1] == self.target.loc[1]:
                self.wins = 2
                self.done = True
                break

        #check if all guards catch all the invaders
        for guard in guards_actions.keys():
            for invader in invaders_actions.keys():
                if guards_actions[guard][0] == invaders_actions[invader][0] and guards_actions[guard][1] == invaders_actions[invader][1]:
                    self.invaders[invader].speed = 0

        invaders_disabled = 0
        for invader in self.invaders.values():
            if invader.speed == 0:
                invaders_disabled += 1

        if invaders_disabled == len(invaders_actions):
            self.wins = 3
            self.done = True

        #update invaders locations
        for invader in self.invaders.values():
            if invader.speed:
                old_loc = invader.loc
                self.grid[invader.loc[0], invader.loc[1]] = 0
                self.grid[invaders_actions[invader.id][0], invaders_actions[invader.id][1]] = invader.id
                invader.loc = invaders_actions[invader.id]
                if invader.loc[0] != old_loc[0] and invader.loc[1] != old_loc[1]:
                    invader.steps += 1

        #update guards locations
        for guard in self.guards.values():
            old_loc = guard.loc
            self.grid[guard.loc[0], guard.loc[1]] = 0
            self.grid[guards_actions[guard.id][0], guards_actions[guard.id][1]] = guard.id
            guard.loc = guards_actions[guard.id]
            if guard.speed and guard.loc[0] != old_loc[0] and guard.loc[1] != old_loc[1]:
                guard.steps += 1

        self.step_count += 1

        return self.grid, self._get_rewards(), self.done

    def render(self):
        img = np.zeros((self.grid.shape[0], self.grid.shape[1], 3))
        img[self.grid == 1] = [255, 255, 255]

        img[self.grid == 7] = [255, 0, 0]
        img[self.grid == 8] = [0, 255, 0]
        img[self.grid == 9] = [0, 0, 255]

        #invaders rendering
        for invader in self.invaders.values():
            img[self.grid == invader.id] = [255, 0, 0]
            invader_observation = invader.get_observed_environment(self)
            img[invader_observation == 0] = [128,5,5]

        #guards rendering
        for guard in self.guards.values():
            img[self.grid == guard.id] = [0, 255, 0]
            guard_observation = guard.get_observed_environment(self)
            img[guard_observation == 0] = [5,128,5]

        plt.imshow(img.astype(np.uint8))
        # cv2.imwrite('test.png', img)
        plt.pause(self.sim_speed)
        plt.draw()
        # return img
