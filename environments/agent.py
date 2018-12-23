import numpy as np
import random
import time

import matplotlib.pyplot as plt
import cv2

from shapely.geometry import Point
from shapely.geometry import box

import dungeonGenerator
from graph import Graph, NxGraph

class Agent(object):

    def __init__(self, id, speed=0, spawn_loc=None, obs_size=None):
        self.id = id
        self.speed = speed
        self.loc = spawn_loc
        self.obs_size = obs_size
        self.history_pos = []

    def set_position(self, environment):
        """
            set position of this agent on the environment, fix x and y location, called during instanciation of
            this class and when agent moves to a new location, can be used to randomly spawn the agent or spawn at
            a user provided location on grid, it is assumed that when used provides the location it is free on the
            environment
        """

        free_space = np.where(environment==0)
        total_free_space = free_space[0].shape[0]
        random_loc = random.randint(0, total_free_space-1)
        x = free_space[0][random_loc]
        y = free_space[1][random_loc]
        self.loc = [x, y]

        return self.loc

    def get_observed_environment(self, environment):
        """
            create observation space for this agent based on environment passed, if partial observation is allowed
            then a partial observation space is created
        """
        if self.obs_size:
            observed_environment = np.copy(environment)
            agent_observes =  np.copy(observed_environment[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                            max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size])

            observed_environment[observed_environment == 0] = 1
            observed_environment[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                            max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size] = np.copy(agent_observes)

            return observed_environment
        else:
            return environment

    def _valid_loc(self, grid, loc):
        if loc[0] > 0 and loc[1] > 0 and loc[0] < grid.shape[0] and loc[1] < grid.shape[1]:
            if grid[loc[0], loc[1]] != 1:
                return True

        return False

    def _get_adjacent_locs(self, current_loc):
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

    def _loc_online_inbetween(self, loc, source, target):
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

    def _target_inside_obs_space(self, target):
        target_point = Point(*target.loc)
        obs_box = box(self.loc[0] - self.obs_size, self.loc[1] - self.obs_size, self.loc[0] + self.obs_size, self.loc[1] + self.obs_size)
        return obs_box.contains(target_point)

    def _get_ordered_empty_adjacent_locs(self, empty_adjacent_locs, loc):
       empty_adjacent_locs.sort(key = lambda p: ((p[0]- loc[0])**2 + (p[1] - loc[1])**2)**0.5)
       return empty_adjacent_locs

    def _action_on_out_of_los(self, environment, target):
        adjacent_locs = self._get_adjacent_locs(self.loc)
        empty_adjacent_locs = [adjacent_loc for adjacent_loc in adjacent_locs if self._valid_loc(environment, adjacent_loc)]
        empty_adjacent_locs = self._get_ordered_empty_adjacent_locs(empty_adjacent_locs, target.loc)

        new_loc = None
        for adjacent_loc in empty_adjacent_locs:
            if adjacent_loc[0] != self.loc[0] and adjacent_loc[1] != self.loc[1]:
                if self._loc_online_inbetween(adjacent_loc, self.loc, target.loc):
                    new_loc = adjacent_loc
                    break

        if not new_loc:
            new_loc = random.choice(empty_adjacent_locs)

        return new_loc

    def _action_on_los(self, environment, target):
        graph = NxGraph()
        graph.grid_to_graph(environment)
        shortest_path = graph.shortest_path((self.loc[0], self.loc[1]), (target.loc[0], target.loc[1]))

        if len(shortest_path) <= self.speed:
            return [shortest_path[-1][0], shortest_path[-1][1]]
        else:
            return [shortest_path[self.speed][0], shortest_path[self.speed][1]]

    def act(self, environment, target):
        observerd_environment = self.get_observed_environment(environment)

        if not self.obs_size:
            return self._action_on_los(observerd_environment, target)
        else:
            if self._target_inside_obs_space(target):
                return self._action_on_los(observerd_environment, target)
            else:
                return self._action_on_out_of_los(observerd_environment, target)


class Invader(Agent):

    def __init__(self, id=7, speed=1, *args, **kwargs):
        super(Invader, self).__init__(id, speed, *args, **kwargs)


    '''Implement specific functions for guard'''

class Guard(Agent):

    def __init__(self, id=8, speed=1, *args, **kwargs):
        super(Guard, self).__init__(id, speed, *args, **kwargs)

    '''Implement specific functions for guard'''

class Target(Agent):

    def __init__(self, id=9, speed=0, *args, **kwargs):
        super(Target, self).__init__(id, speed, *args, **kwargs)

    '''Implement specific functions for target'''
