import numpy as np
import random
import time
import copy

import matplotlib.pyplot as plt
import cv2

from shapely.geometry import Point
from shapely.geometry import box

import dungeonGenerator
from graph import Graph, NxGraph

from utils.folder import populate_adjacent_locs2, print_grid

class Agent(object):

    def __init__(self, id, speed=0, spawn_loc=None, obs_size=None):
        self.id = id
        self.speed = speed
        self.steps = 0
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
            observed_environment = np.copy(environment.grid)
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

    def act(self, environment):
        if self.speed:
            observerd_environment = self.get_observed_environment(environment)
            target = environment.target

            if not self.obs_size:
                return self._action_on_los(observerd_environment, target)
            else:
                if self._target_inside_obs_space(target):
                    return self._action_on_los(observerd_environment, target)
                else:
                    return self._action_on_out_of_los(observerd_environment, target)
        else:
            return self.loc

class Invader(Agent):

    def __init__(self, id=7, speed=1, *args, **kwargs):
        super(Invader, self).__init__(id, speed, *args, **kwargs)

    def get_observed_environment(self, environment):
        """
            create observation space for this agent based on environment passed, if partial observation is allowed
            then a partial observation space is created
        """
        if self.obs_size:
            observed_environment = np.copy(environment.grid)
            agent_observes =  np.copy(observed_environment[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                            max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size])

            observed_environment[observed_environment == 0] = 1

            for other_invader in environment.invaders.values():
                observed_environment[other_invader.loc[0], other_invader.loc[1]] = other_invader.id

            observed_environment[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                            max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size] = np.copy(agent_observes)

            return observed_environment
        else:
            return environment.grid

    def get_featurised_observed_environment(self, environment):
        observed_environment = np.lib.pad(np.copy(environment.grid), (self.obs_size, self.obs_size), 'constant', constant_values=-1)
        agent_observes =  np.copy(observed_environment[max((self.loc[0] + self.obs_size - self.obs_size), 0) : self.loc[0] + self.obs_size + self.obs_size,
                                        max((self.loc[1] + self.obs_size - self.obs_size), 0) : self.loc[1] + self.obs_size + self.obs_size])

        obsspace = np.copy(agent_observes)
        obsspace[agent_observes > 1] = 0

        selfpos = np.zeros(agent_observes.shape)
        selfpos[agent_observes == self.id] = 1

        otherspos = np.zeros(agent_observes.shape)
        for other in environment.invaders.values():
            if other.id != self.id and np.any(agent_observes == other.id):
                otherspos[agent_observes == other.id] = 1

        guardspos = np.zeros(agent_observes.shape)
        for guard in environment.guards.values():
            if np.any(agent_observes == guard.id):
                guardspos[agent_observes == guard.id] = 1

        targetpos = np.zeros(agent_observes.shape)
        if np.any(agent_observes == environment.target.id):
            targetpos[agent_observes == environment.target.id] = 1

        features = np.asarray([obsspace, selfpos, otherspos, guardspos, targetpos])

        return features.flatten()

    def get_full_feat(self, environment):
        observed_environment = np.copy(environment.grid)

        agent_observes =  np.copy(observed_environment[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                        max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size])

        maps = []
        nmap = np.ones(observed_environment.shape)
        nmap[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                        max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size] = np.copy(agent_observes)
        nmap[nmap > 1] = 0
        maps.append(nmap)
        #print_grid(nmap)
        #cv2.imwrite('obs.png', nmap*255)

        # print_grid(populate_adjacent_locs2(self.loc, observed_environment.shape))
        # cv2.imwrite('self.png', populate_adjacent_locs2(self.loc, observed_environment.shape)*255)
        maps.append(populate_adjacent_locs2(self.loc, observed_environment.shape))

        othersspace = np.ones(observed_environment.shape)
        for other in environment.invaders.values():
            if other.id != self.id:
                othersspace = np.logical_and(othersspace, populate_adjacent_locs2(other.loc, observed_environment.shape))
        maps.append(othersspace)
        # print_grid(othersspace)
        # cv2.imwrite('others.png', othersspace*255)

        # print_grid(populate_adjacent_locs2(environment.target.loc, observed_environment.shape))
        # cv2.imwrite('target.png', populate_adjacent_locs2(environment.target.loc, observed_environment.shape)*255)
        maps.append(populate_adjacent_locs2(environment.target.loc, observed_environment.shape))

        nmap = np.ones(observed_environment.shape)
        nmap[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                        max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size] = np.copy(agent_observes)
        guardsspace = np.ones(observed_environment.shape)
        for guard in environment.guards.values():
            if np.any(nmap == guard.id):
                guardsspace = np.logical_and(guardsspace, populate_adjacent_locs2(guard.loc, observed_environment.shape))
        maps.append(guardsspace)
        # print_grid(guardsspace)
        # cv2.imwrite('ops.png',guardsspace*255)
        # exit()
        return np.asarray(maps)

class Guard(Agent):

    def __init__(self, id=8, speed=1, *args, **kwargs):
        super(Guard, self).__init__(id, speed, *args, **kwargs)

    def get_observed_environment(self, environment):
        """
            create observation space for this agent based on environment passed, if partial observation is allowed
            then a partial observation space is created
        """
        if self.obs_size:
            observed_environment = np.copy(environment.grid)
            agent_observes =  np.copy(observed_environment[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                            max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size])

            observed_environment[observed_environment == 0] = 1

            for other_guard in environment.guards.values():
                observed_environment[other_guard.loc[0], other_guard.loc[1]] = other_guard.id

            observed_environment[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                            max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size] = np.copy(agent_observes)

            return observed_environment
        else:
            return environment.grid

    def get_featurised_observed_environment(self, environment):
        observed_environment = np.lib.pad(np.copy(environment.grid), (self.obs_size, self.obs_size), 'constant', constant_values=-1)
        agent_observes =  np.copy(observed_environment[max((self.loc[0] + self.obs_size - self.obs_size), 0) : self.loc[0] + self.obs_size + self.obs_size,
                                        max((self.loc[1] + self.obs_size - self.obs_size), 0) : self.loc[1] + self.obs_size + self.obs_size])

        obsspace = np.copy(agent_observes)
        obsspace[agent_observes > 1] = 0

        selfpos = np.zeros(agent_observes.shape)
        selfpos[agent_observes == self.id] = 1

        otherspos = np.zeros(agent_observes.shape)
        for other in environment.guards.values():
            if other.id != self.id and np.any(agent_observes == other.id):
                otherspos[agent_observes == other.id] = 1

        invaderspos = np.zeros(agent_observes.shape)
        for invader in environment.invaders.values():
            if np.any(agent_observes == invader.id):
                invaderspos[agent_observes == invader.id] = 1

        targetpos = np.zeros(agent_observes.shape)
        if np.any(agent_observes == environment.target.id):
            targetpos[agent_observes == environment.target.id] = 1

        features = np.asarray([obsspace, selfpos, otherspos, invaderspos, targetpos])

        return features.flatten()

    def get_full_feat(self, environment):
        observed_environment = np.copy(environment.grid)

        agent_observes =  np.copy(observed_environment[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                        max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size])

        maps = []
        nmap = np.ones(observed_environment.shape)
        nmap[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                        max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size] = np.copy(agent_observes)
        nmap[nmap > 1] = 0
        maps.append(nmap)
        #print_grid(nmap)
        #cv2.imwrite('obs.png', nmap*255)

        # print_grid(populate_adjacent_locs2(self.loc, observed_environment.shape))
        # cv2.imwrite('self.png', populate_adjacent_locs2(self.loc, observed_environment.shape)*255)
        maps.append(populate_adjacent_locs2(self.loc, observed_environment.shape))

        othersspace = np.ones(observed_environment.shape)
        for other in environment.guards.values():
            if other.id != self.id:
                othersspace = np.logical_and(othersspace, populate_adjacent_locs2(other.loc, observed_environment.shape))
        maps.append(othersspace)
        # print_grid(othersspace)
        # cv2.imwrite('others.png', othersspace*255)

        # print_grid(populate_adjacent_locs2(environment.target.loc, observed_environment.shape))
        # cv2.imwrite('target.png', populate_adjacent_locs2(environment.target.loc, observed_environment.shape)*255)
        maps.append(populate_adjacent_locs2(environment.target.loc, observed_environment.shape))

        nmap = np.ones(observed_environment.shape)
        nmap[max((self.loc[0] - self.obs_size), 0) : self.loc[0] + self.obs_size,
                                        max((self.loc[1] - self.obs_size), 0) : self.loc[1] + self.obs_size] = np.copy(agent_observes)
        invadersspace = np.ones(observed_environment.shape)
        for invader in environment.invaders.values():
            if np.any(nmap == invader.id):
                invadersspace = np.logical_and(invadersspace, populate_adjacent_locs2(invader.loc, observed_environment.shape))
        maps.append(invadersspace)
        # print_grid(guardsspace)
        # cv2.imwrite('ops.png',guardsspace*255)
        # exit()
        return np.asarray(maps)

class Target(Agent):

    def __init__(self, id=9, speed=0, *args, **kwargs):
        super(Target, self).__init__(id, speed, *args, **kwargs)

    '''Implement specific functions for target'''
