from multiprocessing import Queue
from env import *
import cv2
import random


def baseline_testing(grid_size):
    invader = Invader(speed=1, obs_size=5)
    guard = Guard(speed=1, obs_size=5)
    target = Target(speed=0)

    env = Environment(grid_size, guard, invader, target, sim_speed=0.0001)

    done = False
    i = 0
    while not done:
        guard_action, invader_action = env.act()
        obs, reward, done, info = env.step(guard_action, invader_action)
        #env.render()

        # if i == 100:
        #     break

        i += 1
    return env.wins

guard_wins = 0
for i in range(1000):
    wins = baseline_testing([32,32])
    if wins == 'guard':
        guard_wins += 1
print (guard_wins)
