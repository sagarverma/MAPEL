from multiprocessing import Queue
from multi_env import *
from multi_agent import *
import cv2
import random


def baseline_testing(grid_size):
    invader1 = Invader(id=81, speed=1, obs_size=5)
    invader2 = Invader(id=82, speed=1, obs_size=5)
    guard1 = Guard(id=71, speed=1, obs_size=5)
    guard2 = Guard(id=72, speed=1, obs_size=5)
    target = Target(speed=0)

    env = Environment(grid_size, [guard1, guard2], [invader1, invader2], target, sim_speed=0.0001)


    done = False
    while not done:
        guard_action, invader_action = env.act()
        obs, reward, done = env.step(guard_action, invader_action)
        #env.render()

    return env.wins

guard_wins = 0
for i in range(100):
    wins = baseline_testing([32,32])
    if wins == 2 or wins == 3:
        guard_wins += 1
print (guard_wins)
