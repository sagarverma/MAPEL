from multiprocessing import Queue
from env import *
import cv2
import random


def baseline_testing(grid_size):
    invader = Invader(speed=1)
    guard = Guard(speed=1)
    target = Target(speed=0)

    env = Environment(grid_size, guard, invader, target)

    done = False
    while not done:
        guard_action, invader_action = env.act()
        obs, reward, done, info = env.step(guard_action, invader_action)
        env.render()

    return env.wins

guard_wins = 0
for i in range(1):
    wins = baseline_testing([32,32])
    if wins == 'guard':
        guard_wins += 1
print (guard_wins)