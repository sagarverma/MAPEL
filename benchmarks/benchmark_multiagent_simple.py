import sys
sys.path.append('../environments')
from multi_env import *
from multi_agent import *

from multiprocessing import Queue
import cv2
import random


def baseline_testing(grid_size):
    invader1 = Invader(id=81, speed=1, obs_size=3)
    invader2 = Invader(id=82, speed=1, obs_size=3)
    invader3 = Invader(id=83, speed=1, obs_size=3)
    guard1 = Guard(id=71, speed=1, obs_size=3)
    guard2 = Guard(id=72, speed=1, obs_size=3)
    guard3 = Guard(id=73, speed=1, obs_size=3)
    target = Target(speed=0)

    env = Environment(grid_size, [guard1, guard2], [invader1, invader2], target, sim_speed=0.01)

    imgs = []
    done = False
    while not done:
        guard_action, invader_action = env.act()
        obs, reward, done = env.step(guard_action, invader_action)
        img = env.render()
        imgs.append(img)

    return env.wins, imgs

guard_wins = 0
for i in range(100):
    wins, imgs = baseline_testing([32,32])
    print (wins)
    if wins == 2 or wins == 3:
        guard_wins += 1

    # if wins == 1:
    #     for i in range(len(imgs)):
    #         cv2.imwrite('../results/invader_wins_multi_simple/'  + str(i).zfill(10) + '.png', imgs[i])

    # if wins == 2:
    #     for i in range(len(imgs)):
    #         cv2.imwrite('../results/guard_wins_multi_simple/'  + str(i).zfill(10) + '.png', imgs[i])

    # if wins == 3:
    #     for i in range(len(imgs)):
    #         cv2.imwrite('../results/guard_wins_all_multi_simple/'  + str(i).zfill(10) + '.png', imgs[i])


print (guard_wins)
