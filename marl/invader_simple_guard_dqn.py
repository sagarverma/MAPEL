
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multi_env import *
from multi_agent import *
import csv

# hyper parameters
EPISODES = 200  # number of episodes
EPS_START = 1.0  # e-greedy threshold start value
EPS_END = 0.1  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.99  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
BATCH_SIZE = 32  # Q-learning batch size
num_agents = 2

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def action_to_loc(current_loc, action):
    return get_adjacent_locs(current_loc)[action]

def valid_loc(grid, loc):
    if loc[0] >= 0 and loc[1] >= 0 and loc[0] < grid.shape[0] and loc[1] < grid.shape[1]:
        if grid[loc[0], loc[1]] != 1:
            return True

    return False

def get_adjacent_locs(current_loc):
    adjacent_locs = [[current_loc[0] - 1, current_loc[1] - 1],
                    [current_loc[0] - 1, current_loc[1]],
                    [current_loc[0] - 1, current_loc[1] + 1],
                    [current_loc[0], current_loc[1] - 1],
                    [current_loc[0], current_loc[1] + 1],
                    [current_loc[0] + 1, current_loc[1] - 1],
                    [current_loc[0] + 1, current_loc[1]],
                    [current_loc[0] + 1, current_loc[1] + 1],
                    current_loc]
    return adjacent_locs

class NEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super(NEnvironment, self).__init__(*args, **kwargs)

    def act(self):
        """
        Invader and guard act inside environment. Invader and guard new positions are updated.
        """
        invaders_actions = {}
        for invader in self.invaders.values():
            invaders_actions[invader.id] = invader.act(self)

        guards_actions = {}
        # for guard in self.guards.values():
        #     guards_actions[guard.id] = guard.act(self)

        return guards_actions, invaders_actions

class DQNGuard(Guard):
    def __init__(self, *args, **kwargs):
        super(DQNGuard, self).__init__(*args, **kwargs)

    def act(self, environment, action):
        new_loc = action_to_loc(self.loc, action)
        if valid_loc(environment.grid, new_loc) and new_loc[0] != self.loc[0] and new_loc[1] != self.loc[1]:
            return new_loc

        return self.loc

# class DQNInvader(Invader):
#     def __init__(self, *args, **kwargs):
#         super(DQNInvader, self).__init__(*args, **kwargs)
#
#     def act(self, environment, action):
#         new_loc = action_to_loc(self.loc, action)
#         if valid_loc(environment.grid, new_loc) and new_loc[0] != self.loc[0] and new_loc[1] != self.loc[1]:
#             return new_loc
#
#         return self.loc

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class CNNet(nn.Module):
    def __init__(self, in_channels=5, num_actions=9):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        #print(x.size())
        x = x.view(-1, 256)
        x = F.relu(self.fc2(x))
        return x

class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()
        nn.Module.__init__(self)
        self.l1 = nn.Linear(500, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, 32)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x

class SRPNet(nn.Module):
    def __init__(self, num_agents):
        super(SRPNet, self).__init__()
        self.num_agents = num_agents

        self.feature = FeatNet()

        self.ll1 = nn.Linear(num_agents + 1, 32)
        self.rnn = nn.RNNCell(64, 32)
        self.ll2 = nn.Linear(32, 9)

    def forward(self, inp, vis, hid):
        feature = self.feature(inp)

        ll1_out = self.ll1(vis)
        rnn_inp = torch.cat([feature, ll1_out], dim=1)
        out = self.rnn(rnn_inp)
        ll2_out = self.ll2(out)

        return out, ll2_out




def select_action(model, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        return model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randint(0,8)]])


def run_episode(e, environment):
    global steps_done

    # invader_states = {}
    # for invader in environment.invaders.values():
    #     invader_states[invader.id] = invader.get_full_feat(environment)

    guard_states = {}
    for guard in environment.guards.values():
        guard_states[guard.id] = guard.get_full_feat(environment)

    steps = 0

    while True:
        #environment.render()
        # invader_actions = {}
        # for invader in invader_models.keys():
        #     invader_actions[invader] = select_action(invader_models[invader], FloatTensor([invader_states[invader]]))
        steps_done += 1
        guard_actions = {}
        for guard in guard_models.keys():
            guard_actions[guard] = select_action(guard_models[guard], FloatTensor([guard_states[guard]]))

        # invader_action_locs = {}
        # for invader in environment.invaders.values():
        #     invader_action_locs[invader.id] = invader.act(environment, invader_actions[invader.id])

        guard_action_locs = {}
        for guard in environment.guards.values():
            guard_action_locs[guard.id] = guard.act(environment, guard_actions[guard.id])

        _, invader_action_locs = environment.act()
        next_state, rewards, done = environment.step(guard_action_locs, invader_action_locs)
        # environment.render()

        # invader_next_states = {}
        # for invader in environment.invaders.values():
        #     invader_next_states[invader.id] = invader.get_full_feat(environment)

        guard_next_states = {}
        for guard in environment.guards.values():
            guard_next_states[guard.id] = guard.get_full_feat(environment)


        # for invader in invader_memories.keys():
        #     invader_memories[invader].push((FloatTensor([invader_states[invader]]),
        #                  invader_actions[invader],  # action is already a tensor
        #                  FloatTensor([invader_next_states[invader]]),
        #                  FloatTensor([rewards[1][invader]])))

        for guard in guard_memories.keys():
            guard_memories[guard].push((FloatTensor([guard_states[guard]]),
                         guard_actions[guard],  # action is already a tensor
                         FloatTensor([guard_next_states[guard]]),
                         FloatTensor([rewards[0][guard]])))

        # for invader in invader_models.keys():
        #     invader_learn(invader)

        for guard in guard_models.keys():
            guard_learn(guard)

        # invader_states = {}
        # for invader in invader_next_states.keys():
        #     invader_states[invader] = invader_next_states[invader]

        guard_states = {}
        for guard in guard_next_states.keys():
            guard_states[guard] = guard_next_states[guard]

        steps += 1

        if done:
            episode_durations.append(steps)
            #plot_durations()
            break


def guard_learn(guard):
    if len(guard_memories[guard]) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = guard_memories[guard].sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = guard_models[guard](batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = guard_models[guard](batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values.view(-1), expected_q_values)

    # backpropagation of loss to NN
    guard_optimizers[guard].zero_grad()
    loss.backward()
    guard_optimizers[guard].step()


# def invader_learn(invader):
#     if len(invader_memories[invader]) < BATCH_SIZE:
#         return
#
#     # random transition batch is taken from experience replay memory
#     transitions = invader_memories[invader].sample(BATCH_SIZE)
#     batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
#
#     batch_state = Variable(torch.cat(batch_state))
#     batch_action = Variable(torch.cat(batch_action))
#     batch_reward = Variable(torch.cat(batch_reward))
#     batch_next_state = Variable(torch.cat(batch_next_state))
#
#     # current Q values are estimated by NN for all actions
#     current_q_values = invader_models[invader](batch_state).gather(1, batch_action)
#     # expected Q values are estimated from actions which gives maximum Q value
#     max_next_q_values = invader_models[invader](batch_next_state).detach().max(1)[0]
#     expected_q_values = batch_reward + (GAMMA * max_next_q_values)
#
#     # loss is measured from error between current and newly expected Q values
#     loss = F.smooth_l1_loss(current_q_values.view(-1), expected_q_values)
#
#     # backpropagation of loss to NN
#     invader_optimizers[invader].zero_grad()
#     loss.backward()
#     invader_optimizers[invader].step()

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated



# invader_models = {80: CNNet().cuda(0), 81: CNNet().cuda(0)}
guard_models = {70: CNNet().cuda(0), 71: CNNet().cuda(0)}


# invader_memories = {80: ReplayMemory(1000000), 81: ReplayMemory(1000000)}
guard_memories = {70: ReplayMemory(1000000), 71: ReplayMemory(1000000)}

# invader_optimizers = {80: optim.Adam(invader_models[80].parameters(), LR), 81: optim.Adam(invader_models[81].parameters(), LR)}
guard_optimizers = {70: optim.Adam(guard_models[70].parameters(), LR), 71: optim.Adam(guard_models[71].parameters(), LR)}

steps_done = 0
episode_durations = []

w = csv.writer(open('logs/invader_simple_guards_dqn.csv','wb'))

who_wins = {0: 'No One', 1: 'Wow, Invaders!', 2: 'Wow, Guards!', 3:' Woah Guards!'}
for e in range(EPISODES):
    invader1 = Invader(id=80, speed=1, obs_size=5)
    invader2 = Invader(id=81, speed=1, obs_size=5)
    guard1 = DQNGuard(id=70, speed=1, obs_size=5)
    guard2 = DQNGuard(id=71, speed=1, obs_size=5)
    target = Target(speed=0)

    env = NEnvironment([32,32], [guard1, guard2], [invader1, invader2], target, sim_speed=0.00001)

    run_episode(e, env)

    rewards = env._get_rewards()
    print 'Guard 1:', rewards[0][70], 'Guard 2:', rewards[0][71], 'Invader 1:', rewards[1][80], 'Invader 2:', rewards[1][81], who_wins[env.wins]
    w.writerow([rewards[0][70], rewards[0][71], rewards[1][80],  rewards[1][81], who_wins[env.wins]])

torch.save(guard_models[70], 'weights/guard1_against_simple_invaders.pt')
torch.save(guard_models[71], 'weights/guard2_against_simple_invaders.pt')

print('Complete')
# plt.ioff()
# plt.show()
