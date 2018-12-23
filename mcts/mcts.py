import argparse
import multiprocessing
from queue import Empty
import numpy as np
import time

from env import Invader, Guard, Target, Environment, Agent

NUM_AGENTS = 3
NUM_ACTIONS = 9

def action_to_loc(current_loc, action):
    if action == 0:
        return [current_loc[0] - 1, current_loc[1] - 1]
    if action == 1:
        return [current_loc[0] - 1, current_loc[1]]
    if action == 2:
        return [current_loc[0] - 1, current_loc[1] + 1]
    if action == 3:
        return [current_loc[0], current_loc[1] - 1]
    if action == 4:
        return current_loc
    if action == 5:
        return [current_loc[0], current_loc[1] + 1]
    if action == 6:
        return [current_loc[0] + 1, current_loc[1] - 1]
    if action == 7:
        return [current_loc[0] + 1, current_loc[1]]
    if action == 8:
        return [current_loc[0] + 1, current_loc[1] + 1]

def valid_loc(grid, loc):
    if loc[0] > 0 and loc[1] > 0 and loc[0] < grid.shape[0] and loc[1] < grid.shape[1]:
        if grid[loc[0], loc[1]] != 1:
            return True

    return False

class NEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super(NEnvironment, self).__init__(*args, **kwargs)

    def act(self):
        """
        Invader and guard act inside environment. Invader and guard new positions are updated.
        """
        invader_action = self.invader.act(self.grid, self.target)

        return invader_action



def argmax_tiebreaking(Q):
    # find the best action with random tie-breaking
    idx = np.flatnonzero(Q == np.max(Q))
    assert len(idx) > 0, str(Q)
    return np.random.choice(idx)

class MCTSNode(object):
    def __init__(self, p):
        # values for 6 actions
        self.Q = np.zeros(NUM_ACTIONS)
        self.W = np.zeros(NUM_ACTIONS)
        self.N = np.zeros(NUM_ACTIONS, dtype=np.uint32)
        assert p.shape == (NUM_ACTIONS,)
        self.P = p

    def action(self):
        U = args.mcts_c_puct * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
        return argmax_tiebreaking(self.Q + U)

    def update(self, action, reward):
        self.W[action] += reward
        self.N[action] += 1
        self.Q[action] = self.W[action] / self.N[action]

    def probs(self, temperature=1):
        if temperature == 0:
            p = np.zeros(NUM_ACTIONS)
            p[argmax_tiebreaking(self.N)] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            return Nt / np.sum(Nt)


class MCTSGuard(Guard):
    def __init__(self, *args, **kwargs):
        super(MCTSGuard, self).__init__(*args, **kwargs)
        self.env = self.make_env()
        self.reset_tree()

    def make_env(self):
        invader = Invader(speed=1)
        guard = self
        target = Target(speed=0)

        env = NEnvironment([32, 32], guard, invader, target, fixed_grid=True)

        return env

    def reset_tree(self):
        self.tree = {}

    def search(self, root, num_iters, temperature=1):
        # remember current game state
        self.env._init_game_state = root
        root = str(root)

        for i in range(num_iters):
            # restore game state to root node
            obs = self.env.reset()
            # serialize game state
            state = str(self.env.get_json_info())

            trace = []
            done = False
            while not done:
                if state in self.tree:
                    node = self.tree[state]
                    # choose actions based on Q + U
                    action = node.action()
                    trace.append((node, action))
                else:
                    # use unfiform distribution for probs
                    probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS

                    # use current rewards for values
                    reward = self.env._get_rewards()

                    # add new node to the tree
                    self.tree[state] = MCTSNode(probs)

                    # stop at leaf node
                    break

                # make other agents act
                invader_action = self.env.act()
                # add my action to list of actions
                guard_action = self.act(obs, action)
                # step environment forward
                obs, reward, done, info = self.env.step(guard_action, invader_action)
                # fetch next state
                state = str(self.env.get_json_info())

            # update tree nodes with rollout results
            for node, action in reversed(trace):
                node.update(action, reward)
                reward *= args.discount

        # reset env back where we were
        self.env.set_json_info()
        self.env._init_game_state = None
        # return action probabilities
        return self.tree[root].probs(temperature)

    def rollout(self):
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        # guarantees that we are not called recursively
        # and episode ends when this agent dies
        obs = self.env.reset()

        length = 0
        done = False
        while not done:
            if args.render:
               self.env.render()

            root = self.env.get_json_info()
            # do Monte-Carlo tree search
            pi = self.search(root, args.mcts_iters, args.temperature)
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)

            # make other agents act
            invader_action = self.env.act()
            # add my action to list of actions
            guard_action = self.act(obs, action)
            # step environment forward
            obs, reward, done, info = self.env.step(guard_action, invader_action)
            length += 1
            print("Agent:", self.id, "Step:", length, "Action:", action, "Probs:", [round(p, 2) for p in pi], "Reward:", reward, "Done:", done)

        return length, reward

    def act(self, obs, action):
        new_loc = action_to_loc(self.loc, action)
        if valid_loc(obs, new_loc):
            return new_loc
        else:
            return self.loc


def runner(num_episodes, fifo, _args):
    # make args accessible to MCTSAgent
    global args
    args = _args
    # make sure agents play at all positions
    agent = MCTSGuard()

    for i in range(num_episodes):
        # do rollout
        start_time = time.time()
        length, reward = agent.rollout()
        elapsed = time.time() - start_time
        # add data samples to log
        fifo.put((length, reward, elapsed))


def profile_runner(num_episodes, fifo, _args):
    import cProfile
    command = """runner(num_episodes, fifo, _args)"""
    cProfile.runctx(command, globals(), locals(), filename=_args.profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile')
    parser.add_argument('--render', action="store_true", default=False)
    # runner params
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_runners', type=int, default=1)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=50)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    # RL params
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()

    # use spawn method for starting subprocesses
    ctx = multiprocessing.get_context('spawn')

    # create fifos and processes for all runners
    fifo = ctx.Queue()
    for i in range(args.num_runners):
        process = ctx.Process(target=profile_runner if args.profile else runner, args=(args.num_episodes // args.num_runners, fifo, args))
        process.start()

    # do logging in the main process
    all_rewards = []
    all_lengths = []
    all_elapsed = []
    for i in range(args.num_episodes):
        # wait for a new trajectory
        length, reward, elapsed = fifo.get()

        print("Episode:", i, "Reward:", reward, "Length:", length, "Time per step:", elapsed / length)
        all_rewards.append(reward)
        all_lengths.append(length)
        all_elapsed.append(elapsed)

    print("Average reward:", np.mean(all_rewards))
    print("Average length:", np.mean(all_lengths))
    print("Time per timestep:", np.sum(all_elapsed) / np.sum(all_lengths))