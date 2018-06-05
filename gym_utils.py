import gym
import numpy as np
import operator
from IPython.display import clear_output
from time import sleep
from gym.spaces.tuple_space import Tuple
from gym.envs.registration import register
import random
import itertools
import tqdm


tqdm.monitor_interval = 0

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=200
)

register(
    id='FrozenLakeNotSlippery8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False},
    max_episode_steps=200
)


fl_slippery = {
    'small': 'FrozenLake-v0',
    'big': 'FrozenLake8x8-v0'
}

fl_not_slippery = {
    'small': 'FrozenLakeNotSlippery-v0',
    'big': 'FrozenLakeNotSlippery8x8-v0'
}


def create_environment(slippery=False, big=False):
    if slippery:
        env = gym.make(fl_slippery['big'] if big else fl_slippery['small'])
    else:
        env = gym.make(fl_not_slippery['big'] if big else fl_not_slippery['small'])
    env.reset()
    return env

def create_random_policy(env):
    policy = {}
    for key in range(0, env.observation_space.n):
        current_end = 0
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1 / env.action_space.n
        policy[key] = p
    return policy


def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
        Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q

def run_game(env, policy, display=True):
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s

        if display:
            clear_output(True)
            env.render()
            sleep(0.1)

        timestep = []
        timestep.append(s)

        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        for prob in policy[s].items():
            top_range += prob[1]
            if n < top_range:
                action = prob[0]
                break

        state, reward, finished, info =  env.step(action)
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

    if display:
        clear_output(True)
        env.render()
        sleep(0.05)

    return episode

def test_policy(policy, env):
    wins = 0
    r = 100
    for i in range(r):
        w = run_game(env, policy, display=False)[-1][-1]
        if w == 1:
            wins += 1
    return wins / r
