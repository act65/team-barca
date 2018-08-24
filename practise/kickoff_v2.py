import gym
import gym_soccer

import numpy as np
import multi_player

import tensorflow as tf

import argparse
import ray

def argumentparser():
    parser = argparse.ArgumentParser(description='Play soccer')
    parser.add_argument('--max_iters', type=int, default=10000,
                        help='number of trials')
    parser.add_argument('--logdir', type=str, default='/tmp/test/',
                        help='location to save logs')
    parser.add_argument('--task', type=str, default='SoccerEmptyGoal-v0',
                        choices=['Soccer-v0', 'SoccerEmptyGoal-v0', 'SoccerAgainstKeeper-v0', 'SoccerMatch-v0'])
    return parser.parse_args()

class GymEnvironment():
    # seems a little silly, aleady running in a different process...
    def __init__(self, name):
        self.env = gym.make(name)
        self.env.reset()

    def step(self, action):
        action = np.argmax(action[:, :3], axis=1).tolist() + action[0, 3:].tolist()
        obs, r, done, info = self.env.step(action)
        if done:
            obs = self.reset()
            r = 0
        return obs, r

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

def main(args):
    ray.init()
    Node = ray.remote(multi_player.Node)
    player1 = Node.remote(59, 32, logdir=args.logdir+'1')
    player2 = Node.remote(32, 8, logdir=args.logdir+'2')
    player3 = Node.remote(32, 32, logdir=args.logdir+'3')

    env = GymEnvironment(args.task)
    env.render()

    observation = env.reset(); done = False; reward = 0
    old_h = [0.0] * 32
    for _ in range(args.max_iters):
        h = player1.call.remote(observation, reward)
        z = player3.call.remote([a + b for a, b in zip(ray.get(h), old_h)], reward)
        action = player2.call.remote(z, reward)
        old_h = ray.get(h)

        observation, reward = env.step(ray.get(action))

if __name__ == '__main__':
    main(argumentparser())
