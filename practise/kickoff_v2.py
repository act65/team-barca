import gym
import gym_soccer

import numpy as np
import offline_learner

import tensorflow as tf

import argparse
import ray

def argumentparser():
    parser = argparse.ArgumentParser(description='Play soccer')
    parser.add_argument('--max_iters', type=int, default=1000000,
                        help='number of steps')
    parser.add_argument('--logdir', type=str, default='/tmp/test/',
                        help='location to save logs')
    parser.add_argument('--task', type=str, default='SoccerEmptyGoal-v0',
                        choices=['Soccer-v0', 'SoccerEmptyGoal-v0', 'SoccerAgainstKeeper-v0', 'SoccerMatch-v0'])
    return parser.parse_args()

class GymEnvironment():
    def __init__(self, name):
        self.env = gym.make(name)

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
    Node = ray.remote(offline_learner.ActorCritic)
    player1 = Node.remote(59, 32, logdir=args.logdir+'/1')
    player2 = Node.remote(32, 8, logdir=args.logdir+'/2')
    player3 = Node.remote(64, 64, logdir=args.logdir+'/3')
    player4 = Node.remote(32, 32, logdir=args.logdir+'/4')

    env = GymEnvironment(args.task)
    env.render()

    observation = env.reset(); reward = 0
    old_o1 = np.zeros([1, 32])
    old_o3 = np.zeros([1, 64])
    old_o4 = np.zeros([1, 32])
    for _ in range(args.max_iters):
        # NB should only have to get the action. the rest can be done elsewhere
        o1 = ray.get(player1.call.remote(observation, reward))
        o3 = ray.get(player3.call.remote(np.hstack([o1, old_o4]), reward))
        action = ray.get(player2.call.remote(o3[:, :32], reward))
        o4 = ray.get(player4.call.remote(o3[:, 32:], reward))

        old_o1 = o1
        old_o3 = o3
        old_o4 = o4

        observation, reward = env.step(action)

if __name__ == '__main__':
    main(argumentparser())
