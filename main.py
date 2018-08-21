import gym
import gym_soccer

import numpy as np
import player

import tensorflow as tf
tf.enable_eager_execution()

import argparse

def argumentparser():
    parser = argparse.ArgumentParser(description='Play soccer')
    parser.add_argument('--trials', type=int, default=50,
                        help='number of trials')
    parser.add_argument('--logdir', type=str, default='/tmp/test/',
                        help='location to save logs')
    parser.add_argument('--task', type=str, default='SoccerEmptyGoal-v0',
                        choices=['Soccer-v0', 'SoccerEmptyGoal-v0', 'SoccerAgainstKeeper-v0'])
    return parser.parse_args()

def main(args):
    agent = player.Player(0,1, logdir=args.logdir)
    env = gym.make('SoccerEmptyGoal-v0')

    R = [0] * args.trials
    for i in range(args.trials):
        print('New Game')
        observation = env.reset(); done = False; reward = 0
        counter = 0

        while not done:
            action = agent(observation, reward)
            # action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            counter += 1
            R[i] += reward
        print('\rR: {}'.format(R), end='', flush=True)
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            tf.contrib.summary.scalar('R', R[i])

if __name__ == '__main__':
    main(argumentparser())
