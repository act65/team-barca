import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from learner import Learner

class Player(Learner):
    def __init__(self, *args, **kwargs):
        """
        A football player. Designed for HFO.
        """
        super(Player, self).__init__(*args, **kwargs)
        self.build(59, 13, 32)

    def build(self, n_obs, n_actions, n_hidden, width=32):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        self.policy = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                           tf.keras.layers.Dense(n_actions)], name='policy')

        self.value = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                          tf.keras.layers.Dense(1)], name='value')

        # TODO use RNN/DNC for enc. PROBLEM how is training going to work!?
        # will have to set a threshold on the depth?!
        # how will this work with the batching? it wont currently...
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(n_hidden)], name='encoder')
        self.trans = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(n_obs + 8 + 1)], name='trans')

    def __call__(self, obs, reward):
        """
        This functions job is to map the numpy/list/??? inputs into tf
        and back into suitable outputs for the env.

        Args:
            obs (list): the observations revealted at time t
            reward (float): the reward revealed at time t

        Returns:
            (list): actions to be taken
        """
        # TODO should write as a wrapper instead!?
        obs = preprocess(obs)
        reward = to_tf(reward)

        action = self.step(obs, reward)

        return tf.argmax(action[:, :3], axis=1).numpy().tolist() + action[0, 3:].numpy().tolist()

    def choose_action(self, state):
        """
        Args:
            state (tf.tensor): the hidden state at t
                shape [batch, n_hidden] dtype tf.float32

        Returns:
            gumbel (tfp.distribution): the distribution over discrete variables
            normal (tfp.distribution): the distribution over cts variables
        """
        # TODO make more general for some input spec
        # TODO add exploration
        p = self.policy(state)

        dis_vars = p[:,:3]
        cts_vars = p[:,3:]

        # discrete variables
        gumbel = tfp.distributions.RelaxedOneHotCategorical(self.temp, logits=dis_vars)

        # cts variables
        n = 5 # tf.shape()
        normal = tfp.distributions.MultivariateNormalDiag(loc=cts_vars[:,:n], scale_diag=cts_vars[:,n:]**2)
        # TODO if this was a mixture of gaussians then explore/exploit might make more sense?!

        return tf.concat([gumbel.sample(), normal.sample()], axis=1)

def preprocess(obs):
    return tf.cast(tf.reshape(tf.stack(obs, axis=0), [1, 59]), tf.float32)

def to_tf(x):
    return tf.reshape(tf.cast(tf.constant(x), tf.float32), [1, -1])

if __name__ == '__main__':
    tf.enable_eager_execution()
    player = Player(0,1, buffer_size=100, batch_size=10, logdir='/tmp/test2/0')
    for i in range(200):
        observation = [1.0]*59
        action = player(observation, 1.0)
        print('A:{}'.format(action))
