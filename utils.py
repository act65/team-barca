import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

def observation_and_action_space(fn):
    """
    This functions job is to map the numpy/list/??? inputs into tf
    and back into suitable outputs for the env.

    Args:
        obs (list): the observations revealted at time t
        reward (float): the reward revealed at time t

    Returns:
        (list): actions to be taken
    """
    def wrapper(self, obs, reward, *args, **kwargs):

        obs = preprocess_obs(obs)
        reward = preprocess_r(reward)

        result = fn(self, obs, reward, *args, **kwargs)
        return tf.argmax(result[:, :3], axis=1).numpy().tolist() + result[0, 3:].numpy().tolist()
    return wrapper

def preprocess_obs(obs):
    """
    list -> tf.tensor [1, 59]
    """
    return tf.cast(tf.reshape(tf.stack(obs, axis=0), [1, 59]), tf.float32)

def preprocess_r(x):
    """
    float -> tf.tensor [1, 1]
    """
    return tf.reshape(tf.cast(tf.constant(x), tf.float32), [1, -1])

def choose_action(logits, temp):
    """
    Args:
        logits (tf.tensor): the hidden state at t
            shape [batch, n_hidden] dtype tf.float32

    Returns:
        gumbel (tfp.distribution): the distribution over discrete variables
        normal (tfp.distribution): the distribution over cts variables
    """
    # TODO add a way to combine exploration and exploitation!?

    dis_vars = logits[:,:3]
    cts_vars = logits[:,3:]

    # discrete variables
    gumbel = tfp.distributions.RelaxedOneHotCategorical(temp, logits=dis_vars)

    # cts variables
    n = 5 # tf.shape()
    normal = tfp.distributions.MultivariateNormalDiag(loc=cts_vars[:,:n], scale_diag=cts_vars[:,n:]**2)
    # TODO if this was a mixture of gaussians then explore/exploit might make more sense?!

    return tf.concat([gumbel.sample(), normal.sample()], axis=1)


if __name__ == '__main__':
    tf.enable_eager_execution()

    obs = [0.0]*59
    r = 1.0

    class Test():
        @observation_and_action_space
        def __call__(self, x ,r):
            return tf.concat([x, r], axis=1)

    fn = Test()

    y = fn(obs, r)
    assert len(y) == 60-2 # minus 2 because we took the argmax
