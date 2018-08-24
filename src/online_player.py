import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
import functools
import copy

import utils

_DEFAULT_POOL = ThreadPoolExecutor()

def threadpool(f, executor=None):
    # from bj0 on SE
    # https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python
    @functools.wraps(f)
    def wrap(*args, **kwargs):
        return (executor or _DEFAULT_POOL).submit(f, *args, **kwargs)
    return wrap

class RecurrentLearner(object):
    """
    Want a learner that updates when called.
    If has access to memory and can decide what to put into a buffer.
    """
    # problem is how to carry dependencies across calls to tf
    # want to learn long range temporal dependencies
    def __init__(self, n_hidden=32, logdir='/tmp/test/0',
                 learning_rate=0.0001, temp=100.0, discount=0.9):
        """
        Parent class for learners.
        Children need to implement;
        - `build`: must create `value, policy, encoder, trans`
        - `choose_action` must return differentiable actions.

        Args:
            pass
        """
        # TODO reward compression
        # TODO model based planning!?!
        self.learning_rate = learning_rate
        self.opt = tf.train.AdamOptimizer(learning_rate)

        self.old_obs = tf.zeros([1, 59])
        self.old_reward = tf.zeros([1, 1])
        self.action = tf.zeros([1, 8])
        self.old_action = tf.zeros([1, 8])
        self.older_action = tf.zeros([1, 8])

        self.writer = tf.contrib.summary.create_file_writer(logdir)
        self.writer.set_as_default()

        self.temp = temp
        self.discount = discount

        self.batch_size = 1

    # TODO want to write a wrapper that predict inputs and provides predicted inputs
    # related to synthetic grads!
    # TODO could supply action based on old info?
    # this would reduce latency!?
    # need a callback!? once action has been output do ...
    # the learner running on a separate thread!?
    # @threadpool
    def step(self, old_obs, old_reward, a_old, obs, reward):
        """
        Want to take an as soon as possible. No latency.
        So we are going to precompute a_t based on obs_t_1

        Args:
            ...
            a_old: the action taken tha resulted in obs,reward
        """
        with tf.GradientTape() as tape:
            x_old = tf.concat([old_obs, old_reward], axis=1)
            h_old = self.encoder(x_old)

            # choose action based on a prediction of the future state
            # where a_t = trans(encoder(x))
            h_approx = self.trans(h_old)
            action = utils.choose_action(self.policy(h_approx), self.temp)

            x = tf.concat([obs, old_reward], axis=1)
            h = self.encoder(x_old)
            v = self.value(tf.concat([h, action], axis=1))

            # OPTIMIZE implementation here. could write as simply predicting inputs!?
            # predict inputs at t+1 given action taken
            obs_approx = self.decoder(h_old)
            # BUG a_old, where is that coming from!?
            v_approx = self.value(tf.concat([h_old, a_old], axis=1))

            loss_d = tf.losses.mean_squared_error(obs, obs_approx)
            loss_t = tf.losses.mean_squared_error(tf.stop_gradient(h), h_approx)
            loss_v = tf.losses.mean_squared_error(v_approx, reward+self.discount*tf.stop_gradient(v))

            # maximise reward: use the appxoimated reward as supervision
            loss_p_exploit = -tf.reduce_mean(v)
            # explore: do things that result in unpredictable inputs
            loss_p_explore = - loss_d - loss_t - loss_v

        loss = self.train_step(tape, loss_d, loss_t, loss_v, loss_p_exploit, loss_p_explore)

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.histogram('a', action)
            tf.contrib.summary.histogram('state', h)
            tf.contrib.summary.histogram('obs', obs)

        return action

    def train_step(self, tape, loss_d, loss_t, loss_v, loss_p_exploit, loss_p_explore):
        """
        A training step for online learning.
        """
        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('loss_d', loss_d)
            tf.contrib.summary.scalar('loss_t', loss_t)
            tf.contrib.summary.scalar('loss_v', loss_v)
            tf.contrib.summary.scalar('loss_p_exploit', loss_p_exploit)
            tf.contrib.summary.scalar('loss_p_explore', loss_p_explore)

        # losses and variables
        lnvs = [(loss_d, self.encoder.variables + self.decoder.variables),  # the decoder fn
                (loss_t, self.encoder.variables + self.trans.variables),  # the transition fn
                (loss_v, self.encoder.variables + self.value.variables),  # the value fn
                # (loss_p_explore, self.policy.variables),  # the policy fn
                (loss_p_exploit, self.policy.variables)  # the policy fn,
                ]

        grads = tape.gradient(*zip(*lnvs))
        losses = list(zip(*lnvs))[0]
        variables = list(zip(*lnvs))[1]

        for g, v in zip(grads, variables):
            if g is None:
                raise ValueError('No gradient for {}'.format(v.name))

        gnvs = [(tf.clip_by_norm((L**2)*g, 10.0), v) for L, G, V in zip(losses, grads, variables) for g, v in zip(G, V)]

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            for g, v in gnvs:
                tf.contrib.summary.scalar('grads/{}'.format(v.name), tf.norm(g))

        self.opt.apply_gradients(gnvs, global_step=tf.train.get_or_create_global_step())

        return loss_t  + loss_v + loss_p_exploit + loss_p_explore


class OnlinePlayer(RecurrentLearner):
    def __init__(self, *args, **kwargs):
        """
        A football player. Designed for HFO.
        """
        super(self.__class__, self).__init__(*args, **kwargs)
        self.build(n_obs=59, n_actions=13, n_hidden=64)

    def build(self, n_obs, n_actions, n_hidden, width=32):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        self.policy = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                           tf.keras.layers.Dense(n_actions)], name='policy')

        self.value = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                          tf.keras.layers.Dense(1)], name='value')

        # QUESTION HOW CAN YOU LEARN long time scales?
        # TODO Real time recurrent learning? forward AD!?
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(width, activation=tf.nn.selu,
                                  input_shape=(n_obs + 1,),
                                  batch_size=self.batch_size),
            tf.keras.layers.Reshape([width, 1]),
            # TODO verify that `stateful` works as intended
            tf.keras.layers.LSTM(width, stateful=True, activation=tf.nn.selu),
            tf.keras.layers.Reshape([width]),
            tf.keras.layers.Dense(n_hidden)
        ], name='encoder')

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(width, activation=tf.nn.selu),
            tf.keras.layers.Dense(n_obs)],
        name='decoder')

        self.trans = tf.keras.Sequential([
            tf.keras.layers.Dense(width, activation=tf.nn.selu),
            tf.keras.layers.Dense(n_hidden)],
        name='trans')

    @utils.observation_and_action_space
    def __call__(self, obs, reward):
        """
        Step. Given a new observation and reward choose and action and add the
        latest episode to the buffer.

        Args:
            obs (tf.tensor):
            reward (tf.tensor):

        Returns:
            actions (tf.tensor)
        """
        self.older_action = self.old_action
        if not isinstance(self.action, tf.Tensor):
            self.old_action = self.action.result()
        else:
            self.old_action = self.action

        # DEBUG need to make sure the tensors are not mutated in place
        # not sure how to safely loop the actions with the async fn
        self.action = self.step(copy.deepcopy(self.old_obs),
                                copy.deepcopy(self.old_reward),
                                copy.deepcopy(self.older_action),
                                copy.deepcopy(obs),
                                copy.deepcopy(reward),
                                )

        self.old_obs = obs
        self.old_reward = reward

        return self.old_action


if __name__ == '__main__':
    tf.enable_eager_execution()
    player = OnlinePlayer(logdir='/tmp/test2/0')
    for i in range(50):
        observation = [1.0]*59
        action = player(observation, 1.0)
        print('A:{}'.format(action))
