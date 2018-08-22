import tensorflow as tf
import utils

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

        self.old_obs = None
        self.old_reward = None
        self.old_action = tf.zeros([1, 8])
        self.older_action = tf.zeros([1, 8])

        self.writer = tf.contrib.summary.create_file_writer(logdir)
        self.writer.set_as_default()

        self.temp = temp
        self.discount = discount

        self.batch_size = 1

    def get_losses(self, h, older_action, old_obs, old_reward, old_action, obs, reward, action):
        """
        Can be used with/wo eager.

        Args:
            obs_new (tf.tensor): input from t+1.
                shape [batch, n_inputs] dtype tf.float32
            obs_old (tf.tensor): input from t.
                shape [batch, n_inputs] dtype tf.float32
            reward (tf.tensor): reward recieved at t
                shape [batch, 1] dtype tf.float32
            action (tf.tensor): the action taken at t
                shape [batch, n_outputs] dtype tf.float32

        High level pattern.
        - Use inputs (obs_t, r_t) to build an internal state representation.
        - Use internal state at t to predict inputs at t+1 (obs_t+1, r_t+1).
        - Use the learned v(s, a), t(s, a) to evaluate actions chosen

        Returns:
            (tuple): transition_loss, value_loss, policy_loss
        """
        x_old = tf.concat([old_obs, old_reward, older_action], axis=1)
        h_old = self.encoder(x_old)

        # need differentiable actions.
        a_old = utils.choose_action(self.policy(h_old), self.temp)  # it bugs me that I need to recompute this
        # HACK old_action is not differentiable so need to recalc
        # but action is!
        v = self.value(tf.concat([h, action], axis=1))

        # OPTIMIZE implementation here. could write as simply predicting inputs!?
        # predict inputs at t+1 given action taken
        obs_approx = self.decoder(tf.concat([h_old, a_old], axis=1))
        h_approx = self.trans(tf.concat([h_old, a_old], axis=1))
        v_approx = self.value(tf.concat([h_old, a_old], axis=1))

        loss_d = tf.losses.mean_squared_error(obs, obs_approx)
        loss_t = tf.losses.mean_squared_error(tf.stop_gradient(h), h_approx)
        loss_v = tf.losses.mean_squared_error(v_approx, reward+self.discount*tf.stop_gradient(v))

        # maximise reward: use the appxoimated reward as supervision
        loss_p_exploit = -tf.reduce_mean(v)
        # explore: do things that result in unpredictable inputs
        loss_p_explore = - loss_d - loss_t - loss_v

        return loss_d, loss_t, loss_v, loss_p_exploit, loss_p_explore

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
                (loss_p_explore, self.policy.variables),  # the policy fn
                # (loss_p_exploit, self.policy.variables)  # the policy fn,
                ]

        grads = tape.gradient(*zip(*lnvs))
        losses = list(zip(*lnvs))[0]
        variables = list(zip(*lnvs))[1]

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
        self.build(n_obs=104, n_actions=13, n_hidden=128)

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
                                  input_shape=(n_obs + 8 + 1,),
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
        # TODO could supply action based on old info?
        # this would reduce latency!?

        h = self.encoder(tf.concat([obs, reward, self.old_action], axis=1))

        with tf.GradientTape() as tape:
            action = utils.choose_action(self.policy(h), self.temp)

            if self.old_obs is None:
                self.old_obs = obs
                self.old_reward = reward
                self.old_action = action
                self.older_action = self.old_action

            losses = self.get_losses(h, self.older_action, self.old_obs,
                self.old_reward, self.old_action, obs, reward, action)
        loss = self.train_step(tape, *losses)

        # loop the values around for next time. could fetch from the buffer instead...
        self.old_obs = obs
        self.old_reward = reward
        self.old_action = action
        self.older_action = self.old_action

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.histogram('a', action)
            tf.contrib.summary.histogram('state', h)
            tf.contrib.summary.histogram('obs', obs)

        return action

if __name__ == '__main__':
    tf.enable_eager_execution()
    player = OnlinePlayer(logdir='/tmp/test2/0')
    for i in range(50):
        observation = [1.0]*59
        action = player(observation, 1.0)
        print('A:{}'.format(action))
