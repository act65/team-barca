import tensorflow as tf
import numpy as np
import copy

import practise.utils as utils
import practise.ams_grad as opt

class ActorCritic():
    def __init__(self, n_obs, n_actions, n_hidden=64, width=32, logdir='/tmp/test/0',
                 buffer_size=5000, batch_size=64, learning_rate=0.0001,
                 temp=10.0, discount=0.9):
        self.learning_rate = learning_rate
        self.opt = tf.train.AdamOptimizer(learning_rate)
        # self.opt = opt.AMSGrad(learning_rate)

        self.old_obs = np.zeros([1, n_obs])
        self.old_reward = np.zeros([1, 1])
        self.old_action = np.zeros([1, n_actions])
        self.older_action = np.zeros([1, n_actions])

        self.temp = temp
        self.discount = discount

        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.build(n_obs=n_obs, n_actions=n_actions, n_hidden=n_hidden, width=width)
        self.construct_graph()

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def build(self, n_obs, n_actions, n_hidden, width):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        self.policy = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                           tf.keras.layers.Dense((n_actions-3)*2+3)], name='policy')

        self.value = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                          tf.keras.layers.Dense(1)], name='value')

        # TODO use RNN/DNC for enc. PROBLEM how is training going to work!?
        # will have to set a threshold on the depth?!
        # how will this work with the batching? it wont currently...
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(width, activation=tf.nn.selu),
            tf.keras.layers.Dense(width, activation=tf.nn.selu),
            tf.keras.layers.Dense(width, activation=tf.nn.selu),
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

    def construct_graph(self):
        # older_action, old_obs, old_reward, old_action, obs, reward
        self.episode_feed = [
            tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32, name='older_a'),
            tf.placeholder(shape=[None, self.n_obs], dtype=tf.float32, name='old_obs'),
            tf.placeholder(shape=[None, 1], dtype=tf.float32, name='old_r'),
            tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32, name='old_a'),
            tf.placeholder(shape=[None, self.n_obs], dtype=tf.float32, name='obs'),
            tf.placeholder(shape=[None, 1], dtype=tf.float32, name='r'),
            tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32, name='a'),
            ]

        self.losses = self.get_losses(*self.episode_feed)
        loss_d, loss_t, loss_v, loss_p_exploit, loss_p_explore = self.losses
        self.loss = loss_d + loss_t + loss_v + loss_p_exploit + loss_p_explore
        self.global_step = tf.train.get_or_create_global_step()

        self.obs = tf.placeholder(shape=[None, self.n_obs], dtype=tf.float32)
        self.r = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.h = self.encoder(tf.concat([self.obs, self.r, self.old_action], axis=1))
        self.action = utils.choose_action(self.policy(self.h), self.temp)

        loss_p = loss_p_exploit + loss_p_explore
        lnvs = [(loss_d, self.encoder.variables + self.decoder.variables),  # the decoder fn
                (loss_t, self.encoder.variables + self.trans.variables),  # the transition fn
                (loss_v, self.encoder.variables + self.value.variables),  # the value fn
                (loss_p, self.policy.variables)  # the policy fn
                ]

        gnvs = [self.opt.compute_gradients(l, var_list=v) for l, v in lnvs]
        gnvs = [(tf.clip_by_norm(g, 1.0), v) for X in gnvs for g, v in X]

        self.R = tf.Variable(tf.zeros(shape=[], dtype=tf.float32), trainable=False)
        self.update_op = tf.assign(self.R, 0.9*self.R + tf.reduce_mean(self.episode_feed[-2][:, -10:]))

        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss_d', loss_d),
            tf.summary.scalar('loss_t', loss_t),
            tf.summary.scalar('loss_v', loss_v),
            tf.summary.scalar('loss_p_exploit', loss_p_exploit),
            tf.summary.scalar('loss_p_explore', loss_p_explore),
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('R', self.R)
            # tf.summary.histogram('action', self.action),
        ])

        self.train_step = self.opt.apply_gradients(gnvs, global_step=self.global_step)
        self.train_step = tf.group(*[self.train_step, self.update_op])

    def get_losses(self, older_action, old_obs, old_reward, old_action, obs, reward, action, method='meta_critic'):
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
        # TODO want enc to be recurrent and recieve;
        # the old action taken and the old reward recieved
        x_old = tf.concat([older_action, old_obs, old_reward], axis=1)
        h_old = self.encoder(x_old)
        x = tf.concat([old_action, obs, reward], axis=1)
        h = self.encoder(x)

        ### Predict inputs and changes
        # OPTIMIZE implementation here. could write as simply predicting inputs!?
        # predict inputs at t+1 given action taken
        obs_approx = self.decoder(tf.concat([h_old, old_action], axis=1))
        h_approx = self.trans(tf.concat([h_old, old_action], axis=1))

        if method == "meta_critic":
            ### Meta critic
            # maximise reward: use the approximated q/expected reward as supervision

            # need differentiable actions.
            # TODO is this what DICE is for!? need to do some reading
            # NOTE BUG FIXME PROBLEM. action is not differentiable so cannot get
            # grads to the true action chosen. instead of action use a
            # should work out in expectation, but will just be slower learning for now
            # solution is ??? online learning, partial evaluation, predict given the dist
            def mc_sample(x, n_samples=16):
                gumbel, normal = utils.get_dist(self.policy(x), self.temp)

                a = tf.reshape(tf.concat([gumbel.sample(n_samples),
                                              normal.sample(n_samples)], axis=-1),
                                   [tf.shape(x)[0]*n_samples, self.n_actions])

                stacked_x = tf.reshape(tf.concat([x for _ in range(n_samples)], axis=0), [tf.shape(x)[0]*n_samples, self.n_hidden])
                v = self.value(tf.concat([stacked_x, a], axis=1))
                return tf.reduce_mean(tf.reshape(v, [n_samples, tf.shape(x)[0], 1]), axis=0)

            v_old = mc_sample(h_old)
            v = mc_sample(h)

            loss_p_exploit = -tf.reduce_mean(v_old)

        elif method is "a2c":
            ### Advantage actor critic
            v_old = self.value(h_old)
            v = self.value(h)

            # BUG keep getting Nans...
            a_gumbel, a_normal = utils.get_dist(self.policy(h_old), self.temp)
            advantage = (reward+self.discount*tf.stop_gradient(v)) - v_old
            prob = a_gumbel.log_prob(old_action[:,:3]) + a_normal.log_prob(old_action[:,3:])
            prob = tf.where(tf.is_nan(prob), tf.zeros_like(prob), prob)
            loss_p_exploit = tf.reduce_mean(-prob * advantage)

        else:
            raise ValueError('Acceptable methods are a2c and meta_critic')

        loss_d = tf.losses.mean_squared_error(obs, obs_approx)
        loss_t = tf.losses.mean_squared_error(tf.stop_gradient(h), h_approx)
        loss_v = tf.losses.mean_squared_error(v_old, reward+self.discount*tf.stop_gradient(v))

        ### Explore: do things that result in unpredictable outcomes
        loss_p_explore = - loss_d - loss_t # - loss_v
        # NOTE no gradients propagate back throughthe policy to the enc.
        # good or bad? good. the losses are just the inverses so good?

        return loss_d, loss_t, loss_v, loss_p_exploit, loss_p_explore

    def run_train_step(self, episode):
        _, summ, i, L = self.sess.run([self.train_step, self.summaries, self.global_step, self.loss],
                                   feed_dict=dict(zip(self.episode_feed, episode)))
        self.writer.add_summary(summ, i)
        return L

    def run_step(self, obs, r):
        return self.sess.run(self.action, feed_dict={self.obs: obs, self.r: r})

    def offline_update(self):
        """
        Offline updates
        """
        # TODO rewards for history compression. how!?

        # TODO keep a moving avg of the loss normalise the learning rate by it
        # intution being to give more weight to high errors
        # just bc loss is large doesnt mean grad of loss is large

        # TODO want to learn a controller that meta learns to select/keep episodes?
        # brains (must) have the ability to segment histories into meaningful segments!?

        if len(self.buffer) > self.batch_size:
            # TODO selectively choose what goes into the buffer?
            def order(x):
                return np.random.standard_normal()

            inputs = list(sorted(self.buffer, key=order))
            batch = inputs[-(self.batch_size-10):] + self.buffer[-10:]  # take the most recent and some random others
            batch = [np.vstack(val) for val in zip(*batch)]
            loss = self.run_train_step(batch)
            self.buffer = self.buffer[1:]
            print('\rLoss: {}'.format(loss), end='', flush=True)
        else:
            loss = None

        return loss

    def call(self, obs, reward):
        obs = np.reshape(np.array(obs, dtype=np.float32), [1, self.n_obs])
        reward = np.reshape(np.array(reward, dtype=np.float32), [1, 1])

        action = self.run_step(obs, reward)
        self.offline_update()

        if self.old_obs is not None:
            self.buffer.append([
                copy.deepcopy(self.older_action),
                copy.deepcopy(self.old_obs),
                copy.deepcopy(self.old_reward),
                copy.deepcopy(self.old_action),
                copy.deepcopy(obs),
                copy.deepcopy(reward),
                copy.deepcopy(action)
                ])

        # loop the values around for next time. could fetch from the buffer instead...
        self.old_obs = obs
        self.old_reward = reward
        self.old_action = action
        self.older_action = self.old_action

        return action
