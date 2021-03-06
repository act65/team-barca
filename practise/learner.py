import tensorflow as tf
import numpy as np
import utils

class ActorCritic(object):
    def __init__(self, input_specs, action_space, n_hidden=32, logdir='/tmp/test/0',
                 buffer_size=1000, batch_size=100, learning_rate=0.0001,
                 temp=100.0, discount=0.9):
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

        self.action_space = action_space

        self.learning_rate = learning_rate
        self.opt = tf.train.AdamOptimizer(learning_rate)

        self.old_obs = None
        self.old_reward = None
        self.old_action = tf.zeros([1, 8])

        self.writer = tf.contrib.summary.create_file_writer(logdir)
        self.writer.set_as_default()

        self.temp = temp
        self.discount = discount

        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def step(self, obs, reward):
        """
        Step. Given a new observation and reward choose and action and add the
        latest episode to the buffer.

        Args:
            obs (tf.tensor):
            reward (tf.tensor):

        Returns:
            actions (tf.tensor)
        """
        h = self.encoder(tf.concat([obs, reward, self.old_action], axis=1))

        # NOTE implemented by child class. action space will be specific to
        # the problem
        action = self.choose_action(self.policy(h), self.temp)

        if self.old_obs is None:
            self.old_obs = obs
            self.old_reward = reward
            self.old_action = action

        self.buffer.append([self.old_obs, self.old_reward, self.old_action, obs, action, reward])

        self.offline_update()

        # loop the values around for next time. could fetch from the buffer instead...
        self.old_obs = obs
        self.old_reward = reward
        self.old_action = action

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.histogram('a', action)
            tf.contrib.summary.histogram('state', h)
            tf.contrib.summary.histogram('obs', obs)

        return action

    def get_losses(self, old_obs, old_reward, old_action, obs, action, reward):
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
        # TODO would like to see a graph of this part. just for sanity

        # TODO want enc to be recurrent and recieve;
        # the old action taken and the old reward recieved
        x_old = tf.concat([old_obs, old_reward, old_action], axis=1)
        h_old = self.encoder(x_old)
        x = tf.concat([obs, reward, old_action], axis=1)
        h = self.encoder(x)

        # need differentiable actions.
        a = utils.choose_action(self.policy(h_old), self.temp)  # it bugs me that I need to recompute this
        a_new = utils.choose_action(self.policy(h), self.temp)
        # NOTE PROBLEM. old_action is not differentiable so cannot get
        # grads to the true action chosen. instead of old_action use a
        # should work out in expectation, but will just be slower learning for now
        # solution is ??? online learning, partial evaluation, predict given the dist

        v_old = self.value(tf.concat([h_old, a], axis=1))
        v = self.value(tf.concat([h, a_new], axis=1))

        # predict inputs at t+1 given action taken
        y = self.trans(tf.concat([h_old, a], axis=1))

        loss_t = tf.losses.mean_squared_error(x, y)
        loss_v = tf.losses.mean_squared_error(v_old, reward+self.discount*tf.stop_gradient(v))

        # maximise reward: use the appxoimated reward as supervision
        loss_p_exploit = -tf.reduce_mean(v)
        # explore: do things that result in unpredictable inputs
        loss_p_explore = - loss_t - loss_v
        # NOTE no gradients propagate back throughthe policy to the enc.
        # good or bad? good. the losses are just the inverses so good?
        # NOTE not sure it makes sense to train the same fn on both loss_p_explore
        # and loss_p_exploit???

        # # A3C: policy gradients with learned variance adjustment
        # A = 1+tf.stop_gradient(v_old - reward+self.discount*v)
        # p = tf.concat([tf.reshape(dis.prob(a_dis), [-1, 1]), cts.prob(a_cts)], axis=1)
        # loss_a = tf.reduce_mean(-tf.log(p)*A)

        return loss_t, loss_v, loss_p_exploit, loss_p_explore

    def train_step(self, episode):
        """
        Take a training step using tf.eager
        Args:
            *args: a single episode. actions, observations, rewards, ...

        Returns:
            loss
        """
        # TODO WANT partial evaluation so this can be done online!
        # problem with offline learning is that there is a delay between
        # being able to adapt to a change

        # TODO implement a version of this with tf.Session()!?
        with tf.GradientTape() as tape:
            loss_t, loss_v, loss_p_exploit, loss_p_explore = self.get_losses(*episode)

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('loss_t', loss_t)
            tf.contrib.summary.scalar('loss_v', loss_v)
            tf.contrib.summary.scalar('loss_p_exploit', loss_p_exploit)
            tf.contrib.summary.scalar('loss_p_explore', loss_p_explore)


        # losses and variables
        lnvs = [(loss_t, self.encoder.variables + self.trans.variables),  # the transition fn
                (loss_v, self.encoder.variables + self.value.variables),  # the value fn
                # (loss_p_explore, self.policy.variables),  # the policy fn
                (loss_p_exploit, self.policy.variables)  # the policy fn,
                ]

        grads = tape.gradient(*zip(*lnvs))
        losses = list(zip(*lnvs))[0]
        variables = list(zip(*lnvs))[1]

        # TODO want a way to dynamically balance the training of each fn.
        # PROBLEM! quite unstable! for now, use L to weight the grads
        # tf.clip_by_norm(L*g, 1.0)
        gnvs = [(tf.clip_by_norm(g, 1.0), v) for L, G, V in zip(losses, grads, variables) for g, v in zip(G, V)]

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            for g, v in gnvs:
                tf.contrib.summary.scalar('grads/{}'.format(v.name), tf.norm(g))

        self.opt.apply_gradients(gnvs, global_step=tf.train.get_or_create_global_step())

        return loss_t  + loss_v + loss_p_exploit + loss_p_explore

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

        if len(self.buffer) > self.buffer_size:
            # TODO selectively choose what goes into the buffer?
            inputs = list(sorted(self.buffer, key=lambda x: np.random.random()))
            batch = inputs[0:self.batch_size]
            batch = [tf.concat(val, axis=0) for val in zip(*batch)]
            loss = self.train_step(batch)
            self.buffer = self.buffer[1:]
            print('\rLoss: {}'.format(loss), end='', flush=True)
        else:
            loss = None

        return loss
