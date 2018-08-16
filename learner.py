import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Player(object):
    # TODO disentangle soccer and learner
    # want class Player(A3C): ...
    def __init__(self, obs_shape, action_space, n_hidden=32, logdir='/tmp/test/0',
                 buffer_size=1000, batch_size=100, learning_rate=0.0001,
                 temp=100.0, discount=0.9):
        """
        Args:
            pass
        """
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

        self.build(59, 13, n_hidden)

    def build(self, n_obs, n_actions, n_hidden, width=32):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        self.policy = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                           tf.keras.layers.Dense(n_actions)])

        self.value = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                          tf.keras.layers.Dense(1)])

        # TODO use RNN/DNC for enc. PROBLEM how is training going to work!?
        # will have to set a threshold on the depth?!
        # how will this work with the batching? it wont currently...
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(n_hidden)])
        self.trans = tf.keras.Sequential([tf.keras.layers.Dense(width, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(n_obs + 8 + 1)])

    def __call__(self, obs, reward):
        """
        Chooses an action an adds relevant data to a buffer.
        Takes a single step using data from the buffer.

        Args:
            obs (list): the observations revealted at time t
            reward (float): the reward revealed at time t

        Returns:
            (list): actions to be taken
        """
        obs = preprocess(obs)
        reward = to_tf(reward)
        h = self.encoder(tf.concat([obs, reward, self.old_action], axis=1))
        action = self.choose_action(h)

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
        normal = tfp.distributions.Normal(loc=cts_vars[:,:n], scale=cts_vars[:,n:]**2)
        return tf.concat([gumbel.sample(), normal.sample()], axis=1)

    def train_step(self, *args):
        # TODO WANT partial evaluation so this can be done online!
        # problem with offline learning is that there is a delay between
        # being able to adapt to a change
        with tf.GradientTape() as tape:
            loss_t, loss_p, loss_v = self.get_losses(*args)

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('loss_t', loss_t)
            tf.contrib.summary.scalar('loss_p', loss_p)
            tf.contrib.summary.scalar('loss_v', loss_v)

        # losses and variables
        lnvs = [(loss_t, self.encoder.variables + self.trans.variables),  # the transition fn
                (loss_v, self.encoder.variables + self.value.variables),  # the value fn
                (loss_p, self.policy.variables)  # the policy fn
                ]

        grads = tape.gradient(*zip(*lnvs))
        losses = list(zip(*lnvs))[0]
        variables = list(zip(*lnvs))[1]

        # TODO want a way to dynamically balance the training of each fn.
        # PROBLEM! quite unstable! for now, use L to weight the grads
        gnvs = [(tf.clip_by_norm(L*g, 1.0), v) for L, G, V in zip(losses, grads, variables) for g, v in zip(G, V)]
        self.opt.apply_gradients(gnvs, global_step=tf.train.get_or_create_global_step())

        return loss_t + loss_p + loss_v

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
        - Use inputs (obs_t, r_t) to build an internal state representation
        - Use internal state at t to predict inputs at t+1 (obs_t+1, r_t+1).
        - Use the learned v(s, a), t(s, a) to evaluate actions chosen

        Returns:
            (tuple): transition_loss, value_loss, policy_loss
        """
        # TODO want enc to be recurrent and recieve;
        # the old action taken and the old reward recieved
        x_old = tf.concat([old_obs, old_reward, old_action], axis=1)
        h_old = self.encoder(x_old)
        x = tf.concat([obs, reward, old_action], axis=1)
        h = self.encoder(x)

        # need differentiable actions.
        a = self.choose_action(h_old)  # it bugs me that I need to recompute this
        a_new = self.choose_action(h)
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
        loss_p = -tf.reduce_mean(v)
        # explore: do things that result in unpredictable inputs
        loss_p -= loss_t + loss_v
        # NOTE no gradients propagate pack through the policy to the enc.
        # good or bad? good. the losses are just the inverses

        # # A3C: policy gradients with learned variance adjustment
        # A = 1+tf.stop_gradient(v_old - reward+self.discount*v)
        # p = tf.concat([tf.reshape(dis.prob(a_dis), [-1, 1]), cts.prob(a_cts)], axis=1)
        # loss_a = tf.reduce_mean(-tf.log(p)*A)

        return loss_t, loss_p, loss_v

    def offline_update(self):
        """
        Offline updates
        """
        # TODO reward compression
        # TODO model based planning!?!

        # TODO keep a moving avg of the loss normalise the learning rate by it
        # intution being to give more weight to high errors
        # just bc loss is large doesnt mean grad of loss is large

        if len(self.buffer) > self.buffer_size:
            # TODO selectively choose what goes into the buffer?
            inputs = list(sorted(self.buffer, key=lambda x: np.random.random()))[0:self.batch_size]
            inputs = [tf.concat(val, axis=0) for val in zip(*inputs)]
            loss = self.train_step(*inputs)
            self.buffer = self.buffer[1:]
            print('\rLoss: {}'.format(loss), end='', flush=True)
        else:
            loss = None

        return loss

def preprocess(obs):
    return tf.cast(tf.reshape(tf.stack(obs, axis=0), [1, 59]), tf.float32)

def to_tf(x):
    return tf.reshape(tf.cast(tf.constant(x), tf.float32), [1, -1])

if __name__ == '__main__':
    tf.enable_eager_execution()
    player = Player(0,1, buffer_size=100, batch_size=10)
    for i in range(200):
        observation = [1.0]*59
        action = player(observation, 1.0)
        print('A:{}'.format(action))
