import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Player(object):
    # TODO disentangle soccer and learner
    # want class Player(A3C): ...
    def __init__(self, obs_shape, action_space, logdir,
                 buffer_size=500, batch_size=50, learning_rate=0.0001,
                 temp=1.0, stddev=0.1):
        self.action_space = action_space

        self.learning_rate = learning_rate
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.old_obs = tf.zeros(shape=[1, 59], dtype=tf.float32)

        self.writer = tf.contrib.summary.create_file_writer(logdir)
        self.writer.set_as_default()

        self.temp = temp
        self.stddev = stddev

        self.gamma = 0.9

        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.build()

    def build(self):
        self.policy = tf.keras.Sequential([tf.keras.layers.Dense(16, activation=tf.nn.selu),
                                           tf.keras.layers.Dense(3+5*2)])

        self.value = tf.keras.Sequential([tf.keras.layers.Dense(16, activation=tf.nn.selu),
                                          tf.keras.layers.Dense(1)])

        # TODO use RNN/DNC for enc. but how is training going to work!?
        # will have to set a threshold on the depth?!
        self.enc = tf.keras.Sequential([tf.keras.layers.Dense(16, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(16, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(16)])
        self.dec = tf.keras.Sequential([tf.keras.layers.Dense(16, activation=tf.nn.selu),
                                        tf.keras.layers.Dense(59)])

    def __call__(self, obs, r):
        obs = preprocess(obs)
        r = to_tf(r)
        h = self.enc(obs)
        dis, cts = self.choose_action(h)
        a = tf.concat([dis.sample(), cts.sample()], axis=1)

        self.buffer.append([obs, self.old_obs, r, a])
        self.update()
        self.old_obs = obs

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.histogram('a', a)
            tf.contrib.summary.histogram('state', h)
            tf.contrib.summary.histogram('obs', obs)

        return tf.argmax(a[:, :3], axis=1).numpy().tolist() + a[0, 3:].numpy().tolist()

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
        return gumbel, normal

    def step(self, obs_new, obs_old, reward, a, policy_loss='approx_loss'):
        # TODO WANT partial evaluation so this can be done online!
        # problem with offline learning is that there is a delay between
        # being able to adapt to a change
        """
        Args:
            obs_new (tf.tensor): input from t+1.
                shape [batch, n_inputs] dtype tf.float32
            obs_old (tf.tensor): input from t.
                shape [batch, n_inputs] dtype tf.float32
            reward (tf.tensor): reward recieved at t
                shape [batch, 1] dtype tf.float32
            action (tf.tensor): the action taken at t
                shape [batch, n_outputs] dtype tf.float32

        High level patterns.
        - Use internal state at t to predict inputs at t+1.
        - Use the learned v(s, a) to evaluate actions chosen
        """
        with tf.GradientTape() as tape:
            # TODO want enc to be recurrent and recieve the action taken
            h_old = self.enc(obs_old)
            h_new = self.enc(obs_new)

            dis, cts = self.choose_action(h_new)
            a_dis, a_cts = dis.sample(), cts.sample()
            a_new = tf.concat([a_dis, a_cts], axis=1)

            v_old = self.value(tf.concat([h_old, a], axis=1))
            v_new = self.value(tf.concat([h_new, a_new], axis=1))

            y = self.dec(tf.concat([h_old, a], axis=1))

            if policy_loss == 'approx_loss':
                loss_p = tf.reduce_mean(-v_new)
            else:
                # policy gradients with learned variance adjustment
                A = tf.stop_gradient(v_old - reward+self.gamma*v_new)
                p = tf.concat([tf.reshape(dis.prob(a_dis), [-1, 1]), cts.prob(a_cts)], axis=1)
                loss_p = tf.reduce_mean(-tf.log(p)*A)

            loss_t = tf.losses.mean_squared_error(obs_new, y)
            loss_v = tf.losses.mean_squared_error(v_old, reward+self.gamma*tf.stop_gradient(v_new))

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('loss_t', loss_t)
            tf.contrib.summary.scalar('loss_p', loss_p)
            tf.contrib.summary.scalar('loss_v', loss_v)

        losses = [loss_t, loss_p, loss_v]
        variables = [self.enc.variables + self.dec.variables,
                     self.enc.variables + self.policy.variables,
                     self.enc.variables + self.value.variables]

        grads = tape.gradient(losses, variables)
        gnvs = [(tf.clip_by_norm(g, 1.0), v) for G, V in zip(grads, variables) for g, v in zip(G, V)]
        self.opt.apply_gradients(gnvs, global_step=tf.train.get_or_create_global_step())

        return loss_t + loss_p + loss_v

    def update(self):
        # TODO reward curiosity
        # TODO model based learning
        # TODO keep a moving avg of the loss normalise the learning rate by it
        # intution being to give more weight to high errors
        # just bc loss is large doesnt mean grad of loss is large

        if len(self.buffer) > self.buffer_size:
            # TODO selectively choose what goes into the buffer?
            inputs = list(sorted(self.buffer, key=lambda x: np.random.random()))[0:self.batch_size]
            inputs = [tf.concat(val, axis=0) for val in zip(*inputs)]
            loss = self.step(*inputs)
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
    player = Player(0,1, '/tmp/test', buffer_size=100)
    for i in range(200):
        observation = [1.0]*59
        action = player(observation, 1.0)
        print('A:{}'.format(action))
