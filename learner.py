import tensorflow as tf
import tensorflow_probability as tfp

class Player(object):
    def __init__(self, obs_shape, action_space, logdir):
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

        self.opt = tf.train.AdamOptimizer()
        self.old_obs = tf.zeros(shape=[1, 59], dtype=tf.float32)


        self.writer = tf.contrib.summary.create_file_writer(logdir)
        self.writer.set_as_default()

        self.normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        self.temp = 1e-4

        self.buffer = []

    def __call__(self, obs, r):
        obs = preprocess(obs)
        r = to_tf(r)
        h = self.enc(obs)
        a = self.choose_action(h)

        self.buffer.append([obs, self.old_obs, r, a])
        self.old_obs = obs

        return tf.argmax(a[:, :3], axis=1).numpy().tolist() + a[0, 3:].numpy().tolist()

    def choose_action(self, state):
        p = self.policy(state)

        # TODO make more general for some input spec
        dis_vars = p[:,0:3]
        cts_vars = p[:,3:13]

        # discrete variables
        dist = tfp.distributions.RelaxedOneHotCategorical(self.temp, logits=dis_vars)
        a_dis = dist.sample()

        # cts variables
        n = 5 # tf.shape()
        e = tf.random_normal([1, n]) #self.normal.sample([1, n])
        a_cts = cts_vars[0,:n] + e*(cts_vars[0,n:]**2)

        return tf.concat([a_dis, a_cts], axis=1)

    def step(self, obs_new, obs_old, reward, a):
        # TODO WANT partial evaluation so this can be done online!
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
        """
        with tf.GradientTape() as tape:
            h_old = self.enc(obs_old)

            h_new = self.enc(obs_new)
            a_new = self.choose_action(h_new)
            v_old = self.value(tf.concat([h_old, a], axis=1))
            v_new = self.value(tf.concat([h_new, a_new], axis=1))

            y = self.dec(h_old)

            loss_t = tf.losses.mean_squared_error(obs_new, y)
            loss_p = -v_new
            loss_v = tf.losses.mean_squared_error(v_old, reward+tf.stop_gradient(v_new))

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('loss_t', loss_t)
            tf.contrib.summary.scalar('loss_p', loss_p)
            tf.contrib.summary.scalar('loss_v', loss_v)
            tf.contrib.summary.histogram('a', a_new)
            tf.contrib.summary.histogram('state', h_old)


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
        # TODO want to take this offline on a separate process!??

        # TODO enc gets applied multiple times. want to partially evaluate!?!
        # or merge into a signle fn...


        # TODO keep a moving avg of the loss normalise the learning rate by it
        # intution being to give more weight to high errors
        # just bc loss is large doesnt mean grad of loss is large

        if len(self.buffer) > 0:
            # TODO add batching to the buffer
            # TODO selectively choose what goes into the buffer?
            loss = self.step(*self.buffer.pop())
            print('\rLoss: {}'.format(loss), end='', flush=True)
        else:
            pass

        return loss

def preprocess(obs):
    return tf.cast(tf.reshape(tf.stack(obs, axis=0), [1, 59]), tf.float32)

def to_tf(x):
    return tf.reshape(tf.cast(tf.constant(x), tf.float32), [1, -1])

if __name__ == '__main__':
    tf.enable_eager_execution()
    player = Player(0,1, '/tmp/test')
    observation = [0]*59
    action = player(observation, 1.0)
    print(action)
    loss = player.update()
