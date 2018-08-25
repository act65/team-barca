import unittest

import ray
import numpy as np
import tensorflow as tf

from practise.ams_grad import *


class Test(unittest.TestCase):
    def test(self):
        x = tf.random_normal([50, 12])
        net = tf.keras.layers.Dense(10)
        y = net(x)

        t = tf.zeros_like(y)

        loss = tf.losses.mean_squared_error(y, t)

        opt = AMSGrad()
        train_step = opt.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(train_step)

if __name__ == '__main__':
    unittest.main()
