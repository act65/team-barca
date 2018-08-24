import unittest

import ray
import numpy as np
import tensorflow as tf

from practise.multi_player import *

def generate_fake_data(n):
    obs = np.zeros([1, 59])
    action = np.zeros([1, 8])
    reward = np.zeros([1, 1])
    return [[action, obs, reward, action, obs, reward] for _ in range(n)]

class Test(unittest.TestCase):
    def test_integration(self):
        node = Node(59, 8, buffer_size=100, batch_size=20, logdir='/tmp/test2/13')
        for i in range(100):
            observation = [1.0]*59
            action = node.call(observation, 1.0)
            # print('A:{}'.format(action))

        self.assertEqual(action.shape, (1, 8))

    def test_train(self):
        """
        check that when the train_step is run it actually makes an update.
        """
        node = Node(59, 8, buffer_size=100, batch_size=20, logdir='/tmp/test2/13')

        node.buffer = generate_fake_data(50)

        start = node.sess.run(node.global_step)
        node.offline_update()
        end = node.sess.run(node.global_step)

        self.assertTrue(start < end)

class TestBuffer(unittest.TestCase):
    def test_buffer(self):
        pass

class TestRay(unittest.TestCase):
    def test_wrap(self):
        ray.init()
        node = ray.remote(Node)
        node = node.remote(59, 8, buffer_size=100, batch_size=20, logdir='/tmp/test2/13')

        for i in range(100):
            observation = [1.0]*59
            action = node.call.remote(observation, 1.0)

        self.assertEqual(ray.get(action).shape, (1, 8))


if __name__ == '__main__':
    unittest.main()
