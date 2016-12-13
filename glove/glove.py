'''
Defines glove model and performs one mini-batch SGD update in tensorflow.
'''

import tensorflow as tf
import numpy as np
import pdb

class glove(object):

    def __init__(self, batch_size, vocab_size, dim, lr=0.0001):
        self.X = tf.placeholder("float", shape=[batch_size,])
        self.fX = tf.placeholder("float", shape=[batch_size,])
        self.ind_W = tf.placeholder(tf.int32, shape=[batch_size])
        self.ind_W1 = tf.placeholder(tf.int32, shape=[batch_size])
        self.embedding = tf.Variable(tf.random_uniform([vocab_size, dim], -0.1, 0.1))
        self.embedding1 = tf.Variable(tf.random_uniform([vocab_size, dim], -0.1, 0.1))
        self.W = tf.nn.embedding_lookup(self.embedding, self.ind_W)
        self.W1 = tf.nn.embedding_lookup(self.embedding1, self.ind_W1)
        self.cost = tf.reduce_sum(self.fX * ((tf.reduce_sum(self.W * self.W1, reduction_indices=1) - self.X) ** 2))
        self.train_fn = tf.train.AdamOptimizer(lr).minimize(self.cost)
        self.session = tf.Session()
        tf.initialize_all_variables().run(session=self.session)

    def sgd(self, indw, indw1, X, fX):
        '''
        Performs one iteration of SGD.
        '''
        _, cost = self.session.run([self.train_fn, self.cost], feed_dict={self.ind_W: indw, self.ind_W1: indw1, self.X: X, self.fX: fX}) 
        return cost

    def save_params(self):
        '''
        Saves the word embedding lookup matrix to file.
        '''
        w = self.session.run(self.embedding)
        w1 = self.session.run(self.embedding1)
        W = w + w1
        np.save('lookup', W)
