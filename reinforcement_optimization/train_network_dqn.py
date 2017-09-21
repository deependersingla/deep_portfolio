from __future__ import division, print_function, absolute_import

import threading
import random
import numpy as np
import time
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

import gym
import tflearn
from reinforcement_optimization.equity_environment import *
import numpy as np
import ipdb
import tensorflow as tf

assets = ["NIFTY_F1_sort", "NIFTY_sort"]
look_back_reinforcement = 10
# currently just taking closing price but in future we can use ohlc
price_series = 1
num_inputs = 2 * len(assets) + look_back_reinforcement * price_series * len(assets) + 1
num_actions = len(assets) + 1
look_back = 200

# Fix for TF 0.12
try:
    writer_summary = tf.summary.FileWriter
    merge_all_summaries = tf.summary.merge_all
    histogram_summary = tf.summary.histogram
    scalar_summary = tf.summary.scalar
except Exception:
    writer_summary = tf.train.SummaryWriter
    merge_all_summaries = tf.merge_all_summaries
    histogram_summary = tf.histogram_summary
    scalar_summary = tf.scalar_summary

tf.reset_default_graph()

# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, num_inputs], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([num_inputs, num_actions], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.nn.softmax(Qout)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, num_actions], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
episode_length = 300
# create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    env = EquityEnvironment(assets, look_back, episode_length, look_back_reinforcement, price_series, train=True)
    index = 0
    for episode in range(num_episodes):
        print("episode is: " + str(episode))
        # Reset environment and get first new observation
        s = env.get_initial_state(0, episode)
        rAll = 0
        d = False
        j = 0
        # The Q-Network
        while j < episode_length * 2:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            index += 1
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: s})
            if np.random.rand(1) < e:
                a = env.random_sample_actions()
            # Get new state and reward from environment
            s1, r, d, _ = env.step(a, j)
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs1: s})
            # Obtain maxQ' and set our target value for chosen action.
            # TODO this need to be updated
            maxQ1 = Q1
            targetQ = allQ
            targetQ = r + y * maxQ1
            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: s, nextQ: targetQ})
            rAll += r
            s = s1
            if d == True:
                # Reduce chance of random action as we train the model.
                e = 1. / ((episode / 50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
