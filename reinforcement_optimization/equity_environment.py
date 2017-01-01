from __future__ import division, print_function, absolute_import

import threading
import random
import numpy as np
import time
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

import gym
import tensorflow as tf
import tflearn
import pandas as pd
from combine_network import make_asset_input, get_data_from_model

# ====================
#   Equity Environment 
# ====================
class EquityEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size action_repeat from which environment state is constructed.
    """
    def __init__(self, assets, look_back):
        #think about it whether its needed or not
        self.action_repeat = 2

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(len(assets)+1)

        self.look_back = look_back
        # Screen buffer of size action_repeat to be able to build
        # state arrays of size [1, action_repeat, 84, 84]
        self.data = pd.read_csv("data/all_data.csv")
        self.numpy_data = data.as_matrix()
        self.look_back = look_back
        self.look_ahead = 1
        self.batch_size = 50
        self.portfolio = range(len(assets)+1)
        #episode length will be 300
        self.episode_length = 300
        self.models = make_asset_input(assets, look_back, look_ahead, batch_size)
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()
        x_t = self.get_preprocessed_frame(0)
        s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        for i in range(self.action_repeat-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, index):
        """
         Take step of learning and return data
        """
        assets_index = range(0, (len(gym_actions))*4, 4)[1:]
        assets_data = self.numpy_data[index:self.look_back+index][:,assets_index]
        x = []
        for index, model in enumerate(self.models):
            value =  get_data_from_model(model, assets_data[:,index])
            x.append(value)
        x.append(self.portfolio)
        return x

    def calculate_reward(self, action, index):
        """
        Excecutes an action in the Equity environment.
        Update Portflio and return reward
        """
        return reward, info

    def step(self, action, index):
        """
        Excecutes an action in the Equity environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """

        if index == self.episode_length:
            terminal = True
        else:
            terminal = False



        r_t  = self.calculate_reward(action,index)
        x_t1 = self.get_preprocessed_frame(index)

        previous_frames = np.array(self.state_buffer)
        #TODO update 84 by 84
        s_t1 = np.empty((self.action_repeat, 84, 84))
        s_t1[:self.action_repeat-1, :] = previous_frames
        s_t1[self.action_repeat-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info