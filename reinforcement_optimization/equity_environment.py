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
import ipdb
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
    def __init__(self, assets, look_back, episode_length):
        #think about it whether its needed or not
        self.action_repeat = 2

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(len(assets)+1)

        self.look_back = look_back
        # Screen buffer of size action_repeat to be able to build
        # state arrays of size [1, action_repeat, 84, 84]
        self.data = pd.read_csv("data/all_data.csv")
        self.numpy_data = self.data.as_matrix()
        self.look_back = look_back
        self.assets_index = range(0, (len(self.gym_actions))*4, 4)[1:]
        self.look_ahead = 1
        self.batch_size = 50
        #self.portfolio = range(len(assets)+1)
        self.portfolio = 0.0
        self.episode_length = episode_length
        self.models = make_asset_input(assets, look_back, self.look_ahead, self.batch_size)
        self.state_buffer = deque()

    def get_initial_state(self, index):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()
        x_t = self.get_preprocessed_frame(index)
        #s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        #for i in range(self.action_repeat-1):
        #    self.state_buffer.append(x_t)
        return x_t

    def get_preprocessed_frame(self, index):
        """
         Take step of learning and return data
        """
        x = []
        assets_data = self.numpy_data[index:self.look_back+index][:,self.assets_index]
        for index, model in enumerate(self.models):
            value =  get_data_from_model(model, assets_data[:,index])
            x.append(value)
        x.append(self.portfolio)
        #x = [6161.4551648648358, 6168.7575063155873, 0.0]
        x = np.reshape(x, (1,len(x)))
        return x

    def calculate_reward(self, action, index):
        """
        Excecutes an action in the Equity environment.
        Update Portflio and return reward
        """

        #TODO need to be updated
        reward = 5
        info = 0
        return reward, info
    
    
    def random_sample_actions(self):
        return np.random.dirichlet(np.ones(len(self.gym_actions)),size=1)
    
    def step(self, action, index):
        """
        Excecutes an action in the Equity environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """

        if (index % self.episode_length == 0) and (index != 0):
            terminal = True
        else:
            terminal = False



        r_t, info  = self.calculate_reward(action,index)
        x_t1 = self.get_preprocessed_frame(index)
        s_t1 = x_t1

        #previous_frames = np.array(self.state_buffer)
        #TODO update 84 by 84
        #s_t1 = np.empty((self.action_repeat, 84, 84))
        #s_t1[:self.action_repeat-1, :] = previous_frames
        #s_t1[self.action_repeat-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        #self.state_buffer.popleft()
        #self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info