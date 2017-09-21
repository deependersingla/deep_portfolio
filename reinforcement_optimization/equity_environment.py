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
from reinforcement_optimization.combine_network import make_asset_input, get_rescaled_value_from_model
from pandas_helpers.pandas_series_helper import pandas_split_series_into_list


# ====================
#   Equity Environment 
# ====================
class EquityEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size action_repeat from which environment state is constructed.
    """

    def __init__(self, assets, look_back, episode_length, look_back_reinforcement, price_series, train):
        # think about it whether its needed or not
        self.action_repeat = 2

        self.gym_actions = range(len(assets) + 1)

        self.look_back = look_back
        total_data = pd.read_csv("../data/all_data.csv")
        cut_index = int(total_data.shape[0] * 0.8)
        if train:
            data = total_data[0:cut_index]
        else:
            data = total_data[cut_index:-1]
        self.look_back = look_back
        self.assets_index = range(0, (len(self.gym_actions)) * 4, 4)[1:]
        self.look_ahead = 1
        self.batch_size = 50
        self.look_back_reinforcement = look_back_reinforcement
        self.total_data = pandas_split_series_into_list(data, self.look_back + episode_length + 1)
        # ipdb.set_trace();
        # self.numpy_data = self.data.as_matrix()
        self.price_series = price_series
        self.episode_length = episode_length
        self.models = make_asset_input(assets, look_back, self.look_ahead, self.batch_size)
        # self.models = [0,1]
        self.assets = assets

    def get_initial_state(self, index, episode):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()
        self.transaction_buffer = []
        self.initial_cash = 1000000
        self.cash = self.initial_cash
        self.data = self.total_data[episode]
        self.numpy_data = self.data.as_matrix()
        # zero in all assets + 100 percent in cash
        self.portfolio = [0] * (len(self.assets)) + [1]
        self.portfolio_quantity = [0] * (len(self.assets))
        x_t = self.get_preprocessed_frame(index)
        # s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        # for i in range(self.action_repeat-1):
        #    self.state_buffer.append(x_t)
        return x_t

    def get_preprocessed_frame(self, index):
        """
         Take step of learning and return data
        """
        x = []
        # print(index)
        assets_data = self.numpy_data[index:self.look_back + index][:, self.assets_index]
        for index, model in enumerate(self.models):
            value = get_rescaled_value_from_model(model, assets_data[:, index])
            # value = 2
            x.append(value)
            series_string = "ASSET_" + str(index + 1) + "_CLOSE"
            # temp = self.data[series_string].pct_change()[
            #        self.look_back + index - self.look_back_reinforcement:self.look_back + index]
            temp = self.data.iloc[:, self.assets_index[index]].pct_change()[
                   self.look_back + index - self.look_back_reinforcement:self.look_back + index]
            x += temp.tolist()
        x += self.portfolio
        x = np.reshape(x, (1, len(x)))
        return x

    def current_price_of_assets(self, index):
        current_prices = []
        assets_price = self.numpy_data[index:self.look_back + index][:, self.assets_index]
        for index, asset in enumerate(self.portfolio[0:-1]):
            current_price = assets_price[:, index][-1]
            current_prices.append(current_price)
        return current_prices

    def find_new_portfolio_quantity(self, new_portfolio, total_value, current_prices):
        new_portfolio_quantities = []
        total_price = 0
        for index, asset in enumerate(new_portfolio[0:-1]):
            new_value = asset * total_value
            quantitiy = new_value / current_prices[index]
            total_price += new_value
            new_portfolio_quantities.append(quantitiy)
        return new_portfolio_quantities, total_price

    def current_portfolio_holding_value(self, current_prices):
        current_holding_values = []
        for index, asset in enumerate(self.portfolio_quantity):
            current_holding_value = asset * current_prices[index]
            current_holding_values.append(current_holding_value)
        current_holding_values.append(self.cash)
        return current_holding_values

    def calculate_reward(self, action, index, terminal):
        """
        Excecutes an action in the Equity environment.
        Update Portflio and return reward
        """
        self.transaction_buffer.append(self.portfolio_quantity)
        current_portfolio = self.portfolio
        current_prices = self.current_price_of_assets(index)
        current_holding_values = self.current_portfolio_holding_value(current_prices)
        total_value = sum(current_holding_values)
        new_portfolio = action[0]
        new_portfolio_quantities, total_price = self.find_new_portfolio_quantity(new_portfolio, total_value,
                                                                                 current_prices)
        transaction_charges = 0  # assuming zero for now
        self.portfolio_quantity = new_portfolio_quantities
        self.portfolio = action[0].tolist()
        self.cash = total_value - total_price
        # currently terminal reward only
        if terminal:
            reward = total_value - self.initial_cash
            print(reward)
        else:
            reward = 0
        info = 0
        return reward, info

    def random_sample_actions(self):
        return np.random.dirichlet(np.ones(len(self.gym_actions)), size=1)

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

        r_t, info = self.calculate_reward(action, index, terminal)
        x_t1 = self.get_preprocessed_frame(index)
        s_t1 = x_t1

        # previous_frames = np.array(self.state_buffer)
        # TODO update 84 by 84
        # s_t1 = np.empty((self.action_repeat, 84, 84))
        # s_t1[:self.action_repeat-1, :] = previous_frames
        # s_t1[self.action_repeat-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        # self.state_buffer.popleft()
        # self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info
