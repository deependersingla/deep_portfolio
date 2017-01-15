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


def pandas_split_series_into_list(data, list_size):
	final_data = []
	shape = data.shape[0]
	for index in range(0,(shape-list_size-5)):
		temp_data = data[index:index+list_size]
		final_data.append(temp_data)
	return final_data