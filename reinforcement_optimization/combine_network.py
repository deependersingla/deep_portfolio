import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import tflearn as tfl
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
import ipdb
from tensor_helpers.tensormetrics_helper import tf_metrics
import csv
import pandas
from supervised_learning.lstm_single_stock import load_model_tflearn

def make_asset_input(assets, look_back, look_ahead, batch_size):
	asset = "NIFTY_sort"
	model = load_model_tflearn(look_back, batch_size, asset)
	#WIP


epochs = 50
split = (0.8, 0.1, 0.1)
use_csv = True
look_back = 500
look_ahead = 1
should_train_network = True
assets = ["data/NIFTY_sort.csv"]
batch_size = 50
if __name__ == "__main__":
    make_asset_input(assets, look_back, look_ahead, batch_size)
