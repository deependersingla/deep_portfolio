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
	models = []
	for asset in assets:
		model = load_model_tflearn(look_back, batch_size, asset)
		models.append(model)
		tf.reset_default_graph()
	return models


epochs = 50
split = (0.8, 0.1, 0.1)
use_csv = True
look_back = 200
look_ahead = 1
should_train_network = True
assets = ["NIFTY_F1_sort", "NIFTY_sort"]
batch_size = 50
if __name__ == "__main__":
    models = make_asset_input(assets, look_back, look_ahead, batch_size)
    print("I am here")