'''
    File name: export.py
    Author: Deepender Singla
    Python Version: 2.7
    Purpose: Logic for exporting data the way it can be fed to NN.
'''

import csv
import logging
from datetime import datetime
import ipdb
import numpy as np
from dateutil.relativedelta import relativedelta
import collections
import datetime
import dateutil.parser
from functools import reduce


def export_time_series_data(files, usecols=6):
    '''
    input: times series csv of different assets and columns
    assumptions: all have same column format
    output log returns on then
    '''
    header = []
    data_dict = {}
    keys_dict = {}
    for file in files:
        file_data = np.genfromtxt(file, delimiter=",", skip_header=1, dtype="|S20", usecols=np.arange(usecols))
        keys = file_data[:, 0]
        keys_dict[file] = keys
        data_dict[file] = dict(zip(keys, file_data))
    whitelisted_keys = reduce(set.intersection, (set(val) for val in keys_dict.values()))
    keys_dict = {}  # free memory
    super_dict = collections.defaultdict(list)
    for file in files:
        header += [file + "_" + str(s) for s in np.arange(usecols)]
        values = data_dict[file]
        d = dict((k, values[k]) for k in whitelisted_keys if k in values)
        for k, v in d.items():
            v = v[2:]
            v = v.astype(np.float)
            super_dict[k] += v.tolist()
    new_file_name = "all_data.csv"
    whitelisted_keys = sorted(whitelisted_keys, key=lambda x: dateutil.parser.parse(x))
    with open(new_file_name, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        for row in whitelisted_keys:
            k = row
            v = super_dict[k]
            temp = [k] + v
            writer.writerow(temp)
    return super_dict


if __name__ == '__main__':
    files = ["NIFTY_sort.csv", "NIFTY_F1_sort.csv"]
    export_time_series_data(files, 6)
