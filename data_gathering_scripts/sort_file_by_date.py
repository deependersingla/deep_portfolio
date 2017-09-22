import shutil
import glob
import csv
import dateutil.parser
import ipdb
from collections import OrderedDict
import os

for filename in glob.glob(os.path.dirname(os.getcwd()) + "\\data\\*.txt"):
    reader = csv.reader(open(filename, 'r'))
    data_dict = {}
    for row in reader:
        # print(row)
        date_time = dateutil.parser.parse(row[1] + "-" + row[2])
        # if len(row) < 8:
        #	row.append('0')
        data_dict[date_time] = row
    ordered = OrderedDict(sorted(data_dict.items(), key=lambda t: t[0]))
    new_file_name = filename.replace(".txt", "_sort.csv")
    with open(new_file_name, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["DateTime", "index", "OPEN", "HIGH", "LOW", "CLOSE"])
        for row in ordered.items():
            k = row[0]
            v = row[1]
            writer.writerow([k, v[0], v[3], v[4], v[5], v[6]])
