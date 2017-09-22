import os
import glob

first_directory = os.path.dirname(os.getcwd()) + "\\data\\oneminutedata"
directory_name = first_directory
concatfilename = ["NIFTY", "NIFTY_F1"]
for confn in concatfilename:
    new_files = []
    for path, dirs, files in os.walk(directory_name):
        # path = root.split('/')
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            if str(confn + ".txt") in file:
                new_files.append(os.path.join(path, file))
                print(os.path.join(path, file))
    read_files = new_files
    with open(os.path.dirname(os.getcwd()) + "\\data\\" + confn + ".txt", "w") as outfile:
        for f in read_files:
            with open(f, "r") as infile:
                outfile.write(infile.read())
