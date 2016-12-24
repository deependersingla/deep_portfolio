import os
import glob


first_directory  = "/Users/deep/development/deep_portfolio/data/oneminutedata"
directory_name = first_directory
new_files = []
for path, dirs, files in os.walk(directory_name):
    #path = root.split('/')
    #print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
    	if "INDIAVIX.txt" in file:
    		new_files.append(os.path.join(path,file))
    		print os.path.join(path,file)
read_files = new_files
with open("/Users/deep/development/deep_portfolio/data/INDIAVIX.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())