#########################################################################
#
# Script name: test_read.py
# Purpose: Reads .json file to make sure that the format is not corrupted
#
# Pueh Leng Tan, 23 July 2020

import json
import glob

path_bag = glob.glob('dummy*.json') 

for loc_path in path_bag:
    print('\nReading %s ...' % loc_path)
    with open(loc_path, 'r') as fid:
        tmp = json.load(fid)
        print(tmp)
