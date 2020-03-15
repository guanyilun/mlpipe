#!/bin/env python

"""This script dumps the metadata from a given h5 file
and save them into pickle files for each group (train / test)
respectively

Example: 
./dump_metadata.py -h
usage: dump_metadata.py [-h] -i INPUT -o OUTPUT

Dump the metadata of an h5 file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input h5 file
  -o OUTPUT, --output OUTPUT
                        Output directory


Example:
./dump_metadata.py -i data/dataset.h5 -o tests

Found fields: [u'DELive', u'MFELive', u'corrLive', u'darkRatioLive',
u'feat1', u'feat2', u'feat3', u'feat5', u'gainLive', u'jumpDark',
u'jumpLive', u'kurtLive', u'kurtpLive', u'label', u'normLive',
u'rmsDark', u'rmsLive', u'skewLive', u'skewpLive']
Found groups: [u'train', u'validate']
Parsing group: train...
-> Total number of entries: 24119
Processed 1000/24119
Processed 2000/24119
...

"""

import argparse
import os, sys
import h5py
import numpy as np

try:
    import pickle
except ImportError:
    import cPickle as pickle

###################
# parse arguments #
###################

parser = argparse.ArgumentParser(description="Dump the metadata of an h5 file")

parser.add_argument("-i", "--input", help="Input h5 file", required=True)
parser.add_argument("-o", "--output", help="Output directory", required=True)
# parser.add_argument("-m", "--merge", help="Merge different groups as one group", action="store_true")

args = parser.parse_args()

input_file = args.input
output_dir = args.output
# to_merge = args.merge

##################
# util functions #
##################

def get_fields(data):
    """Get the metadata fields from a given h5 file by scalping"""
    return data['train'][data['train'].keys()[0]].attrs.keys()

########
# main #
########

# check if output directory exists, otherwise create it
if not os.path.exists(output_dir):
    print("Directory %s doesn't exists, creating now..." % output_dir)
    os.makedirs(output_dir)

# check if h5 file exists
if not os.path.isfile(input_file):
    sys.exit("Cannot find input file!")

# now we assume everything goes well
# proceed to load h5 file
data = h5py.File(input_file, "r")

# parse the fields for metadata
fields = get_fields(data)
print("Found fields: %s" % fields)
n_fields = len(fields)

# get groups
groups = data.keys()
print("Found groups: %s" % groups)


for group_name in groups:
    print("Parsing group: %s..." % group_name)
    group = data[group_name]
    data_keys = group.keys()
    n_keys = len(data_keys)
    print("-> Total number of entries: %d" % n_keys)

    # initializing empty dictionary to store data
    output = {}

    for field in fields:
        output[field] = np.zeros(n_keys)

    for i in range(n_keys):
        if i % 1000 == 0 and i > 0:
            print("Processed %d/%d" % (i, n_keys))

        # get data entry
        d = group[data_keys[i]]

        # retrieve metadata
        for field in fields:
            output[field][i] = d.attrs[field]

    # dump the output
    filename = os.path.join(output_dir, "%s.pickle" % group_name)
    print("Saving to: %s..." % filename)
    with open(filename, "wb") as f:
        pickle.dump(output, f, 2)  # highest protocol

print("Done!")
