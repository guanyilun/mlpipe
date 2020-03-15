#!/bin/env python

"""This script prints the number of data entries
for each group in a given dataset file. It's meant
to be used for a quick inspection

Example:

./inspect_dataset.py -f ../outputs/*.h5
Content of ../outputs/dataset.h5:
-> train:	 2411 entries
Content of ../outputs/dataset_2f.h5:
-> test:	 2526 entries
Content of ../outputs/merge.h5:
-> test:	 32839 entries
-> train:	 131439 entries
-> validate:	 49271 entries
Content of ../outputs/mr3_pa2_s16.h5:
-> test:	 32839 entries
-> validate:	 16417 entries
Content of ../outputs/pa2_s14_c10_v4.h5:
-> train:	 65731 entries
-> validate:	 16426 entries
Content of ../outputs/pa2_s15_c10_v4.h5:
-> train:	 65708 entries
-> validate:	 16428 entries
Content of ../outputs/pa3_f90_s16_n400.h5:
-> test:	 32156 entries
-> train:	 96470 entries
-> validate:	 32158 entries
"""

import os, sys
import h5py

########################
# command-line options #
########################

import argparse

parser = argparse.ArgumentParser(description="Inspect h5 dataset")
parser.add_argument("-f", "--files", help="files to inspect", nargs="+", required=True)
args = parser.parse_args()
files = args.files

# check if the files exist
for f in files:
    if not os.path.isfile(f):
        sys.exit("Cannot find %s, exiting..." % f)

################
# main program #
################

opened_files = []
for f in files:
    h5file = h5py.File(f, "r")
    print("Content of %s:" % f)
    for group in list(h5file.keys()):
        print("-> %s:\t %d entries" % (group, len(list(h5file[group].keys()))))

    opened_files.append(h5file)


# close all opened files
for f in opened_files:
    f.close()
