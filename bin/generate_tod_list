#!/usr/bin/env python

"""This script takes a pickle file and generate three lists of
TODs corresponding to train, validate and test for developing
machine learning pipeline.

Example:
./bin/generate_tod_list.py -t pa2_s14_c10_v4 -p ../../share/pa2/pa2_s14_c10_v4_results.pickle \
--n_train=80 --n_validate=20 --n_test=20 -o inputs/

Outputs:
Loading pickle file: ../../share/pa2/pa2_s14_c10_v4_results.pickle
Writing to inputs/pa2_s14_c10_v4_train.txt
Writing to inputs/pa2_s14_c10_v4_validate.txt
Writing to inputs/pa2_s14_c10_v4_test.txt
Done!

"""

import os
import pickle
import random
import argparse

################################
# parse command-line arguments #
################################

parser = argparse.ArgumentParser(description="Generate lists of TODs for machine learning")

parser.add_argument("-t", "--tag", help="Tag to use to generate output lists. Example: pa2_s14_c10_v4",
                    required=True)
parser.add_argument("-p", "--pickle", help="Path to the pickle file", required=True)
parser.add_argument("-o", "--output", help="Output directory to store tod lists",
                    default="inputs")
parser.add_argument("--n_train", help="Number of TODs for training. Default 60", type=int, default=60)
parser.add_argument("--n_validate", help="Number of TODs for validation. Default 20", type=int, default=20)
parser.add_argument("--n_test", help="Number of TODs for testing. Default 20", type=int, default=20)

args = parser.parse_args()

#########################
# define run parameters #
#########################

tag = args.tag

pickle_file = args.pickle
output_dir = args.output

n_train = args.n_train
n_validate = args.n_validate
n_test = args.n_test

#########
# main  #
#########

train_fname = "%s_train.txt" % tag
validate_fname = "%s_validate.txt" % tag
test_fname = "%s_test.txt" % tag

if not os.path.exists(output_dir):
    print("Folder %s doesn't exist, creating..." % output_dir)
    os.makedirs(output_dir)

# load pickle file
print("Loading pickle file: %s" % pickle_file)
with open(pickle_file, "r") as f:
    data = pickle.load(f)

# random select required number of tods
total_tods = n_train + n_validate + n_test
tod_list = random.sample(data['name'], total_tods)

# split into train, validate and test
train_list = tod_list[:n_train]
validate_list = tod_list[n_train:n_train+n_validate]
test_list = tod_list[n_train+n_validate:]

# write list to file
def write_to_file(outfile, lst):
    if len(lst) > 0:
        print("Writing to %s" % outfile)
        with open(outfile, "w") as f:
            for l in lst:
                f.write("%s\n" % l)
    else:
        print("list is empty, skip writing...")


# output filenames
outfile_train = os.path.join(output_dir, train_fname)
outfile_validate = os.path.join(output_dir, validate_fname)
outfile_test = os.path.join(output_dir, test_fname)

# write to files
write_to_file(outfile_train, train_list)
write_to_file(outfile_validate, validate_list)
write_to_file(outfile_test, test_list)

print("Done!")

