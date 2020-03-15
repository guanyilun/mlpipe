#!/bin/env python

"""This script provides a command-line utility function
to merge different h5 files.
"""

import argparse
import os, sys
import h5py


##############
# parse args #
##############

parser = argparse.ArgumentParser(description="Merge different h5 files")
parser.add_argument("-f", "--files", help="List of pickle files to look at", required=True, nargs="+")
parser.add_argument("-o", "--output", help="File to save the merged h5 files", required=True)
args = parser.parse_args()

files = args.files
outfile = args.output

# check if all the files exists
for f in files:
    if not os.path.isfile(f):
        sys.exit("File %s is not found! Exiting...")

# check if the destination file can be created successfully
try:
    dest_file = h5py.File(outfile, "w")
except:
    sys.exit("Error creating destination file! Exiting...")

# now I trust that I can proceed
files_lookup = {}

print("Scanning files...")
for f in files:
    # load h5 file
    h5file = h5py.File(f, "r")

    # load groups
    h5groups = list(h5file.keys())

    # save info to look-up tables
    files_lookup[f] = {
        'container': h5file,
    }
    for g in h5groups:
        files_lookup[f][g] = h5file[g]


################
# main program #
################

import inquirer

questions = [
    inquirer.Checkbox('train',
                      message="Train set: which files do you want to include?",
                      choices=[f for f in files if 'train' in files_lookup[f]],
    ),
    inquirer.Checkbox('validate',
                      message="Validation set: which files do you want to include?",
                      choices=[f for f in files if 'validate' in files_lookup[f]],
    ),
    inquirer.Checkbox('test',
                      message="Test set: which files do you want to include?",
                      choices=[f for f in files if 'test' in files_lookup[f]],
    ),
]
answers = inquirer.prompt(questions)

# create groups in dest file if they don't exist
dest_file.require_group('train')
dest_file.require_group('validate')
dest_file.require_group('test')

# train group
selected_files = answers['train']

for f in selected_files:
    n_keys = len(list(files_lookup[f]['train'].keys()))
    print("Copying %s train group (n_keys = %d) to dest file..." % (f, n_keys))
    for k in list(files_lookup[f]['train'].keys()):
        files_lookup[f]['container'].copy('train/%s'%k, dest_file['train'])
print("-> Dest train: n_key = %d" % len(list(dest_file['train'].keys())))

# validate group
selected_files = answers['validate']

for f in selected_files:
    n_keys = len(list(files_lookup[f]['validate'].keys()))
    print("Copying %s validate group (n_keys = %d) to dest file..." % (f, n_keys))
    for k in list(files_lookup[f]['validate'].keys()):
        files_lookup[f]['container'].copy('validate/%s'%k, dest_file['validate'])
print("-> Dest validate: n_key = %d" % len(list(dest_file['validate'].keys())))

# validate group
selected_files = answers['test']

for f in selected_files:
    n_keys = len(list(files_lookup[f]['test'].keys()))
    print("Copying %s test group (n_keys = %d) to dest file..." % (f, n_keys))
    for k in list(files_lookup[f]['test'].keys()):
        files_lookup[f]['container'].copy('test/%s'%k, dest_file['test'])
print("-> Dest test: n_key = %d" % len(list(dest_file['test'].keys())))

# clean up by closing all files
for f in files:
    files_lookup[f]['container'].close()

dest_file.close()

print("Done!")
