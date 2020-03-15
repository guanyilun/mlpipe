#!/bin/bash

# The aim of this script is to generate lists of
# TODs for cross season study of PA2. It will generate
# the following list:

# Train set:
# s14 80 TODs, s15 80 TODs

# validation set:
# s14 20 TODs, s15 20 TODs, s16 20 TODs

# Test set:
# s16 40 TODs

cd /mnt/act3/users/yilun/work/act-cutflow/

# s14
echo Generating list for s14...
./bin/generate_tod_list.py -t pa2_s14_c10_v4 -p ../../share/pa2/pa2_s14_c10_v4_results.pickle -o inputs/ --n_train=80  --n_validate=20 --n_test=0

# s15
echo Generating list for s15...
./bin/generate_tod_list.py -t pa2_s15_c10_v4 -p ../../share/pa2/pa2_s15_c10_v4_results.pickle -o inputs/ --n_train=80  --n_validate=20 --n_test=0

# s16
echo Generating list for s16
./bin/generate_tod_list.py -t mr3_pa2_s16 -p ../../share/pa2/mr3_pa2_s16_results.pickle -o inputs/ --n_train=0 --n_validate=20 --n_test=40

