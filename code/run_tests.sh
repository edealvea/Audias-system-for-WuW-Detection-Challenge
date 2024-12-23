#!/bin/bash

###############################################################################################
# This script runs the evaluation of the system on the extended test set.
# It assumes that the model has been trained and the test set is available.
# In addition, it generates the evaluation results and saves them in the output directory.
# Finally, the script also evaluates the system performance and prints the results.
#
# The provided code and the following cofiguration paraters are just an example.
# You can change the code and the parameters according to your needs.
###############################################################################################

# Audio configuration
sampling_rate=16000 # Hz
time_window=1.5 # seconds
hop_size=0.256 # seconds

# WuW detection criteria
threshold=0.65
n_positives=2 # Minimum number of windows to detect a WuW event

# Output directory
output_dir="/home/ealvear/wuw_detection/wuw-challenge-2024-master/outputs/"
SYSID="LRN256MFCC"
SITE="UAM"

# Test file path
dataset_path="/home/ealvear/wuw_detection/wuw-challenge-2024-master/okey-aura-v1.1.0"
extended_dataset_path="/home/ealvear/wuw_detection/wuw-challenge-2024-master/okey-aura-v1.1.0/extended_test"
test_file="$dataset_path/metadata/test.tsv"
extended_test="$dataset_path/extended_test/test-extended.tsv"



# Model path
model_path="/home/ealvear/wuw_detection/wuw-challenge-2024-master/models/trained/LRN256MFCC.jit"
device="cuda"


# Run the test
python test_system.py \
    --sampling_rate $sampling_rate \
    --time_window $time_window \
    --hop $hop_size \
    --threshold $threshold \
    --dataset_path $dataset_path \
    --extended_dataset_path $extended_dataset_path\
    --test_tsv $test_file \
    --extended_test_tsv $extended_test_tsv\
    --output_dir $output_dir \
    --n_positives $n_positives \
    --sysid $SYSID \
    --model_path $model_path\
    --device $device

#python src/evaluate_system.py \
#    --results_dir $output_dir
