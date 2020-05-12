#!/bin/bash
#PBS -l walltime=3:00:00
#PBS -q qbigmem
#PBS -l nodes=1:ppn=16
#PBS -l vmem=12000mb

# needed variables:
# config_file
# train_file
# test_file
# output_dir
# cam_key
# intensity_cut

source /work/se1/unige/software/SetupPackages.sh
conda activate lst-dev-our_MC

cd $output_dir
/work/se1/unige/software/prog/lst/unige_sensitivity/disp_train.py --config_file=$config_file --train_file=$train_file --test_file=$test_file --telescope=3 --cam_key="$cam_key" --intensity_cut=$intensity_cut
