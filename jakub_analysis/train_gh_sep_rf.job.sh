#!/bin/bash
#PBS -l walltime=3:00:00
#PBS -q qbigmem
#PBS -l nodes=1:ppn=16
#PBS -l vmem=12000mb

# needed variables:
# config_file
# train_proton
# train_gamma
# test_proton
# test_gamma
# output_dir
# cam_key
# intensity_cut

source /work/se1/unige/software/SetupPackages.sh
conda activate lst-dev-our_MC

cd $output_dir
/work/se1/unige/software/prog/lst/unige_sensitivity/gh_sep_train.py --config_file=$config_file --train_gamma=$train_gamma --train_proton=$train_proton --test_gamma=$test_gamma --test_proton=$test_proton --telescope=3 --cam_key="$cam_key" --intensity_cut=$intensity_cut
