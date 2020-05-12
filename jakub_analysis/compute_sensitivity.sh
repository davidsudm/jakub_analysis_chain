#!/bin/bash
# BASH script to compute sensitivity of a give MC simtel production via Jakub's method.
#
# dl2_data_dir :                path to the directory where the dl2 production is stored, i.e. the input directory
# rf_dir :                      porcentage of events to be trained, integer number from 1 to 100.
# cam_key :                     path searched within the *.h5 file, defines the telescope characteristic. Either dl1/event/telescope/parameters/LST_LSiTCam or dl1/event/telescope/parameters/LST_LPMTCam
# intensity_min                 value of the intensity ctus used during the FR training
# production_name :             name of the simtel configuration (also known as simtel production), useful to trace in the log files if issues appear (it is a name tag)
#
# EXAMPLE :
#
# dl2_data_dir                  /work/se2/davidsudm/lst_sipm_task/DL2/config_2
# rf_dir                        /work/se2/davidsudm/lst_sipm_task/rf/config_2
# cam_key                       dl1/event/telescope/parameters/LST_LSiTCam
# intensity_min                 350
# production_name               config_2
#
# USAGE :
# bash convert_dl1_to_dl2.sh /work/se2/davidsudm/lst_sipm_task/DL2/config_2 /work/se2/davidsudm/lst_sipm_task/rf/config_2 dl1/event/telescope/parameters/LST_LSiTCam 350 config_2

dl2_data_dir=$1
rf_dir=$2
cam_key=$3
intensity_min=$4
production_name=$5

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  dl2_data_dir :     $1 "
echo "*-*-*-*-*-*-*-*-*-*  rf_dir :           $2 "
echo "*-*-*-*-*-*-*-*-*-*  cam_key :          $3 "
echo "*-*-*-*-*-*-*-*-*-*  intensity_min :    $4 "
echo "*-*-*-*-*-*-*-*-*-*  production_name :  $5 "
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

source /work/se1/unige/software/SetupPackages.sh
conda activate lst-dev-our_MC

telescope=3
script_dir="$PWD/$(dirname $BASH_SOURCE)"

echo "compute sensitivity for prod $production_name"
dl2_prod_path="$dl2_data_dir/jakub"
dl2_gamma="$dl2_prod_path/dl2_gamma_test.h5"
gamma_weight=${dl2_gamma/.h5/_weight.h5}
dl2_proton="$dl2_prod_path/dl2_proton_test.h5"
proton_weight=${dl2_proton/.h5/_weight.h5}
dl2_gamma_diffuse="$dl2_prod_path/dl2_gamma_diffuse_test.h5"
gamma_diffuse_weight=${dl2_gamma_diffuse/.h5/_weight.h5}

cd $dl2_prod_path
python -u $script_dir/compute_sensitivity.py --input_gamma=$dl2_gamma --input_gamma_diffuse=$dl2_gamma_diffuse --input_proton=$dl2_proton --weights_gamma=$gamma_weight --weights_proton=$proton_weight --cam_key=$cam_key --intensity_min=$intensity_min

echo "done"
