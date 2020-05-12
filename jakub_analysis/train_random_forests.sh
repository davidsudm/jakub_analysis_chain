#!/bin/bash
#
# BASH script to launch jobs training via RFs (random forests) the dl1 files from the production
#
#
# dl1_data_dir :                path to the directory where the dl1 production is stored, i.e. the input directory
# rf_dir :                      porcentage of events to be trained, integer number from 1 to 100.
# json_config_file :            path to the configuration file requiered to run the RF training (similar file to the json used by "lstchain_mc_r0_to_dl1)
# cam_key :                     path searched within the *.h5 file, defines the telescope characteristic. Either dl1/event/telescope/parameters/LST_LSiTCam or dl1/event/telescope/parameters/LST_LPMTCam
# intensity_cut :               cut on the number of photo-electrons on camera. A number too high will supress low energy events.
# production_name :             name of the simtel configuration (also known as simtel production), useful to trace in the log files if issues appear (it is a name tag)
#
# EXAMPLE :
#
# dl1_data_dir :                /work/se2/davidsudm/lst_sipm_task/DL1/config_2
# rf_dir :                      /work/se2/davidsudm/lst_sipm_task/rf/config_2
# json_config_file :            /work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_bis.json
# cam_key :                     dl1/event/telescope/parameters/LST_LSiTCam
# intensity_cut :               200
# energy_training :             Either "gamma_point_like" or "gamma_diffuse". IMPORTANT :  if "gamma_point_like" is chosen, then the json_config_file should not contain the RF traininnig option for "leakage"
# production_name :             config_2 # also, simtel_configuration_name
#
# USAGE :
#
# bash train_random_forests.sh /work/se2/davidsudm/lst_sipm_task/DL1/config_2 /work/se2/davidsudm/lst_sipm_task/rf/config_2 /work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_bis.json dl1/event/telescope/parameters/LST_LSiTCam gamma_diffuse 200 config_2


dl1_data_dir=$1
rf_dir=$2
json_config_file=$3
cam_key=$4
intensity_cut=$5
energy_training=$6
production_name=$7

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  dl1_data_dir :       $1 "
echo "*-*-*-*-*-*-*-*-*-*  rf_dir :             $2 "
echo "*-*-*-*-*-*-*-*-*-*  json_config_file :   $3 "
echo "*-*-*-*-*-*-*-*-*-*  cam_key :            $4 "
echo "*-*-*-*-*-*-*-*-*-*  intensity_cut :      $5 "
echo "*-*-*-*-*-*-*-*-*-*  energy_training :    $6 "
echo "*-*-*-*-*-*-*-*-*-*  production_name :    $7 "
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

echo "run the merging for production $production_name"
mkdir -p $rf_dir

gamma_train="$dl1_data_dir/dl1_gamma_merge_train.h5"
gamma_test="$dl1_data_dir/dl1_gamma_merge_test.h5"
gamma_diffuse_train="$dl1_data_dir/dl1_gamma_diffuse_merge_train.h5"
gamma_diffuse_test="$dl1_data_dir/dl1_gamma_diffuse_merge_test.h5"
proton_train="$dl1_data_dir/dl1_proton_merge_train.h5"
proton_test="$dl1_data_dir/dl1_proton_merge_test.h5"

# the following condition is important for the energy RF trainining (look above in "energy_training" option)
if [[ $energy_training == "gamma_point_like" ]]; then
  gamma_training_file=$gamma_train
elif [[ $energy_training == "gamma_diffuse" ]]; then
  gamma_training_file=$gamma_diffuse_train
else
  echo "energy_training not valid : gamma_point_like or gamma_diffuse"
  exit -1
fi

disp_rf_dir="$rf_dir/jakub/disp_rf"
mkdir -p $disp_rf_dir
echo "train random forest for disp regression in $disp_rf_dir"
qsub -o ${disp_rf_dir}/log.out -e ${disp_rf_dir}/log.err -v config_file=$json_config_file,train_file=$gamma_diffuse_train,test_file=$gamma_test,output_dir=${disp_rf_dir},cam_key=${cam_key},intensity_cut=${intensity_cut} -N disp_rf train_disp_rf.job.sh;

energy_rf_dir="$rf_dir/jakub/energy_rf"
mkdir -p $energy_rf_dir
echo "train random forest for energy regression in $energy_rf_dir"
qsub -o ${energy_rf_dir}/log.out -e ${energy_rf_dir}/log.err -v config_file=$json_config_file,train_file=$gamma_training_file,test_file=$gamma_test,output_dir=${energy_rf_dir},cam_key=${cam_key},intensity_cut=${intensity_cut} -N energy_rf train_energy_rf.job.sh;

gh_sep_rf_dir="$rf_dir/jakub/gh_sep_rf"
mkdir -p $gh_sep_rf_dir
echo "train random forest for gamma hadron separation in $gh_sep_rf_dir"
qsub -o ${gh_sep_rf_dir}/log.out -e ${gh_sep_rf_dir}/log.err -v config_file=$json_config_file,train_proton=$proton_train,train_gamma=$gamma_diffuse_train,test_proton=$proton_test,test_gamma=$gamma_diffuse_test,output_dir=${gh_sep_rf_dir},cam_key=${cam_key},intensity_cut=${intensity_cut} -N gh_sep_rf train_gh_sep_rf.job.sh;
sleep 0.1
