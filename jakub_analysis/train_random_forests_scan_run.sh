#!/bin/bash
#
# train_random_forests_run.sh allows you to run train_random_forests.sh in a more convenient way. Just replace the variables below accordingly to your needs
#

dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_1"
rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_1"
json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
intensity_cut=0
energy_training="gamma_diffuse"
production_name="config_1" # also, simtel_configuration_name

# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_2"
# rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_2"
# json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
# cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
# intensity_cut=0
# energy_training="gamma_diffuse"
# production_name="config_2" # also, simtel_configuration_name

# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_3"
# rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_3"
# json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
# cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
# intensity_cut="350"
# energy_training="gamma_diffuse"
# production_name="config_3" # also, simtel_configuration_name

intensity_cut=( 100 150 200 250 300 350 400 450 500 550 600 700 800 1000 1500 2000 )

for cut in ${intensity_cut[@]}; do
  rf_dir_cut="$rf_dir/intensity_cut_scans/$cut"
  echo "rf directory for cut $cut : $rf_dir_cut "
  bash train_random_forests.sh $dl1_data_dir $rf_dir_cut $json_config_file $cam_key $cut $energy_training $production_name

  sleep 0.1
done
