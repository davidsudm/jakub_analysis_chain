#!/bin/bash
#
# train_random_forests_run.sh allows you to run train_random_forests.sh in a more convenient way. Just replace the variables below accordingly to your needs
#

dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_1"
rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_1"
json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
intensity_cut=350
energy_training="gamma_diffuse"
production_name="config_1" # also, simtel_configuration_name

# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_2"
# rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_2"
# json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
# cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
# intensity_cut=350
# energy_training="gamma_diffuse"
# production_name="config_2" # also, simtel_configuration_name

# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_3"
# rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_3"
# json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
# cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
# intensity_cut=???
# energy_training="gamma_diffuse"
# production_name="config_3" # also, simtel_configuration_name

bash train_random_forests.sh $dl1_data_dir $rf_dir $json_config_file $cam_key $intensity_cut $energy_training $production_name
