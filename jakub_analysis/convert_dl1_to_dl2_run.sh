#!/bin/bash
#
# convert_dl1_to_dl2_run.sh allows you to run convert_dl1_to_dl2.sh in a more convenient way. Just replace the variables below accordingly to your needs

dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_1"
dl2_data_dir="/work/se2/davidsudm/lst_sipm_task/DL2/config_1"
rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_1"
json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
production_name="config_1"

# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_2"
# dl2_data_dir="/work/se2/davidsudm/lst_sipm_task/DL2/config_2"
# rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_2"
# json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
# cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
# production_name="config_2"

# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_3"
# dl2_data_dir="/work/se2/davidsudm/lst_sipm_task/DL2/config_3"
# rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_3"
# json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_david_run.json"
# cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
# production_name="config_3"

bash convert_dl1_to_dl2.sh $dl1_data_dir $dl2_data_dir $rf_dir $json_config_file $cam_key $production_name
