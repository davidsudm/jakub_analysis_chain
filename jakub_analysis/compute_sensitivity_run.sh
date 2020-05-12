#!/bin/bash
#
# compute_sensitivity_run.sh allows you to run compute_sensitivity.sh in a more convenient way. Just replace the variables below accordingly to your needs
#

dl2_data_dir="/work/se2/davidsudm/lst_sipm_task/DL2/config_1"
rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_1"
cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
intensity_min=350
production_name="config_1"

# dl2_data_dir="/work/se2/davidsudm/lst_sipm_task/DL2/config_2"
# rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_2"
# cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
# intensity_min=350
# production_name="config_2"

# dl2_data_dir="/work/se2/davidsudm/lst_sipm_task/DL2/config_3"
# rf_dir="/work/se2/davidsudm/lst_sipm_task/rf/config_3"
# cam_key="dl1/event/telescope/parameters/LST_LSiTCam"
# intensity_min=???
# production_name="config_3"

bash compute_sensitivity.sh $dl2_data_dir $rf_dir $cam_key $intensity_min $production_name
