#!/bin/bash
#
# convert_simtel_dl1_run.sh allows you to run convert_simtel_dl1.sh in a more convenient way. Just replace the four variables below accordingly to your needs
#

simtel_data_dir="/work/se2/davidsudm/lst_sipm_task/simtel/config_1/LST-SiPM-20deg"
dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_1"
json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_standard_config_david_run.json"
production_name="config_1" # also, simtel_configuration_name

# simtel_data_dir="/work/se2/davidsudm/lst_sipm_task/simtel/config_2/LST-SiPM-20deg"
# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_2"
# json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_standard_config_david_run.json"
# production_name="config_2" # also, simtel_configuration_name

# simtel_data_dir="/work/se2/davidsudm/lst_sipm_task/simtel/config_3/LST-SiPM-20deg"
# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_3"
# json_config_file="/work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_standard_config_david_run.json"
# production_name="config_3" # also, simtel_configuration_name

bash convert_simtel_dl1.sh $simtel_data_dir $dl1_data_dir $json_config_file $production_name
