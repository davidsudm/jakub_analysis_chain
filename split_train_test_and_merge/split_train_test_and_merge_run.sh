#!/bin/bash
#
# split_train_test_and_merge_run.sh allows you to run split_train_test_and_merge.sh in a more convenient way.
# Just replace the variables below accordingly to your needs
#

dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_1"
percent_event_train=50
production_name="config_1"

# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_2"
# percent_event_train=50
# production_name="config_2"

# dl1_data_dir="/work/se2/davidsudm/lst_sipm_task/DL1/config_3"
# percent_event_train=50
# production_name="config_3"

bash split_train_test_and_merge.sh $dl1_data_dir $percent_event_train $production_name
