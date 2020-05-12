#!/bin/bash
#
# BASH script to launch conversion of dl1 to dl2 files from the production. This conversion is done via Jakub's script. It doesn't used the lstchain method
#
#
# dl1_data_dir :                path to the directory where the dl1 production is stored, i.e. the input directory
# dl2_data_dir :                path to the directory where the dl2 production will be stored, i.e. the output directory
# rf_dir :                      porcentage of events to be trained, integer number from 1 to 100.
# json_config_file :            path to the configuration file requiered to run the RF training (similar file to the json used by "lstchain_mc_r0_to_dl1)
# cam_key :                     path searched within the *.h5 file, defines the telescope characteristic. Either dl1/event/telescope/parameters/LST_LSiTCam or dl1/event/telescope/parameters/LST_LPMTCam
# production_name :             name of the simtel configuration (also known as simtel production), useful to trace in the log files if issues appear (it is a name tag)
#
# EXAMPLE :
#
# dl1_data_dir                  /work/se2/davidsudm/lst_sipm_task/DL1/config_2
# dl2_data_dir                  /work/se2/davidsudm/lst_sipm_task/DL2/config_2
# rf_dir                        /work/se2/davidsudm/lst_sipm_task/rf/config_2
# json_config_file              /work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_bis.json
# cam_key                       dl1/event/telescope/parameters/LST_LSiTCam
# production_name               config_2
#
# USAGE :
# bash convert_dl1_to_dl2.sh /work/se2/davidsudm/lst_sipm_task/DL1/config_2 /work/se2/davidsudm/lst_sipm_task/DL2/config_2 /work/se2/davidsudm/lst_sipm_task/rf/config_2 /work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_own_prod_config_SiPM_bis.json dl1/event/telescope/parameters/LST_LSiTCam config_2

dl1_data_dir=$1
dl2_data_dir=$2
rf_dir=$3
json_config_file=$4
cam_key=$5
production_name=$6

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  dl1_data_dir :     $1 "
echo "*-*-*-*-*-*-*-*-*-*  dl2_data_dir :     $2 "
echo "*-*-*-*-*-*-*-*-*-*  rf_dir :           $3 "
echo "*-*-*-*-*-*-*-*-*-*  json_config_file : $4 "
echo "*-*-*-*-*-*-*-*-*-*  cam_key :          $5 "
echo "*-*-*-*-*-*-*-*-*-*  production_name :  $6 "
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

source /work/se1/unige/software/SetupPackages.sh
conda activate lst-dev-our_MC

telescope=3
sensitivity_scripts_dir="/work/se1/unige/software/prog/lst/unige_sensitivity"

echo "start conversion for prod $production_name"
if [ ! -e $json_config_file ]; then
  echo "config file not found: $json_config_file"
  exit -1
fi

output_dl2_dir="$dl2_data_dir/jakub"
mkdir -p $output_dl2_dir
cp $json_config_file $output_dl2_dir
energy_model="${rf_dir}/jakub/energy_rf/energy_rf_model.joblib"
disp_model="${rf_dir}/jakub/disp_rf/disp_rf_model.joblib"
sep_model="${rf_dir}/jakub/gh_sep_rf/separation_rf_model.joblib"
for particle in proton gamma gamma_diffuse ; do # Take out electron
    input_dl1_file="${dl1_data_dir}/dl1_${particle}_merge_test.h5"
    output_dl2_file="${output_dl2_dir}/dl2_${particle}_test.h5"
    weight_file="${output_dl2_file/.h5/_weight.h5}"
    if [ -e $output_dl2_file ]; then
        echo "skiping conversion to DL2 as $output_dl2_file exists"
    else
        echo "creating $output_dl2_file"
        python -u $sensitivity_scripts_dir/all_reco.py --energy_model=$energy_model --disp_model=$disp_model --sep_model=$sep_model --input=$input_dl1_file --output=$output_dl2_file --cam_key=$cam_key --config_file=$json_config_file --telescope=$telescope &> ${output_dl2_file/.h5/.log}
    fi

    echo "computing $particle rate"
    weight_file="${output_dl2_file/.h5/_weight.h5}"
    if [ -e $weight_file ]; then
        echo "skip weight creation as $weight_file exists."
        continue
    fi
    if [[ $particle == "proton" ]]; then
        python -u $sensitivity_scripts_dir/proton_rate.py --config_file=$json_config_file --telescope=$telescope --cam_key=$cam_key --output=$weight_file $input_dl1_file &> ${weight_file/.h5/.log}
    elif [[ $particle == "gamma" ]]; then
        python -u $sensitivity_scripts_dir/gamma_rate.py --config_file=$json_config_file --telescope=$telescope --cam_key=$cam_key --output=$weight_file $input_dl1_file &> ${weight_file/.h5/.log}
    else
        echo "computing of $particle rate is not implemented"
        continue
    fi
done
