#!/bin/bash
#
# BASH script to launch jobs to convert simtel files from a whole production into dl1 using lstchain (lstchain_mc_r0_to_dl1)
#
# simtel_data_dir :             path to the directory where the simtel production is saved for a given simtel configuration, i.e. the input directory
# dl1_data_dir :                path to the directory where the dl1 files will be stored, i.e. the output directory
# json_config_file :            path to the configuration file requiered by "lstchain_mc_r0_to_dl1"
# production_name :             name of the simtel configuration (also known as simtel production), useful to trace in the log files if issues appear (it is a name tag)
#
# EXAMPLE :
#
# simtel_data_dir :             /work/se2/davidsudm/lst_sipm_task/simtel/config_2/LST-SiPM-20deg
# dl1_data_dir :                /work/se2/davidsudm/lst_sipm_task/DL1/config_2
# json_data_dir :               /work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_standard_config_bis.json
# production_name :             config_2
#
# USAGE :
#
# bash convert_simtel_dl1.sh /work/se2/davidsudm/lst_sipm_task/simtel/config_2/LST-SiPM-20deg /work/se2/davidsudm/lst_sipm_task/DL1/config_2 /work/se1/unige/software/lstchain/cta-lstchain.ourMC/lstchain/data/lstchain_standard_config_bis.json config_2

simtel_data_dir=$1
dl1_data_dir=$2
json_config_file=$3
production_name=$4

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  simtel_data_dir :      $1 "
echo "*-*-*-*-*-*-*-*-*-*  dl1_data_dir :         $2 "
echo "*-*-*-*-*-*-*-*-*-*  json_config_file :     $3 "
echo "*-*-*-*-*-*-*-*-*-*  production_name :      $4 "
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

mkdir -p $dl1_data_dir
nfile_per_job=10

if [ ! -e $json_config_file ]; then
    echo "config file not found: $json_config_file"
    exit -1
fi

for particle in proton ; do # take out : electrons gamma gamma_diffuse
    list_simtel_files="$simtel_data_dir/$particle/list.txt"
    ls $simtel_data_dir/$particle/*.simtel.gz | sort -V > $list_simtel_files
    n_simtel_files=$(cat $list_simtel_files | wc -l)
    echo "n_simtel_files: $n_simtel_files"
    n_jobs=$(( $n_simtel_files / $nfile_per_job ))
    if [ $(( $n_jobs*$nfile_per_job )) -lt $n_simtel_files ]; then
        n_jobs=$(( $n_jobs + 1 ))
    fi
    h5_dir="$dl1_data_dir/$particle"
    mkdir -p $h5_dir
    #one job per simtelfile, at maximum 50 jobs running:
    PBSARRAY="1-${n_jobs}%50"
    qsub -t $PBSARRAY -o $h5_dir -e $h5_dir -v h5_dir=$h5_dir,list_simtel_files=$list_simtel_files,config_file=$json_config_file,nfile_per_job=$nfile_per_job -N dl0to1_${particle} convert_simtel_dl1.job.sh;
    sleep 0.1

    echo "particle $particle"
    echo "simtel_data_dir $simtel_data_dir"
    echo "dl1_data_dir $dl1_data_dir"
    echo "h5_dir $h5_dir"

done

#lstchain_mc_r0_to_dl1
