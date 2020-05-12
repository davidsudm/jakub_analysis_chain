#!/bin/bash
#
# BASH script to launch jobs the merging of two set of  DL1 files from a whole production using
# lstchain (lstchain_merge_hdf5_files)
# A fraction of the total number of files in a given sub-production (a sub-production is a given
# particle folder inside the production folder, e.g. proton, or gamma) is used to merge the merge "train" file
# and the fraction is defined by the variable percent_event_train
# The totality of the files is used to create the merged "test" file
# The output files will be stored inside the production folder (and not in each sub-production folder)
#
#
# dl1_data_dir :                path to the directory where the dl1 production is stored, i.e. the input directory
# percent_event_train :         porcentage of events to be trained, integer number from 1 to 100.
# production_name :             name of the simtel configuration (also known as simtel production), useful to trace in the log files if issues appear (it is a name tag)
#
# EXAMPLE :
#
# dl1_data_dir :                /work/se2/davidsudm/lst_sipm_task/DL1/config_2
# percent_event_train :         50
# production_name :             config_2
#
# USAGE :
#
# bash split_train_test_and_merge.sh /work/se2/davidsudm/lst_sipm_task/DL1/config_2 50 config_2

dl1_data_dir=$1
percent_event_train=$2
production_name=$3

echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "*-*-*-*-*-*-*-*-*-*  dl1_data_dir :         $1 "
echo "*-*-*-*-*-*-*-*-*-*  percent_event_train :  $2 "
echo "*-*-*-*-*-*-*-*-*-*  production_name :      $3 "
echo "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"

echo "run the merging for production $production_name"

for particle in proton gamma gamma_diffuse; do # Take out : electron
    list_dl1_files="$dl1_data_dir/$particle/list.txt"
    ls $dl1_data_dir/$particle/*.h5 | grep -i -v merged |sort -V > $list_dl1_files
    n_dl1_files=$(cat $list_dl1_files | wc -l)
    n_dl1_train=$(( $n_dl1_files*$percent_event_train/100 ))
    for run in "test" "train" ; do
        if [[ "$run" == "train" ]]; then
            dl1_files_run=$(awk 'NR<='$n_dl1_train' {print}' $list_dl1_files)
        elif [[ "$run" == "test" ]]; then
            dl1_files_run=$(awk 'NR>'$n_dl1_train' {print}' $list_dl1_files)
        else
            "WARNING: unknown run \"$run\" skipped"
            continue
        fi
        run_dir="$dl1_data_dir/$particle/$run"
        mkdir -p $run_dir
        nfile_run=$(echo $dl1_files_run| wc -w)
        for f in $dl1_files_run; do
            link_name="$run_dir/$(basename $f)"
            if [ -e "$link_name" ]; then
                echo "$link_name exists"
                continue
            fi
            # ln -t $run_dir -s $f; # when work with file (link of a file)
            ln -s $f $run_dir/$(basename $f); # when work with link of file (link of link)
        done
        merged_file="$dl1_data_dir/dl1_${particle}_merge_${run}.h5"
        if [ -e $merged_file ]; then
            echo "skiping the $run dataset with $particle for prod $prod as $merged_file exists"
            continue
        fi
        echo "submitting merging of $nfile_run dl1 files for the $run dataset with $particle for prod $prod."
        echo "the output is in $merged_file"
        qsub -o ${merged_file/.h5/.log} -e ${merged_file/.h5/.err} -v run_dir=$run_dir,merged_file=$merged_file -N merge_dl1_${particle}_${run} split_train_test_and_merge.job.sh;
        sleep 0.1
    done
done

# lstchain_merge_hdf5_files
