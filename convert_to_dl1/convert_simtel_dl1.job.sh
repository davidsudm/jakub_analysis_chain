#PBS -l walltime=24:00:00
#PBS -q qlong
#PBS -l vmem=1500mb

# script need folowing variables to be passed to qsub with -v option:
# PBS_ARRAYID (set by PBS when called with -t )
# nfile_per_job
# list_simtel_files
# h5_dir
# config_file

source /work/se1/unige/software/SetupPackages.sh
conda activate lst-dev-our_MC

echo "PBS_ARRAYID: $PBS_ARRAYID"
echo "list_simtel_files: $list_simtel_files"
simtel_files=( $(cat $list_simtel_files) )
index_job=$(( $PBS_ARRAYID - 1 ))
first_job=$(( $index_job * $nfile_per_job ))
last_job=$(( $(( $index_job+1 )) * $nfile_per_job - 1 ))
if [ $last_job -gt ${#simtel_files[@]} ]; then
    last_job=${#simtel_files[@]}
fi
for i in $(seq $first_job $last_job); do
    simtel_file=${simtel_files[$i]}
    echo "simtel_file: $simtel_file"
    file_basename="$(basename $simtel_file)"
    output_file="$h5_dir/dl1_${file_basename/.gz/.h5}"
    if [ -e $output_file ]; then
        echo "$output_file exists, skipping conversion to DL1"
        exit 0
    fi
    lstchain_mc_r0_to_dl1 --infile $simtel_file --outdir $h5_dir --config_file $config_file
done
