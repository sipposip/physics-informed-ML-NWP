

PYTHON=/proj/bolinc/users/x_sebsc/anaconda3/envs/nn-reanalysis-env/bin/python

# submit file for unet_base. i_train=0 was already trained for
# testing, therefore we can skip it here.


resolution="1.40625"
nettype=unet_sphereconv_hemconv_shared
    for i_train in 2 3; do
sbatch << EOF
#!/bin/sh
#SBATCH -A snic2021-1-2
#SBATCH --time=7-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gpus-per-task=1
#SBATCH --output="train_${nettype}_${resolution}_${i_train}.out"

# copy files to local scratch
cp -r /proj/bolinc/users/x_sebsc/nn_reanalysis/era5_benchmarkdata/tfrecord_${resolution} /scratch/local/

${PYTHON} era5_train_batch_tetralith.py ${nettype} ${resolution} ${i_train}
EOF
    done

