

PYTHON=/proj/bolinc/users/x_sebsc/anaconda3/envs/nn-reanalysis-env/bin/python


resolution="1.40625"
for nettype in  unet_sphereconv unet_sphereconv_hemconv unet_sphereconv_hemconv_shared; do
    for i_train in 0 1 2 3; do
sbatch << EOF
#!/bin/sh
#SBATCH -A snic2021-1-2
#SBATCH --time=7-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gpus-per-task=1
#SBATCH --output="train_${nettype}_${resolution}_${i_train}_continued.out"

# copy files to local scratch
cp -r /proj/bolinc/users/x_sebsc/nn_reanalysis/era5_benchmarkdata/tfrecord_${resolution} /scratch/local/

${PYTHON} continue_train.py ${nettype} ${resolution} ${i_train}
EOF
    done
done


