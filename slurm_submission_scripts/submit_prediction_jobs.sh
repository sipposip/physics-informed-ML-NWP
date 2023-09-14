

PYTHON=/proj/bolinc/users/x_sebsc/anaconda3/envs/nn-reanalysis-env/bin/python

resolution="2.8125"
for nettype in unet_base unet_sphereconv unet_hemconv unet_sphereconv_hemconv unet_hemconv_sharedweights unet_sphereconv_hemconv_shared; do
    for i_train in 0 1 2 3; do
sbatch << EOF
#!/bin/sh
#SBATCH -A snic2020-1-31
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gpus-per-task=1
#SBATCH --output="predict_${nettype}_${resolution}_${i_train}.out"
${PYTHON} era5_predict_batch.py ${nettype} ${resolution} ${i_train}
EOF
done
done
