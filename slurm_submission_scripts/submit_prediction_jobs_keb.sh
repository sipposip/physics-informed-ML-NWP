

PYTHON=/pfs/nobackup/home/s/sebsc/miniconda3/envs/nn-reanalysis-env/bin/python

i_train=0
resolution="2.8125"
for nettype in unet_base unet_sphereconv unet_hemconv; do

sbatch << EOF
#!/bin/sh
#SBATCH -A SNIC2020-5-628
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --output="predict_${nettype}_${resolution}_${i_train}.out"
${PYTHON} era5_predict_batch.py ${nettype} ${resolution} ${i_train}
EOF
done
