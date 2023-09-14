

PYTHON=/pfs/nobackup/home/s/sebsc/miniconda3/envs/nn-reanalysis-env/bin/python

# submit file for unet_base. i_train=0 was already trained for
# testing, therefore we can skip it here.


resolution="2.8125"
nettype=unet_hemconv
    for i_train in 2 3; do
sbatch << EOF
#!/bin/sh
#SBATCH -A SNIC2020-5-628
#SBATCH --time=4-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:v100:1
#SBATCH --output="train_${nettype}_${resolution}_${i_train}.out"
${PYTHON} era5_train_batch.py ${nettype} ${resolution} ${i_train}
EOF
    done

