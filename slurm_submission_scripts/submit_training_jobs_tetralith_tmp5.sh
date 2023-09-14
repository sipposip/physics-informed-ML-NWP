

PYTHON=/proj/bolinc/users/x_sebsc/anaconda3/envs/nn-reanalysis-env/bin/python



resolution="1.40625"
nettype=unet_base
i_train=0
sbatch << EOF
#!/bin/sh
#SBATCH -A snic2020-1-31
#SBATCH --time=7-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gpus-per-task=1
#SBATCH --output="train_${nettype}_${resolution}_${i_train}.out"

# copy files to local scratch
cp -r /proj/bolinc/users/x_sebsc/nn_reanalysis/era5_benchmarkdata/tfrecord_${resolution} /scratch/local/

${PYTHON} era5_train_batch_tetralith.py ${nettype} ${resolution} ${i_train}
EOF


