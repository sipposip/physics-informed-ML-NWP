

PYTHON=/proj/bolinc/users/x_sebsc/anaconda3/envs/nn-reanalysis-env/bin/python


resolution="2.8125"

nettype=unet_hemconv_halfsize
for i_train in 0 1 2 3; do
sbatch << EOF
#!/bin/sh
#SBATCH -A naiss2023-1-5
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output="eval_${nettype}_${resolution}_${i_train}.out"
${PYTHON} eval_net_fcs_batch.py ${nettype} ${resolution} ${i_train}
EOF
done
