

PYTHON=/proj/bolinc/users/x_sebsc/anaconda3/envs/nn-reanalysis-env/bin/python

for resolution in 1.40625 2.8125; do
for nettype in unet_base unet_hemconv_sharedweights unet_hemconv unet_sphereconv unet_sphereconv_hemconv unet_sphereconv_hemconv_shared; do
    for i_train in 0 1 2 3; do
sbatch << EOF
#!/bin/sh
#SBATCH -A snic2022-1-1
#SBATCH --time=03-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output="eval_per_fc_${nettype}_${resolution}_${i_train}.out"
${PYTHON} eval_net_fcs_per_fc.py ${nettype} ${resolution} ${i_train}
EOF
done
done
done

