

PYTHON=/proj/bolinc/users/x_sebsc/anaconda3/envs/nn-reanalysis-env/bin/python

for nettype in unet_hemconv_sharedweights unet_sphereconv_hemconv_shared; do
    for i_train in 0 1 2 3; do
sbatch << EOF
#!/bin/sh
#SBATCH -A snic2022-1-1
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output="eval_longterm_${nettype}_${i_train}.out"
${PYTHON} analyze_longterm_stability_hemconv.py ${nettype} ${i_train}
EOF
done
done

