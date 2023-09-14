
# due to version problems, installed miniconda in addition to anaconda,
# beacuse of this we also need to set the esmf env variable
PYTHON=/proj/bolinc/users/x_sebsc/miniconda3/envs/xesmf_env/bin/python
export ESMFMKFILE=/proj/bolinc/users/x_sebsc/miniconda3/envs/xesmf_env/lib/esmf.mk

for nettype in unet_base unet_hemconv_sharedweights unet_hemconv unet_sphereconv unet_sphereconv_hemconv unet_sphereconv_hemconv_shared; do
    for i_train in 0 1 2 3; do
sbatch << EOF
#!/bin/sh
#SBATCH -A snic2022-1-1
#SBATCH --time=03-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output="eval_${nettype}_regrid_${i_train}.out"
${PYTHON} eval_net_fcs_regrid.py ${nettype} ${i_train}
EOF
done
done

