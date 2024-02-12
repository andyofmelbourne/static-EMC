#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=allgpu
#SBATCH --export=ALL
#SBATCH -J static_emc
#SBATCH -o .%J.out
#SBATCH -e .%J.out
#SBATCH --constraint="GPUx4&A100"

# fail on first error 
set -e

source /etc/profile.d/modules.sh
module load exfel exfel-python
conda activate /home/amorgan/.conda/envs/EMC

python static_emc_init.py
mpirun -np 32 python static_emc.py




