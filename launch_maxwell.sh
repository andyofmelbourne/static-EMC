#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --partition=allgpu
#SBATCH --export=ALL
#SBATCH -J static_emc
#SBATCH -o .%J.out
#SBATCH -e .%J.out
#SBATCH --constraint="GPUx4&A100"

# Change the runs to process using the --array option on line 3

PREFIX=/gpfs/exfel/exp/SQS/202302/p003004

source /etc/profile.d/modules.sh
source ${PREFIX}/usr/Shared/xfel3004/source_this_at_euxfel

conda activate /home/amorgan/conda/envs/EMC

python static_emc_init.py
mpirun -np 32 python static_emc.py




