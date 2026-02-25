#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=diff-fwi
#SBATCH --partition=hive
#SBATCH --qos=hive_flash

#SBATCH --gres=gpu:1                      # Number of GPUs to allocate to this job
#SBATCH --time=00:10:00                   # The walltime

#SBATCH -e /scratch_hive/dp4018/scripts/diffusion-fwi-2d/slurm/slurm-%j.err              # File to redirect stderr
#SBATCH -o /scratch_hive/dp4018/scripts/diffusion-fwi-2d/slurm/slurm-%j.out              # File to redirect stdout
#SBATCH --nodes=1                          # Single node (multi-node requires different setup)
#SBATCH --ntasks=1                         # Single task (accelerate handles multi-GPU internally)
#SBATCH --ntasks-per-socket=1              # Max tasks per CPU socket
#SBATCH --cpus-per-task=12                 # CPU cores per task (more for data loading)
#SBATCH --mem=40G                          # Total memory for the job 


# This is where the actual work is done.

export HOME="/scratch_hive/dp4018"

nvidia-smi
nvcc --version
source ~/anaconda3/etc/profile.d/conda.sh
conda activate df2i

# These are individual tasks
srun python /scratch_hive/dp4018/scripts/diffusion-fwi-2d/diffusion.py
wait

# Finish the script
exit 0
