#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=diff-fwi
#SBATCH --partition=hive
#SBATCH --qos=hive_flash

#SBATCH --gres=gpu:1                      # Number of GPUs to allocate to this job
#SBATCH --time=00:10:00                   # The walltime

#SBATCH -e /scratch_hive/dp4018/scripts/diffusion-fwi-2d/slurm/slurm.err              # File to redirect stderr %j
#SBATCH -o /scratch_hive/dp4018/scripts/diffusion-fwi-2d/slurm/slurm.out              # File to redirect stdout %j

#SBATCH --nodes=1                          # Single node (multi-node requires different setup)
#SBATCH --ntasks=1                         # Single task (accelerate handles multi-GPU internally)
#SBATCH --ntasks-per-socket=2              # Max tasks per CPU socket
#SBATCH --cpus-per-task=16                 # CPU cores per task (more for data loading)
#SBATCH --mem=40G                          # Total memory for the job 


# This is where the actual work is done.

export HOME="/scratch_hive/dp4018"

module avail
module use /opt/nvidia/hpc_sdk/modulefiles
module load nvhpc

nvidia-smi
nvcc --version
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stride

# These are individual tasks
# srun python /scratch_hive/dp4018/scripts/diffusion-fwi-2d/scripts/training.py

num_nodes=1
num_workers_per_node=2
num_threads_per_worker=16
mrun --local -n $num_nodes -nw $num_workers_per_node -nth $num_threads_per_worker python /scratch_hive/dp4018/scripts/diffusion-fwi-2d/scripts/forward.py  &> stride-output.log
# srun python /scratch_hive/dp4018/scripts/diffusion-fwi-2d/scripts/inverse.py
wait

echo "Job completed successfully."

# Finish the script
exit 0
