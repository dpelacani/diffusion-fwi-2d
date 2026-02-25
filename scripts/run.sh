#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=diff-fwi
#SBATCH --partition=hive
#SBATCH --qos=hive_intermediate

#SBATCH --gres=gpu:1                      # Number of GPUs to allocate to this job
#SBATCH --time=12:00:00                   # The walltime

# Change these paths if you want to redirect output to different files or locations
#SBATCH -e /scratch_hive/dp4018/scripts/diffusion-fwi-2d/slurm/slurm.err              # File to redirect stderr %j
#SBATCH -o /scratch_hive/dp4018/scripts/diffusion-fwi-2d/slurm/slurm.out              # File to redirect stdout %j

#SBATCH --nodes=1                          # Single node (multi-node requires different setup)
#SBATCH --ntasks=1                         # Single task (accelerate handles multi-GPU internally)
#SBATCH --ntasks-per-socket=2              # Max tasks per CPU socket
#SBATCH --cpus-per-task=16                 # CPU cores per task (more for data loading)
#SBATCH --mem=40G                          # Total memory for the job 


########################################################
############# Module and Environment Setup #############
########################################################

module avail
module use /opt/nvidia/hpc_sdk/modulefiles
module load nvhpc

nvidia-smi
nvcc --version
source /scratch_hive/dp4018/miniconda3/etc/profile.d/conda.sh # change this path if your conda is installed elsewhere
conda activate stride

num_nodes=1
num_workers_per_node=2
num_threads_per_worker=16


########################################################
############# Training the Diffusion model #############
########################################################

# This performs training of the diffusion model.
# srun python /scratch_hive/dp4018/scripts/diffusion-fwi-2d/scripts/training.py

########################################################
############## Running Forward Modelling ###############
########################################################

# # This performs the forward modelling to generate the synthetic data.
# mrun --local -n $num_nodes -nw $num_workers_per_node -nth $num_threads_per_worker python /scratch_hive/dp4018/scripts/diffusion-fwi-2d/scripts/forward.py  &> stride-output.log

########################################################
############## Running Inverse Modelling ###############
########################################################

# This performs the inversion to reconstruct the underlying model from the synthetic data, with or without diffusion
mrun --local -n $num_nodes -nw $num_workers_per_node -nth $num_threads_per_worker python /scratch_hive/dp4018/scripts/diffusion-fwi-2d/scripts/inverse.py  &> stride-output.log



wait

echo "Job completed successfully."

# Finish the script
exit 0
