#!/bin/bash -l
#SBATCH --job-name=diff-bounds
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/bounds_%j.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/bounds_%j.err

set -e
module load eth_proxy
export TORCH_HOME=/cluster/scratch/fscharitzer/models/torch_cache
source $HOME/envs/diffusion_env/bin/activate
cd /cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
python paper-pipeline/compute_bounds.py