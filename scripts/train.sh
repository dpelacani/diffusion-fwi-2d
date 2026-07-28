#!/bin/bash -l
#SBATCH --job-name=diff-train
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=30:00:00
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/train_%j.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/train_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G

source $HOME/envs/diffusion_env/bin/activate
export TORCH_HOME=/cluster/scratch/fscharitzer/models/torch_cache
export WANDB_MODE=offline
export WANDB_DIR=/cluster/scratch/fscharitzer/wandb
mkdir -p $WANDB_DIR

cd /cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
python paper-pipeline/train.py --seed 0

echo "Job completed."
exit 0