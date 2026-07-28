#!/bin/bash -l
# eval_reference_seed0.sh
# One-off: generate samples.npy for the reference model (seed 0), needed by
# Figure 3, and its upd_metrics.json (one row of Table 2).
#
# First attempt: samples.npy saved fine, but the metric-computation phase
# crashed (URLError: "Cannot assign requested address") trying to download
# torchvision's ImageNet-pretrained InceptionV3 (for FID-In, separate from
# the RadImageNet checkpoint already restored via
# download_radimagenet_weights.py) -- the compute node has no general
# internet access without `module load eth_proxy` (the pattern every other
# internet-needing job script in this repo uses; this one was missing it).
# Also note: without `set -e` / explicit exit-code checking, this script
# would report success even if `python evaluate.py` crashed -- add both.
#
# Since samples.npy already exists, evaluate.py's own check will skip
# regeneration and go straight to (now-working) metric computation.

#SBATCH --job-name=diff-eval-reference-s0
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/eval_%j.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/eval_%j.err
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G

set -e
module load eth_proxy
source $HOME/envs/diffusion_env/bin/activate
export TORCH_HOME=/cluster/scratch/fscharitzer/models/torch_cache

cd /cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
python paper-pipeline/evaluate.py --name reference --seed 0

echo "Evaluation done."
