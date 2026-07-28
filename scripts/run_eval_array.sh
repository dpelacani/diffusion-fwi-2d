#!/bin/bash -l
# One array task = one (experiment, seed) pair (generation + metrics), the
# full N=10-seed grid behind run_eval_seed0_array.sh's seed-0-only preview.
# Skips generation if samples.npy already exists (handled inside evaluate.py).

#SBATCH --job-name=diff-eval-array
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/eval_%A_%a.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/eval_%A_%a.err

set -e
module load eth_proxy

########################################################
# Environment
########################################################
export TORCH_HOME=/cluster/scratch/fscharitzer/models/torch_cache
source $HOME/envs/diffusion_env/bin/activate

REPO=/cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
SCRIPTS=$REPO/paper-pipeline

########################################################
# Experiment x seed selection
# NOTE: EXPS must match the "name" field of EXPERIMENTS in diff_experiments.py,
# in order; SEEDS must match diff_experiments.py's BASE["seeds"].
# task_id = exp_idx * N_SEEDS + seed_idx
########################################################
EXPS=(reference size_small size_large no_split aug_none aug_full)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_SEEDS=${#SEEDS[@]}

EXP_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
EXP_NAME=${EXPS[$EXP_IDX]}
SEED=${SEEDS[$SEED_IDX]}

if [ -z "$EXP_NAME" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: index out of range, skipping."
    exit 0
fi

echo "=========================================="
echo " Task: ${SLURM_ARRAY_TASK_ID}"
echo " Exp:  ${EXP_NAME}  seed: ${SEED}"
echo "=========================================="

python $SCRIPTS/evaluate.py --name "$EXP_NAME" --seed "$SEED"

echo "Evaluation done."