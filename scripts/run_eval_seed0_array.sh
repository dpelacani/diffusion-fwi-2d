#!/bin/bash -l
# run_eval_seed0_array.sh
# Quick partial preview of Table 2: evaluate the 5 remaining experiments
# that only have a seed0 checkpoint trained ("reference" already evaluated
# separately). One array task = one experiment, all at seed 0. This gives
# single-seed numbers per experiment (no real std across seeds) -- NOT the
# paper's N=10 statistic, just a sanity-check preview. The full N=10
# ablation needs 9 more training seeds per experiment first (54 runs,
# deferred pending a separate decision on that bigger commitment).

#SBATCH --job-name=diff-eval-seed0
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/eval_seed0_%A_%a.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/eval_seed0_%A_%a.err

set -e
module load eth_proxy
source $HOME/envs/diffusion_env/bin/activate
export TORCH_HOME=/cluster/scratch/fscharitzer/models/torch_cache

EXPS=(size_small size_large no_split aug_none aug_full)
EXP_NAME=${EXPS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$EXP_NAME" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: index out of range, skipping."
    exit 0
fi

echo "=========================================="
echo " Task: ${SLURM_ARRAY_TASK_ID}"
echo " Exp:  ${EXP_NAME}  seed: 0"
echo "=========================================="

cd /cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
python paper-pipeline/evaluate.py --name "$EXP_NAME" --seed 0

echo "Evaluation done."
