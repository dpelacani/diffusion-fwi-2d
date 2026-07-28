#!/bin/bash
# Run from the repo root. Submits the full N=10-seed evaluation grid --
# 6 experiments x 10 seeds, one array task per (experiment, seed) pair.
# See run_eval_array.sh for the task_id -> (experiment, seed) decoding.

REPO=/cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
ARRAY_SCRIPT=$REPO/scripts/run_eval_array.sh

N_EXPS=6   # reference, size_small, size_large, no_split, aug_none, aug_full
N_SEEDS=10 # seeds 0-9, per diff_experiments.py's BASE["seeds"]
N_TASKS=$(( N_EXPS * N_SEEDS ))

sbatch \
  --array=0-$((N_TASKS - 1))%20 \
  "$ARRAY_SCRIPT"

echo "Submitted ${N_TASKS} evaluation jobs (array 0-$((N_TASKS - 1)), ${N_EXPS} experiments x ${N_SEEDS} seeds)."
echo "Monitor: squeue -u \$USER -o '%.10i %.9P %.20j %.8u %.8T %.10M %.9l %.6D %R'"
echo ""
echo "Each task writes samples.npy + metrics.json to its run_dir."
echo "Rerunning the same array later (e.g. after the tissue_statistics rewrite)"
echo "will skip generation and only recompute metrics."