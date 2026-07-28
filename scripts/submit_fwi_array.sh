#!/bin/bash
# submit_fwi_array.sh
# Run from the repo root. Submits baseline + all 4 guidance methods (reference prior).
# --dependency=afterok:3861476

REPO=/cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
BRAIN_LIST=$REPO/fwi_brains.txt
ARRAY_SCRIPT=$REPO/scripts/run_fwi_array.sh

N=$(wc -l < "$BRAIN_LIST")
LAST=$((N))
echo "Submitting for $((LAST + 1)) brains (array indices 0-${LAST})."

# All 5 methods run all 3 experiments (standard, reduced_compute, missing_low_freq) per brain

# Baseline: cheap, no diffusion model
sbatch \
  --job-name=fwi-baseline \
  --array=0-${LAST}%20 \
  --time=02:30:00 \
  --export=ALL,METHOD=baseline \
  "$ARRAY_SCRIPT"

# Score guidance, merged (SG)
sbatch \
  --job-name=fwi-sg \
  --array=0-${LAST}%20 \
  --time=02:30:00 \
  --export=ALL,METHOD=sg \
  "$ARRAY_SCRIPT"

# Score guidance, per-channel (SG-C)
sbatch \
  --job-name=fwi-sg-c \
  --array=0-${LAST}%20 \
  --time=02:30:00 \
  --export=ALL,METHOD=sg_c \
  "$ARRAY_SCRIPT"

# Denoising guidance, merged (DG)
sbatch \
  --job-name=fwi-dg \
  --array=0-${LAST}%20 \
  --time=02:30:00 \
  --export=ALL,METHOD=dg \
  "$ARRAY_SCRIPT"

# Denoising guidance, per-channel (DG-C)
sbatch \
  --job-name=fwi-dg-c \
  --array=0-${LAST}%20 \
  --time=02:30:00 \
  --export=ALL,METHOD=dg_c \
  "$ARRAY_SCRIPT"

echo ""
echo "All five arrays submitted."
echo "Monitor: squeue -u \$USER -o '%.10i %.9P %.20j %.8u %.8T %.10M %.9l %.6D %R'"
echo ""
echo "Progress (done files written):"
echo "  find /cluster/scratch/fscharitzer/inversion/final_runs_reference -name done | wc -l"
echo "  Expected when all complete: $(( (LAST + 1) * 3 * 5 ))"