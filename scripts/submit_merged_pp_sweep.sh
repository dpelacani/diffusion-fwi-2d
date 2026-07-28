#!/bin/bash -l
# Denoising guidance, merged (DG) hyperparameter sweep (a 1D alpha grid
# as alpha_skull == alpha_tissue)
# Grid: 3 brains x 4 alpha values = 12 array jobs.

#SBATCH --job-name=dg-merged-sweep
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/mergedpp_%A_%a.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/mergedpp_%A_%a.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-11%20

########################################################
############# Module and Environment Setup #############
########################################################

module load stack/2025-06
module load nvhpc/24.9
module load eth_proxy

export DEVITO_ARCH=gcc
export DEVITO_LANGUAGE=C
export TMPDIR=/cluster/scratch/fscharitzer/devito-tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export DEVITO_JITCACHE=/cluster/home/fscharitzer/devito-jitcache-gcc-C

export MOSAIC_PORT=$(( 4000 + SLURM_ARRAY_TASK_ID * 200 ))

mkdir -p $TMPDIR

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/home/fscharitzer/envs/stride

REPO=/cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
SCRIPTS=$REPO/paper-pipeline

########################################################
#  Grid layout
#  task_id = brain_idx * 4 + alpha_idx
#  alpha ∈ {0.3, 0.5, 0.7, 0.9}
########################################################
BRAINS=(vp_177746 vp_268749 vp_580347)
ALPHAS=(0.3 0.5 0.7 0.9)

N_ALPHA=${#ALPHAS[@]}

BRAIN_IDX=$(( SLURM_ARRAY_TASK_ID / N_ALPHA ))
ALPHA_IDX=$(( SLURM_ARRAY_TASK_ID % N_ALPHA ))

ALPHA=${ALPHAS[$ALPHA_IDX]}
MODEL=${BRAINS[$BRAIN_IDX]}

echo "=== Task ${SLURM_ARRAY_TASK_ID}: brain=${MODEL}  alpha=${ALPHA} (merged) ==="

########################################################
################## Forward (if needed) #################
########################################################

NAME_UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
FORWARD_DIR="/cluster/scratch/fscharitzer/exps/forward/${NAME_UPPER}"
ACQ_FILE="${FORWARD_DIR}/${NAME_UPPER}-Acquisitions.h5"

if [ -f "$ACQ_FILE" ]; then
    echo "Acquisitions already exist for ${MODEL}, skipping forward."
else
    echo "--- Forward modelling: ${MODEL} ---"
    mkdir -p "$FORWARD_DIR"
    (cd $TMPDIR && MODEL_NAME=$MODEL mrun --local -n 1 -nw 32 -nth 4 \
        python $SCRIPTS/forward.py)
    if [ $? -ne 0 ]; then
        echo "ERROR: forward modelling failed for ${MODEL}"
        rm -rf $TMPDIR
        exit 1
    fi
    echo "[done] Forward: ${MODEL}"
fi

########################################################
#################### Sweep inversion ##################
########################################################

(cd $TMPDIR && \
    MODEL_NAME=$MODEL \
    SWEEP_T_START=500 \
    SWEEP_ALPHA_SKULL=$ALPHA \
    SWEEP_ALPHA_TISSUE=$ALPHA \
    SWEEP_WARMSTART_ITERS=8 \
    SWEEP_OUT_DIR=/cluster/scratch/fscharitzer/inversion/merged_pp_sweep \
    mrun --local -n 1 -nw 2 -nth 64 \
        python $SCRIPTS/fwi_sweep_ppsplit.py)

EXIT_CODE=$?
rm -rf $TMPDIR

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: sweep point failed (brain=${MODEL} alpha=${ALPHA})"
    exit 1
fi

echo "[task done] ${MODEL}  alpha=${ALPHA}"
exit 0
