#!/bin/bash -l
# Postprocessing-split hyperparameter sweep.
# Grid: 10 brains × 4 alpha_skull × 4 alpha_tissue = 160 array jobs.

#SBATCH --job-name=pp-split-sweep
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/ppsplit_%A_%a.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/ppsplit_%A_%a.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-59%20

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
#  task_id = brain_idx * 16 + skull_idx * 4 + tissue_idx
#
#  alpha_skull  ∈ {0.3, 0.5, 0.7, 0.9}  (4 values)
#  alpha_tissue ∈ {0.1, 0.3, 0.5, 0.7}  (4 values)
########################################################
BRAINS=(vp_177746 vp_268749 vp_580347)
ALPHA_SKULLS=(0.3 0.5 0.7 0.9)
ALPHA_TISSUES=(0.1 0.3 0.5 0.7 0.9)

N_BRAINS=${#BRAINS[@]}
N_SKULL=${#ALPHA_SKULLS[@]}
N_TISSUE=${#ALPHA_TISSUES[@]}
N_PER_BRAIN=$(( N_SKULL * N_TISSUE ))

TASK_ID=$SLURM_ARRAY_TASK_ID

BRAIN_IDX=$(( SLURM_ARRAY_TASK_ID / N_PER_BRAIN ))
REM=$(( SLURM_ARRAY_TASK_ID % N_PER_BRAIN ))
SKULL_IDX=$(( REM / N_TISSUE ))
TISSUE_IDX=$(( REM % N_TISSUE ))

ALPHA_SKULL=${ALPHA_SKULLS[$SKULL_IDX]}
ALPHA_TISSUE=${ALPHA_TISSUES[$TISSUE_IDX]}
MODEL=${BRAINS[$BRAIN_IDX]}

echo "=== Task ${SLURM_ARRAY_TASK_ID}: brain=${MODEL}  alpha_skull=${ALPHA_SKULL}  alpha_tissue=${ALPHA_TISSUE} ==="

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
    SWEEP_T_END=100 \
    SWEEP_ALPHA_SKULL=$ALPHA_SKULL \
    SWEEP_ALPHA_TISSUE=$ALPHA_TISSUE \
    SWEEP_WARMSTART_ITERS=8 \
    SWEEP_OUT_DIR=/cluster/scratch/fscharitzer/inversion/pp_split_sweep \
    mrun --local -n 1 -nw 2 -nth 64 \
        python $SCRIPTS/fwi_sweep_ppsplit.py)

EXIT_CODE=$?
rm -rf $TMPDIR

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: sweep point failed (brain=${MODEL} as=${ALPHA_SKULL} at=${ALPHA_TISSUE})"
    exit 1
fi

echo "[task done] ${MODEL}  ls=${LAMBDA_SKULL}  lt=${LAMBDA_TISSUE}"
exit 0