#!/bin/bash -l
# Score guidance, per-channel (SG-C) hyperparameter sweep
# Grid: 3 brains x 4 lambda_skull x 4 lambda_tissue = 48 array jobs, full grid

#SBATCH --job-name=sg-split-sweep
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/gradsplit_%A_%a.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/gradsplit_%A_%a.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-47%20

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
#  lambda_skull  ∈ {1.0, 1.3, 1.6, 2.0}  (4 values)
#  lambda_tissue ∈ {0.3, 0.8, 1.3, 1.6}  (4 values)
#  Skip: lambda_tissue >= lambda_skull
########################################################
BRAINS=(vp_177746 vp_268749 vp_580347)
LAMBDA_SKULLS=(1.0 1.3 1.6 2.0)
LAMBDA_TISSUES=(0.3 0.8 1.3 1.6)

N_BRAINS=${#BRAINS[@]}
N_SKULL=${#LAMBDA_SKULLS[@]}
N_TISSUE=${#LAMBDA_TISSUES[@]}
N_PER_BRAIN=$(( N_SKULL * N_TISSUE ))

BRAIN_IDX=$(( SLURM_ARRAY_TASK_ID / N_PER_BRAIN ))
REM=$(( SLURM_ARRAY_TASK_ID % N_PER_BRAIN ))
SKULL_IDX=$(( REM / N_TISSUE ))
TISSUE_IDX=$(( REM % N_TISSUE ))

LAMBDA_SKULL=${LAMBDA_SKULLS[$SKULL_IDX]}
LAMBDA_TISSUE=${LAMBDA_TISSUES[$TISSUE_IDX]}
MODEL=${BRAINS[$BRAIN_IDX]}

echo "=== Task ${SLURM_ARRAY_TASK_ID}: brain=${MODEL}  lambda_skull=${LAMBDA_SKULL}  lambda_tissue=${LAMBDA_TISSUE} ==="

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
    SWEEP_LAMBDA_SKULL=$LAMBDA_SKULL \
    SWEEP_LAMBDA_TISSUE=$LAMBDA_TISSUE \
    SWEEP_WARMSTART_ITERS=8 \
    SWEEP_OUT_DIR=/cluster/scratch/fscharitzer/inversion/gradsplit_sweep \
    mrun --local -n 1 -nw 2 -nth 64 \
        python $SCRIPTS/fwi_sweep_gradsplit.py)

EXIT_CODE=$?
rm -rf $TMPDIR

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: sweep point failed (brain=${MODEL} ls=${LAMBDA_SKULL} lt=${LAMBDA_TISSUE})"
    exit 1
fi

echo "[task done] ${MODEL}  ls=${LAMBDA_SKULL}  lt=${LAMBDA_TISSUE}"
exit 0
