#!/bin/bash -l
# run_fwi_array.sh
# One array task = one brain.
# Runs forward modeling (if needed), then all 3 experiments for $METHOD.

#SBATCH --job-name=fwi-array
#SBATCH --gpus=2
# --gres=gpumem:24g
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/fwi_%A_%a.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/fwi_%A_%a.err

########################################################
# Environment
########################################################
module load stack/2025-06
module load nvhpc/24.9
module load eth_proxy

export DEVITO_ARCH=gcc
export DEVITO_LANGUAGE=C
export TMPDIR=/cluster/scratch/fscharitzer/devito-tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
JOB_OFFSET=$(( SLURM_JOB_ID % 50 ))

BASE_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
export MOSAIC_PORT=$(( 20000 + (BASE_ID % 100) * 400 + (SLURM_ARRAY_TASK_ID % 100) * 20 ))

mkdir -p $TMPDIR
cleanup() {
    echo "Cleaning up local node scratch space..."
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/home/fscharitzer/envs/stride

REPO=/cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
SCRIPTS=$REPO/paper-pipeline
BRAIN_LIST=$REPO/fwi_brains.txt

########################################################
# Brain selection
########################################################
BRAIN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BRAIN_LIST" | tr -d '[:space:]')

if [ -z "$BRAIN" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: index out of range, skipping."
    exit 0
fi

echo "=========================================="
echo " Task:   ${SLURM_ARRAY_TASK_ID}"
echo " Brain:  ${BRAIN}"
echo " Method: ${METHOD}"
echo " Port:   ${MOSAIC_PORT}"
echo "=========================================="

########################################################
# Forward modeling (once per brain, shared across methods)
########################################################
NAME_UPPER=$(echo "$BRAIN" | tr '[:lower:]' '[:upper:]')
FORWARD_DIR="/cluster/scratch/fscharitzer/exps/forward/${NAME_UPPER}"
ACQ_FILE="${FORWARD_DIR}/${NAME_UPPER}-Acquisitions.h5"

LOCK_DIR=/cluster/scratch/fscharitzer/locks
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/forward_${NAME_UPPER}.lock"

(
    flock -x 200

    if [ -f "$ACQ_FILE" ]; then
        echo "Acquisitions exist for ${BRAIN}, skipping forward modeling."
        exit 0
    fi

    echo "--- Forward modeling: ${BRAIN} ---"
    mkdir -p "$FORWARD_DIR"

    cd $TMPDIR && MODEL_NAME=$BRAIN \
        mrun --local -n 1 -nw 32 -nth 4 --port $MOSAIC_PORT \
        python $SCRIPTS/forward.py
    exit $?
) 200>"$LOCK_FILE"
FWD_STATUS=$?

if [ $FWD_STATUS -ne 0 ]; then
    echo "ERROR: Forward modeling failed for ${BRAIN}"
    rm -rf $TMPDIR
    exit 1
fi
echo "[checked] Forward modeling ready for ${BRAIN}"

########################################################
# FWI inversion: all 3 scenarios for this method
########################################################
(cd $TMPDIR && BRAIN=$BRAIN METHOD=$METHOD \
    mrun --local -n 1 -nw 2 -nth 64 --port $MOSAIC_PORT \
    python $SCRIPTS/fwi.py)

if [ $? -ne 0 ]; then
    echo "ERROR: FWI inversion failed for ${BRAIN} method=${METHOD}"
    rm -rf $TMPDIR
    exit 1
fi

rm -rf $TMPDIR
echo "[all done] ${BRAIN}  method=${METHOD}"
exit 0