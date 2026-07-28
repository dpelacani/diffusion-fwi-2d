#!/bin/bash -l

# FWI inversion for a single brain, all (scenario, method) combinations.

#SBATCH --job-name=inversion
#SBATCH --gpus=2
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/fwi_%j.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/fwi_%j.err
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
# --dependency=afterok:2632769

########################################################
############# Module and Environment Setup #############
########################################################
module load stack/2025-06
module load nvhpc/24.9
module load eth_proxy
export DEVITO_ARCH=gcc
export DEVITO_LANGUAGE=C

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/home/fscharitzer/envs/stride

export TMPDIR=/cluster/scratch/fscharitzer/devito-tmp/${SLURM_JOB_ID}
export DEVITO_JITCACHE=/cluster/home/fscharitzer/devito-jitcache-gcc-C
mkdir -p $TMPDIR

SCRIPTS=/cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean/paper-pipeline
cd /cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean

JOB_OFFSET=$(( SLURM_JOB_ID % 50 ))
MOSAIC_PORT=$(( 20000 + (JOB_OFFSET * 500)))

########################################################
################## Forward (if needed) #################
########################################################
NAME_UPPER=$(echo "$BRAIN" | tr '[:lower:]' '[:upper:]')
ACQ_FILE="/cluster/scratch/fscharitzer/exps/forward/${NAME_UPPER}/${NAME_UPPER}-Acquisitions.h5"
FORWARD_DIR="/cluster/scratch/fscharitzer/exps/forward/${NAME_UPPER}"

if [ -f "$ACQ_FILE" ]; then
    echo "Acquisitions already exist for ${BRAIN}, skipping forward modeling."
else
    echo "--- Forward modeling: BRAIN=${BRAIN} ---"
    mkdir -p "$FORWARD_DIR"
 
    # Forward modeling can be run CPU-only (no NVHPC/GPU needed).
    (cd $TMPDIR && MODEL_NAME=$BRAIN mrun --local -n 1 -nw 32 -nth 4 \
        python $SCRIPTS/forward.py)
 
    if [ $? -ne 0 ]; then
        echo "ERROR: Forward modeling failed for ${BRAIN}"
        rm -rf $TMPDIR
        exit 1
    fi
    echo "[done] Forward modeling: ${BRAIN}"
fi

########################################################
#################### FWI Inversion #####################
########################################################
echo "--- FWI inversion: BRAIN=${BRAIN}  scenario=${SCENARIO:-all}  method=${METHOD:-all} ---"

ARGS="--brain ${BRAIN}"
[ -n "${SCENARIO}" ] && ARGS="${ARGS} --scenario ${SCENARIO}"
[ -n "${METHOD}" ]   && ARGS="${ARGS} --method ${METHOD}"

(cd $TMPDIR && BRAIN=$BRAIN SCENARIO=${SCENARIO:-} METHOD=${METHOD:-} \
    mrun --local -n 1 -nw 2 -nth 64 python $SCRIPTS/fwi.py)

if [ $? -ne 0 ]; then
    echo "ERROR: FWI inversion failed for ${BRAIN}"
    rm -rf $TMPDIR
    exit 1
fi

rm -rf $TMPDIR
echo "[done] FWI inversion ${BRAIN}"
exit 0