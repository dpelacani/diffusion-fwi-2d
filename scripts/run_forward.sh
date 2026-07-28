#!/bin/bash -l
# Run forward modeling job for a specified brain
# Generate acquisitions .h5 file used by the FWI inversion

#SBATCH --job-name=forward
#SBATCH --time=00:30:00                   
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/forward_%j.out       
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/forward_%j.err         
#SBATCH --cpus-per-task=128                 
#SBATCH --mem-per-cpu=1G

########################################################
############# Module and Environment Setup #############
########################################################

module load stack/2025-06
module load eth_proxy

source $HOME/envs/stride/bin/activate
export TMPDIR=/cluster/scratch/fscharitzer/devito-tmp
mkdir -p $TMPDIR

########################################################
################## Forward Modelling ###################
########################################################

NAME_UPPER=$(echo "$BRAIN" | tr '[:lower:]' '[:upper:]')
FORWARD_DIR="/cluster/scratch/fscharitzer/exps/forward/${NAME_UPPER}"
ACQ_FILE="${FORWARD_DIR}/${NAME_UPPER}-Acquisitions.h5"

# Skip if acquisitions file already exists
if [ -f "$ACQ_FILE" ]; then
    echo "Acquisitions already exist for ${BRAIN}, skipping forward modeling."
    exit 0
fi

mkdir -p "$FORWARD_DIR"
echo "=== Forward modeling: BRAIN=${BRAIN} ==="
cd /cluster/home/fscharitzer/semester-project/diffusion-fwi-2d

MODEL_NAME=$BRAIN mrun --local -n 1 -nw 32 -nth 4 python paper-pipeline/forward.py

if [ $? -ne 0 ]; then
    echo "ERROR: Forward modeling failed for ${BRAIN}"
    exit 1
fi

echo "[done] Forward modeling: ${BRAIN}"
exit 0