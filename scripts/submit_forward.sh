#!/bin/bash -l
# Submit one forward-modeling job per brain.
# Read brain IDs from brains.txt 

REPO=/cluster/home/fscharitzer/semester-project/diffusion-fwi-2d-clean
BRAIN_FILE=$REPO/rest_brain.txt
SCRIPT_DIR=$REPO/scripts

if [ ! -f "$BRAIN_FILE" ]; then
    echo "ERROR: Brain list file not found: $BRAIN_FILE"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while IFS= read -r BRAIN || [ -n "$BRAIN" ]; do
    # Skip blank lines and comments
    [[ -z "$BRAIN" || "$BRAIN" == \#* ]] && continue
 
    sbatch \
        --job-name="fwd-${BRAIN}" \
        --export=ALL,BRAIN="${BRAIN}" \
        "${SCRIPT_DIR}/run_forward.sh"
    echo "Submitted forward job for brain: ${BRAIN}"
done < "$BRAIN_FILE"