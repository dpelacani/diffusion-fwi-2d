#!/bin/bash -l

# Submit one FWI inversion job per brain.
# Read brain IDs from brains.txt.
# Each job runs all (scenario, method) combinations for that brain sequentially.

#SBATCH --job-name=submission
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/scratch/fscharitzer/slurm/subm_%j.out
#SBATCH --error=/cluster/scratch/fscharitzer/slurm/subm_%j.err
# --cpus-per-task=128
# --mem-per-cpu=1G
# --dependency=afterok:3162187

BRAIN_FILE="${1:-$(dirname "${BASH_SOURCE[0]}")/eval_brains.txt}"
SCENARIO="${2:-}"   # optional: restrict to one scenario
METHOD="${3:-}"     # optional: restrict to one method

if [ ! -f "$BRAIN_FILE" ]; then
    echo "ERROR: Brain list file not found: $BRAIN_FILE"
    exit 1
fi

SCRIPT_DIR="scripts"

while IFS= read -r BRAIN || [ -n "$BRAIN" ]; do
    [[ -z "$BRAIN" || "$BRAIN" == \#* ]] && continue
 
    EXPORTS="ALL,BRAIN=${BRAIN}"
    [ -n "$SCENARIO" ] && EXPORTS="${EXPORTS},SCENARIO=${SCENARIO}"
    [ -n "$METHOD" ]   && EXPORTS="${EXPORTS},METHOD=${METHOD}"
 
    sbatch \
        --job-name="fwi-${BRAIN}" \
        --export="${EXPORTS}" \
        "${SCRIPT_DIR}/run_fwi.sh"
    echo "Submitted FWI job for brain: ${BRAIN}  scenario=${SCENARIO:-all}  method=${METHOD:-all}"
done < "$BRAIN_FILE"