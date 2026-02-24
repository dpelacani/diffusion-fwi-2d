#!/usr/bin/env bash
#SBATCH --job-name=jobname
#SBATCH --partition=root
#SBATCH -e slurm-%j.err              # File to redirect stderr
#SBATCH -o slurm-%j.out              # File to redirect stdout

#SBATCH --qos=epic
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00

# now run your program
srun nvidia-smi

# Get the number of cores
CORES=$(nproc)

# Launch your process on each core
for ((i=1; i<=$CORES; i++)); do
    # Replace the command below with your actual process
    (echo "Process $i started"; sleep 30; echo "Process $i finished") &
done
wait