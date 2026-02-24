#!/usr/bin/env bash

#SBATCH -J pytorch
#SBATCH -p tidemaker
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --gpus 1

module load pytorch

python3 torch_example.py