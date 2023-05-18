#!/bin/bash

#SBATCH --job-name=rad51protein
#SBATCH --account=histoneclf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:7g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

module load cudatoolkit/11.6.0
module load tensorflow/2.6.2

python3 Training_main.py

echo "Job submit done"
