#!/bin/bash

#SBATCH --job-name=rad51protein
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=histoneclf
#SBATCH --partition=fast
#SBATCH --cpus-per-task=16

python3 time.py

echo "Job submit done"
