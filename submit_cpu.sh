#!/bin/bash

#SBATCH --job-name=rad51protein
#SBATCH --mem 8GB
#SBATCH --account=histoneclf
#SBATCH --partition=long
#SBATCH --cpus-per-task=8

python3 imageAugmentation.py

echo "Job submit done"
