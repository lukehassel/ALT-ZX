#!/bin/bash
#SBATCH --job-name=genzx_sample
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g

# Change to project directory
cd /home/wo057552/ALT-ZX

# Environment Setup
module purge
module load GCCcore/13.3.0
module load Python/3.12.3
source venv/bin/activate
export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3

# Run GenZX Sampling
PYTHONPATH=. python -u GenZX/sample.py
