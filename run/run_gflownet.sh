#!/bin/bash
#SBATCH --job-name=gflow_train
#SBATCH --output=logs/gflownet_%j.out
#SBATCH --error=logs/gflownet_%j.err
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
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

# Tmp handling
mkdir -p .tmp
export TMPDIR=$(pwd)/.tmp

# Run GflowNet Training
PYTHONPATH=. python -u GflowNet/train.py
