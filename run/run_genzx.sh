#!/bin/bash
#SBATCH --job-name=genzx_train
#SBATCH --output=logs/genzx_%j.out
#SBATCH --error=logs/genzx_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
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
export CUDA_VISIBLE_DEVICES=1

# Tmp handling
mkdir -p .tmp
export TMPDIR=$(pwd)/.tmp

# Run GenZX Training (with GflowNet Loss)
PYTHONPATH=. python -u GenZX/train.py
