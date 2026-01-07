#!/bin/bash
#SBATCH --job-name=train_gflow_encoder
#SBATCH --output=logs/train_gflow_encoder_%j.out
#SBATCH --error=logs/train_gflow_encoder_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Change to project directory
cd /home/wo057552/ALT-ZX

# Purge and load modules
module purge
module load GCCcore/13.3.0
module load Python/3.12.3

# Activate virtual environment
source venv/bin/activate

# Set CUDA_HOME (hardcoded path from cluster config)
export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3

# Set TMPDIR
mkdir -p .tmp
export TMPDIR=$(pwd)/.tmp

echo "Starting GflowEncoder Training..."
echo "Date: $(date)"

# Run training
export PYTHONPATH=$PYTHONPATH:.
python -m GflowEncoder.train

echo "Training finished at: $(date)"
