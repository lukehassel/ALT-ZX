#!/bin/bash
#SBATCH --job-name=test_gflow
#SBATCH --output=test_gflow.out
#SBATCH --error=test_gflow.err
#SBATCH --time=0:05:00
#SBATCH --partition=c23g

cd /home/wo057552/ALT-ZX

# Environment setup
module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source venv/bin/activate

python GflowEncoder/tests/test_dataset_integrations.py
