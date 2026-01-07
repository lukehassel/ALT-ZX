#!/bin/bash
#SBATCH --job-name=count_graphs
#SBATCH --output=logs/count_graphs_%j.out
#SBATCH --error=logs/count_graphs_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=c23g

# Change to project directory
cd /home/wo057552/ALT-ZX

# Environment Setup
module purge
module load GCCcore/13.3.0
module load Python/3.12.3
source venv/bin/activate

# Execute counting script
PYTHONPATH=. python -u count_small_graphs.py
