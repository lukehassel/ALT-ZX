#!/bin/bash
#SBATCH --job-name=gflow_dataset
#SBATCH --output=logs/gflow_dataset_%j.out
#SBATCH --error=logs/gflow_dataset_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=c23m

# No GPU needed for dataset generation - use c23m (memory) partition

cd /home/wo057552/ALT-ZX

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source venv/bin/activate

mkdir -p .tmp
export TMPDIR=$(pwd)/.tmp

PYTHONPATH=. python -m GflowClassification.dataset
