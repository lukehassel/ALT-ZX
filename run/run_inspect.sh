#!/bin/bash
#SBATCH --job-name=inspect_pt
#SBATCH --output=logs/inspect_%j.out
#SBATCH --error=logs/inspect_%j.err
#SBATCH --time=0:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=c23m

cd /home/wo057552/ALT-ZX

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source venv/bin/activate

echo "=== Inspecting ZXRepair Dataset ==="
echo "Start time: $(date)"

PYTHONPATH=. python inspect_pt.py ZXRepair/repair_dataset.pt

echo ""
echo "End time: $(date)"
