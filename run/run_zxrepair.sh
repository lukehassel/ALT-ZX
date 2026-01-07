#!/bin/bash
#SBATCH --job-name=zxrepair_train
#SBATCH --output=logs/zxrepair_%j.out
#SBATCH --error=logs/zxrepair_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g

cd /home/wo057552/ALT-ZX

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source venv/bin/activate

export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3

mkdir -p .tmp
export TMPDIR=$(pwd)/.tmp

echo "=== ZXRepair Training ==="
echo "Start time: $(date)"

# Generate dataset if it doesn't exist
if [ ! -f "ZXRepair/repair_dataset.pt" ]; then
    echo ""
    echo "=== Generating Dataset (50000 samples) ==="
    PYTHONPATH=. python ZXRepair/dataset.py --num_samples 50000 --workers 6
fi

# Train the model
echo ""
echo "=== Training ZXRepairNet ==="
PYTHONPATH=. python -u ZXRepair/train.py --epochs 100 --batch_size 32

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
