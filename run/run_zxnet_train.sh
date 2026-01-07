#!/bin/bash
#SBATCH --job-name=zxnet_train
#SBATCH --output=logs/zxnet_train_%j.out
#SBATCH --error=logs/zxnet_train_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

echo "=== ZXNet Training ==="
PYTHONPATH=. python ZXNet/train.py

echo ""
echo "=== Training Complete ==="
echo "Model saved to ZXNet/model.pth"
