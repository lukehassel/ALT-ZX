#!/bin/bash
#SBATCH --job-name=dataset_generation
#SBATCH --output=logs/dataset_generation_%j.out
#SBATCH --error=logs/dataset_generation_%j.err
#SBATCH --time=7:00:00
##SBATCH --account=lect0163
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g

# Change to project directory (absolute path required for SLURM)
cd /home/wo057552/ALT-ZX

# Purge all modules to avoid Intel MKL/MPI conflicts with torch
module purge

# Load only required modules
module load GCCcore/13.3.0
module load Python/3.12.3

# Activate virtual environment
source venv/bin/activate

# Prevent user-site packages from polluting the environment
export PYTHONNOUSERSITE=1

# Set CUDA_HOME explicitly (hardcoded path from known valid location)
export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3

# Set TMPDIR to avoid /tmp permission/space issues
mkdir -p .tmp
export TMPDIR=$(pwd)/.tmp

# Exclude GPU 0 (busy) and GPU 2 (busy 8GB)
# Note: Multi-GPU (1,3) fails with NCCL Error on this node due to interconnect/topology issues.
# Reverting to Single-GPU (GPU 1) for stability.
#export CUDA_VISIBLE_DEVICES=3
# export CUDA_VISIBLE_DEVICES=1,3
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# python mpo/test_fidelity_comparison.py
# if [ $? -ne 0 ]; then
#     echo "Fidelity verification FAILED. Aborting."
#     exit 1
# fi

#python mpo/profile_20q.py

#python encoder/combine_chunks.py 

#PYTHONPATH=. python ZXNet/train.py

#PYTHONPATH=. python -m GenZX.train

# GflowNet - differentiable gflow testing
#PYTHONPATH=. python GflowNet/dataset.py
#PYTHONPATH=. python -u GflowNet/train.py
#PYTHONPATH=. python -u debug_genzx_integration.py
#PYTHONPATH=. python debug_reconstruction.py
#PYTHONPATH=. python migrate_7to6_features.py

# GenZX Training (with GflowNet Loss)
#PYTHONPATH=. python -u GenZX/train.py

# GflowClassification dataset generation
#PYTHONPATH=. python -m GflowClassification.dataset

# ZXRepair dataset generation
#PYTHONPATH=. python -m ZXRepair.dataset

# Inspect ZXRepair dataset with images (before/after corruption)
#python inspect_pt.py /work/wo057552/repair_dataset.pt --images

# Train GflowDense node-level model
#PYTHONPATH=. python -m GflowDense.train

# GflowClassification dataset generation
#PYTHONPATH=. python -m GflowClassification.dataset

# python verify_batching.py
# Run GflowEncoder training
#python -m GflowEncoder.train
#python -m simple_debug
#python encoder/dataset.py --size 500 --workers 4 --restart-count 6 --pause 20

#PYTHONPATH=. python -m GflowEncoder.train

#PYTHONPATH=. python -m GflowEncoder.compute_centroid
#PYTHONPATH=. python -m GflowEncoder.evaluate

# Generate GenZX dataset with small graphs (< 64 nodes)
#PYTHONPATH=. python -u -m GenZX.dataset --size 50000 --max-nodes 64

# Combine GenZX dataset chunks
#PYTHONPATH=. python -u GenZX/combine_chunks.py

# Run GenZX sampling
#PYTHONPATH=. python -u -m GenZX.sample

# Evaluate GenZX extractability (1000 samples)
PYTHONPATH=. python -u -m GenZX.evaluate_samples

#PYTHONPATH=. python -m GenZX.train

# Regenerate optimization dataset
#python dataset_creation/optimization_dataset.py --size 5000 --workers 8 --verbose --mode both --timeout 28
