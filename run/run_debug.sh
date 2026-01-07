#!/bin/bash
# Change to project directory
cd /home/wo057552/ALT-ZX

# Environment Setup
module purge
module load GCCcore/13.3.0
module load Python/3.12.3
source venv/bin/activate
export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3

# Run sample script
PYTHONPATH=. python -u GenZX/sample.py
