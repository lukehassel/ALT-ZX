#!/bin/bash
cd /home/wo057552/ALT-ZX

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source venv/bin/activate

python -c "
import torch
import glob

chunk_patterns = ['GflowEncoder/dataset_chunk_*.pt', 'GflowEncoder/dataset_new_chunk_*.pt']
chunks = []
for p in chunk_patterns:
    chunks.extend(glob.glob(p))
chunks = sorted(chunks)
print(f'Found {len(chunks)} chunks')

combined = []
for c in chunks:
    print(f'Loading {c}...')
    try:
        data = torch.load(c, weights_only=False)
        combined.extend(data)
        print(f'  Added {len(data)} samples, total: {len(combined)}')
    except Exception as e:
        print(f'  SKIPPED (corrupt): {e}')

torch.save(combined, 'GflowEncoder/dataset.pt')
print(f'Saved combined dataset with {len(combined)} samples to GflowEncoder/dataset.pt')
"
