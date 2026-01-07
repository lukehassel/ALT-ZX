# ZXRepair: Graph Restoration Model

Encoder-decoder model that repairs corrupted ZX-graphs back to valid gflow-compliant form.

## Architecture

1. **GCN Encoder**: 3-layer DenseGCN encodes corrupted structure to node embeddings
2. **Adjacency Decoder**: MLP predicts repaired adjacency matrix
3. **Feature Decoder**: MLP predicts corrected node features
4. **GflowEncoder Loss**: Optional validity regularization

## Dataset Generation

The `dataset.py` module creates training data:

1. Generate valid ZX-graphs from random circuits
2. Apply corruptions:
   - Edge removal: 10-30%
   - Edge addition: 5-15%
   - Node removal: 5-15% (non-boundary)
   - Phase noise: Gaussian

## Files

| File | Description |
|------|-------------|
| `model.py` | ZXRepairNet architecture |
| `train.py` | Training loop |
| `dataset.py` | Dataset generator |
