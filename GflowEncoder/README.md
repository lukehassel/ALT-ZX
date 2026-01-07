# GflowEncoder: Graph Embedding for ZX Validity Scoring

Graph Neural Network encoder that maps ZX-graphs to embeddings for continuous validity scoring.

## Purpose

Provides gradient signal for graph validity:
- Valid graphs cluster near a centroid (high score)
- Invalid graphs are far from centroid (low score, non-zero gradient)

## Architecture

- **GCN**: 4 layers with BatchNorm and ReLU
- **Pooling**: Global Mean + Max
- **Output**: 64-dimensional L2-normalized embedding

## Training: Triplet Loss

Contrastive learning pulls valid graphs together and pushes invalid graphs apart:

```
Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

Margin = 0.3, distance = cosine distance.

### Data Generation

- **Anchor**: Valid quantum circuit graph
- **Positive**: Same circuit with 2 valid ZX transformations
- **Negative**: Same circuit with 50 corrupting transformations

## Scoring

1. Encode graph to 64-dim embedding
2. Compute cosine similarity to centroid
3. Score = (similarity + 1) / 2

## Files

| File | Description |
|------|-------------|
| `model.py` | GraphEncoder class |
| `train.py` | Contrastive training |
| `encoder.pth` | Trained weights |
| `valid_centroid.pt` | Centroid embedding |
