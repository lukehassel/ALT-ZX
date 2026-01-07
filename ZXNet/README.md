# ZXNet: Siamese Fidelity Predictor

Siamese Graph Neural Network that predicts fidelity between two quantum circuits represented as ZX-graphs.

## Architecture

1. **Shared Encoder**: GCN layers process both graphs
2. **Embeddings**: Graph-level vectors E1 and E2
3. **Comparison**: Concatenates [E1, E2, |E1 - E2|]
4. **Prediction**: MLP outputs fidelity score in [0, 1]

## Files

| File | Description |
|------|-------------|
| `model.py` | ZXNet class definition |
| `train.py` | Training script |
| `evaluate.py` | Evaluation metrics |

## Dataset

Expects `combined/dataset.pt` with tuples of `(graph1, graph2, fidelity_label)`.
