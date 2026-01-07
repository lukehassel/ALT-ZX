# GenZX: Generative Model for ZX-Graphs

Graph Variational Autoencoder that generates valid, optimizable ZX-diagrams.

## Architecture

- **Encoder**: GNN maps graph to latent space z
- **Decoder**: Reconstructs adjacency matrix and node features from z
- **Gflow Loss**: Uses frozen GflowEncoder for validity scoring
- **Semantic Loss**: Uses frozen ZXNet to enforce semantic similarity
- **Boundary Loss**: Ensures correct input/output node degrees

## Files

| File | Description |
|------|-------------|
| `model.py` | GraphVAE implementation |
| `train.py` | Training loop |
| `sample.py` | Sampling from latent space |
| `layers.py` | Custom GNN layers |
| `data_loader.py` | Dataset loading utilities |
| `dataset.py` | Dataset generator |
