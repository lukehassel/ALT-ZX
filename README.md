# ALT-ZX

> [!IMPORTANT]
> Load the required modules (`module load Python/3.12.3`) before activating the virtual environment to avoid segmentation faults.

Generative Quantum Circuit Design with ZX-Calculus

## Overview

This project generates valid quantum circuits using a Variational Autoencoder (VAE) trained on ZX-Calculus graphs. The generated graphs possess the gflow property, which guarantees circuit extraction.

## Components

### GenZX (Generative Model)

Autoencoder that reconstructs ZX-graphs. Uses GflowEncoder embeddings as a differentiable validity signal.

### ZXNet (Fidelity Predictor)

Siamese GNN that predicts semantic similarity between two ZX-graphs. Used as a frozen semantic loss in GenZX.

### GflowEncoder (Validity Scorer)

Metric-learned graph encoder that maps ZX-graphs to embeddings. Valid graphs cluster near a centroid, providing continuous validity scores.

### ZXRepair (Graph Restoration)

Encoder-decoder model that repairs corrupted ZX-graphs back to valid form. Post-processes GenZX outputs.

## Node Features

Each ZX-graph node uses a 6-dimensional feature vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | VertexID | Node index |
| 1 | NodeType | 0=Boundary, 1=Z spider, 2=X spider |
| 2 | Row | Temporal layer position |
| 3 | Degree | Connected edge count |
| 4 | Phase | Spider phase (multiples of π) |
| 5 | Qubit | Qubit wire index |

See `zx_loader.py::pyzx_graph_to_pyg()` for implementation.
