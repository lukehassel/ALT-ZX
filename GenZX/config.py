import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZXNET_CHECKPOINT = os.path.join(PROJECT_ROOT, 'ZXNet', 'model.pth')
DATASET_PATH = os.path.join(PROJECT_ROOT, 'combined', 'dataset.pt')
GENZX_CHECKPOINT = os.path.join(PROJECT_ROOT, 'GenZX', 'genzx_model.pth')

# Features: [VertexID, NodeType, Row, Degree, Phase, Qubit]
NUM_NODE_FEATURES = 6
GCN_HIDDEN_DIM = 64
LATENT_DIM = 32
MAX_NODES = 64

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 100
VAL_SPLIT = 0.2

LAMBDA_KL = 0.1
LAMBDA_CRITIC = 1.0

PERTURB_STD = 0.1
OPT_STEPS = 10
OPT_LR = 0.01

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
