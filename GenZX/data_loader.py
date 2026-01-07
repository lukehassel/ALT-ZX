import pickle
import torch
from torch.utils.data import Dataset


def load_dataset(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return torch.load(filepath, weights_only=False)


class GraphVAEDataset(Dataset):
    def __init__(self, graphs, max_nodes, num_node_features=6):
        self.max_nodes = max_nodes
        self.num_node_features = num_node_features
        
        print(f"Pre-caching {len(graphs)} graphs...")
        self.cached_features = []
        self.cached_adj = []
        self.cached_node_features = []
        
        for g in graphs:
            features, adj, node_features = self._process_graph(g)
            self.cached_features.append(features)
            self.cached_adj.append(adj)
            self.cached_node_features.append(node_features)
        
        self.features_tensor = torch.stack(self.cached_features)
        self.adj_tensor = torch.stack(self.cached_adj)
        self.node_features_tensor = torch.stack(self.cached_node_features)
        
        del self.cached_features, self.cached_adj, self.cached_node_features
        
        if torch.cuda.is_available():
            print("Moving dataset to GPU...")
            self.features_tensor = self.features_tensor.cuda()
            self.adj_tensor = self.adj_tensor.cuda()
            self.node_features_tensor = self.node_features_tensor.cuda()
        
        print(f"Cached tensors: features {self.features_tensor.shape}, adj {self.adj_tensor.shape}")
    
    def _process_graph(self, g):
        if hasattr(g, 'x') and g.x is not None:
            n_nodes = min(g.x.shape[0], self.max_nodes)
            node_feat_dim = g.x.shape[1]
        else:
            n_nodes = min(g.num_nodes, self.max_nodes)
            node_feat_dim = self.num_node_features
        
        node_features = torch.zeros(self.max_nodes, self.num_node_features)
        if hasattr(g, 'x') and g.x is not None:
            feat_to_copy = min(node_feat_dim, self.num_node_features)
            node_features[:n_nodes, :feat_to_copy] = g.x[:n_nodes, :feat_to_copy]
        
        adj = torch.zeros(self.max_nodes, self.max_nodes)
        if hasattr(g, 'edge_index') and g.edge_index.numel() > 0:
            edges = g.edge_index
            mask = (edges[0] < self.max_nodes) & (edges[1] < self.max_nodes)
            valid_edges = edges[:, mask]
            if valid_edges.numel() > 0:
                adj[valid_edges[0], valid_edges[1]] = 1.0
                adj[valid_edges[1], valid_edges[0]] = 1.0
        
        adj.fill_diagonal_(1.0)
        
        features = adj.clone()
        
        return features, adj, node_features
    
    def __len__(self):
        return self.features_tensor.shape[0]
    
    def __getitem__(self, idx):
        return {
            'features': self.features_tensor[idx],
            'adj': self.adj_tensor[idx],
            'node_features': self.node_features_tensor[idx],
        }
