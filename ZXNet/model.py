import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DenseGCNConv, global_mean_pool
from torch_geometric.data import Batch, Data


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class ZXNet(nn.Module):
    def __init__(self, num_node_features):
        super(ZXNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        
        self.conv1_dense = DenseGCNConv(num_node_features, 64)
        self.conv2_dense = DenseGCNConv(64, 64)
        
        # 3 vectors concatenated: u, v, |u-v| -> 64*3 = 192
        self.fc1 = nn.Linear(192, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self._dense_weights_synced = False

    def _sync_dense_weights(self):
        if not self._dense_weights_synced:
            self.conv1_dense.lin.weight.data = self.conv1.lin.weight.data.clone()
            self.conv1_dense.bias.data = self.conv1.bias.data.clone()
            self.conv2_dense.lin.weight.data = self.conv2.lin.weight.data.clone()
            self.conv2_dense.bias.data = self.conv2.bias.data.clone()
            self._dense_weights_synced = True

    def forward_one_graph(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return x
    
    def forward_one_graph_dense(self, x, adj, mask=None):
        self._sync_dense_weights()
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        h = self.conv1_dense(x, adj, mask)
        h = F.relu(h)
        h = self.conv2_dense(h, adj, mask)
        h = F.relu(h)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        
        return h

    def forward(self, data):
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            batch_g1, batch_g2 = data[0], data[1]
        else:
            raise ValueError(f"Expected tuple/list of 2 Batch objects, got {type(data)}")

        emb1 = self.forward_one_graph(batch_g1.x, batch_g1.edge_index, batch_g1.batch)
        emb2 = self.forward_one_graph(batch_g2.x, batch_g2.edge_index, batch_g2.batch)
        
        combined = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)], dim=1)
        
        out = self.fc1(combined)
        out = F.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out.squeeze(-1)

    def zxnet_loss(self, predictions, targets, batch_data):
        return F.mse_loss(predictions, targets)