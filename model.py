
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class ZXSemanticDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.type_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)
        )
        self.phase_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.type_head(z), self.phase_head(z)


class ZXVGAE(VGAE):
    def __init__(self, in_channels, hidden_channels, out_channels):
        encoder = VariationalGCNEncoder(in_channels, hidden_channels, out_channels)
        super().__init__(encoder)
        self.semantic_decoder = ZXSemanticDecoder(out_channels)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        edge_adj_recon = self.decode(z, edge_index)
        node_type_logits, phase_pred = self.semantic_decoder(z)
        return z, edge_adj_recon, node_type_logits, phase_pred

    def loss(self, x, edge_index, pos_edge_label_index, neg_edge_label_index=None):
        z, _, node_type_logits, phase_pred = self.forward(x, edge_index)
        loss_structure = self.recon_loss(z, pos_edge_label_index, neg_edge_label_index)
        loss_kl = (1 / x.size(0)) * self.kl_loss()
        target_types = x[:, :5].argmax(dim=1)
        loss_types = F.cross_entropy(node_type_logits, target_types)
        target_phases = x[:, 5:6]
        loss_phases = F.mse_loss(phase_pred, target_phases)
        total_loss = loss_structure + loss_kl + loss_types + loss_phases
        return total_loss