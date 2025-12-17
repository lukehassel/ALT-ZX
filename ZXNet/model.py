import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ZXNet(nn.Module):
    def __init__(self, num_node_features):
        super(ZXNet, self).__init__()
        
        # Shared GCN Encoder
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        
        # Classification Head (concatenating two 64-dim vectors -> 128 input)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2) # Output: Logits for [Not Equivalent, Equivalent]

    def forward_one_graph(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global Mean Pooling 
        x = global_mean_pool(x, data.batch)
        return x

    def forward(self, data1, data2):
        # Siamese Architecture: Process both graphs with same weights
        emb1 = self.forward_one_graph(data1)
        emb2 = self.forward_one_graph(data2)
        
        # Combine representations (Concatenation)
        combined = torch.cat([emb1, emb2], dim=1)
        
        # Classifier
        out = self.fc1(combined)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out
    
    def compute_jaccard(self, data1, data2):
        """
        Computes Jaccard index of edge sets for a batch. 
        Note: Simplified for single-pair batches for clarity.
        """
        # Convert edge_index to set of tuples
        e1 = set(map(tuple, data1.edge_index.t().tolist()))
        e2 = set(map(tuple, data2.edge_index.t().tolist()))
        
        intersection = len(e1.intersection(e2))
        union = len(e1.union(e2))
        
        return intersection / union if union > 0 else 0.0
    
    def zxnet_loss(self, logits, labels, data1, data2, 
                uncertainty_threshold=0.5, 
                jaccard_threshold=0.6, 
                alpha=0.5, beta=1.0):
        
        # 1. Base Cross Entropy Loss [cite: 165]
        base_loss = F.cross_entropy(logits, labels)
        
        # 2. Compute Uncertainty (Eq 7) [cite: 199]
        # The paper defines delta = -log(sum(exp(logits)))
        # This is effectively negative LogSumExp.
        log_sum_exp = torch.logsumexp(logits, dim=1)
        uncertainty = -log_sum_exp 
        
        
        if uncertainty.mean() > uncertainty_threshold:
            j_score = self.compute_jaccard(data1, data2)
            
            # Determine target based on Jaccard (heuristic equivalence)
            j_prediction = 1.0 if j_score > jaccard_threshold else 0.0
            
            # Model prediction (probability of class 1)
            probs = F.softmax(logits, dim=1)
            pred_prob = probs[:, 1]
            
            # Jaccard Penalty Term
            jaccard_loss = alpha * torch.abs(pred_prob - j_prediction).mean()
            
            total_loss = base_loss + (beta * jaccard_loss)
        else:
            total_loss = base_loss
            
        return total_loss