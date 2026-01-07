import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GflowEncoder.model import GraphEncoder, embedding_distance_score
from GflowEncoder.data_loader import integration_load_dataset


def evaluate_model(encoder_path, centroid_path, test_dataset_path="GflowEncoder/test_dataset_chunk_0.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    encoder = GraphEncoder(num_node_features=6, hidden_dim=128, embedding_dim=64)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    
    centroid = torch.load(centroid_path, map_location=device)
    
    print(f"Loading test dataset from {test_dataset_path}...")
    triplets = integration_load_dataset(test_dataset_path)
    print(f"Loaded {len(triplets)} test samples")
    
    anchor_scores = []
    pos_scores = []
    neg_scores = []
    
    print("Scoring test set...")
    with torch.no_grad():
        for anchor, pos, neg in tqdm(triplets):
            s_anchor = embedding_distance_score(encoder, anchor, centroid, device)
            s_pos = embedding_distance_score(encoder, pos, centroid, device)
            s_neg = embedding_distance_score(encoder, neg, centroid, device)
            
            anchor_scores.append(s_anchor)
            pos_scores.append(s_pos)
            neg_scores.append(s_neg)
            
    anchor_scores = np.array(anchor_scores)
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    
    print("\n--- Evaluation Results ---")
    print(f"Anchor (Valid) Score:   {anchor_scores.mean():.4f} ± {anchor_scores.std():.4f}")
    print(f"Positive (Valid) Score: {pos_scores.mean():.4f} ± {pos_scores.std():.4f}")
    print(f"Negative (Invalid) Score: {neg_scores.mean():.4f} ± {neg_scores.std():.4f}")
    
    # Accuracy: Valid > Invalid
    acc_anchor = (anchor_scores > neg_scores).mean()
    acc_pos = (pos_scores > neg_scores).mean()
    
    print(f"\nDistinction Accuracy (Valid > Invalid):")
    print(f"Anchor vs Negative:   {acc_anchor*100:.2f}%")
    print(f"Positive vs Negative: {acc_pos*100:.2f}%")
    
    # Best threshold to separate valid from invalid
    valid_scores = np.concatenate([anchor_scores, pos_scores])
    invalid_scores = neg_scores
    
    best_acc = 0
    best_thresh = 0
    
    for thresh in np.linspace(0, 1, 101):
        tp = (valid_scores >= thresh).sum()
        tn = (invalid_scores < thresh).sum()
        acc = (tp + tn) / (len(valid_scores) + len(invalid_scores))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    tp = (valid_scores >= best_thresh).sum()
    fp = (invalid_scores >= best_thresh).sum()
    fn = (valid_scores < best_thresh).sum()
    tn = (invalid_scores < best_thresh).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
    print(f"\nOptimal Classification Threshold: {best_thresh:.2f}")
    print(f"Max Classification Accuracy: {best_acc*100:.2f}%")
    print(f"\n--- Metrics at Threshold {best_thresh:.2f} ---")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%") 
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp}, FP={fp}")
    print(f"  FN={fn}, TN={tn}")
    
    return {
        'anchor_scores': anchor_scores,
        'pos_scores': pos_scores, 
        'neg_scores': neg_scores
    }


if __name__ == "__main__":
    if not os.path.exists("GflowEncoder/encoder.pth"):
        print("Error: GflowEncoder/encoder.pth not found. Train the model first.")
        sys.exit(1)
        
    evaluate_model("GflowEncoder/encoder.pth", "GflowEncoder/valid_centroid.pt")
