"""
Comprehensive validation script based on the ZXNet paper evaluation methodology.

This script implements the validation approach described in:
"ZXNet: ZX Calculus-Driven Graph Neural Network Framework for Quantum Circuit Equivalence Checking"

Evaluation includes:
1. Accuracy, Precision, Recall, F1 metrics
2. Error detection (adding/removing CX gates)
3. Uncertainty analysis
4. Runtime and scalability analysis
5. Per-qubit verification time
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import os
import sys
import time
from collections import defaultdict
from tqdm import tqdm

# Apply DGL compatibility patch before any DGL imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import dgl_patch
except ImportError:
    pass

# --- IMPORTS ---
try:
    from encoder.encoder import FidelityEncoder
    from encoder.dataset import EncoderDataset, collate_quantum_graphs
    from mpo.circuit_utils import get_universal_gate_set
    from qiskit import QuantumCircuit, qasm2
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Ensure 'encoder/encoder.py' and 'encoder/dataset.py' exist.")
    exit(1)


# ==========================================
# 1. Classification Metrics (Accuracy, Precision, Recall, F1)
# ==========================================
def compute_classification_metrics(model, loader, device, threshold=0.5):
    """
    Compute accuracy, precision, recall, and F1 score for binary classification.
    
    Args:
        model: Trained FidelityEncoder model
        loader: DataLoader with circuit pairs
        device: torch device
        threshold: Similarity threshold for equivalence classification
        
    Returns:
        dict: Metrics dictionary
    """
    print("\n--- Computing Classification Metrics ---")
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_similarities = []
    
    with torch.no_grad():
        for batch_idx, (batch_A, batch_B) in enumerate(loader):
            if batch_idx > 50:  # Use more batches for stable metrics
                break
            
            g1, t1, l1 = batch_A
            g2, t2, l2 = batch_B
            
            g1, t1, l1 = g1.to(device), t1.to(device), l1.to(device)
            g2, t2, l2 = g2.to(device), t2.to(device), l2.to(device)
            
            # Get embeddings
            z1 = model(g1, t1, l1)
            z2 = model(g2, t2, l2)
            
            # Normalize embeddings
            z1_norm = F.normalize(z1, dim=1)
            z2_norm = F.normalize(z2, dim=1)
            
            # Compute cosine similarity
            similarities = F.cosine_similarity(z1_norm, z2_norm, dim=1)
            all_similarities.extend(similarities.cpu().numpy())
            
            # For positive pairs (equivalent circuits), label = 1
            # In the dataset, circuit_1 and circuit_2 are equivalent pairs
            labels = torch.ones(similarities.shape[0])
            all_labels.extend(labels.numpy())
            
            # Predictions based on threshold
            predictions = (similarities >= threshold).cpu().numpy()
            all_predictions.extend(predictions)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Compute ROC AUC if we have enough variation
    try:
        roc_auc = roc_auc_score(all_labels, all_similarities)
    except ValueError:
        roc_auc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_similarity': np.mean(all_similarities),
        'threshold': threshold
    }
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"Avg Similarity: {np.mean(all_similarities):.4f}")
    
    return metrics, all_similarities, all_labels


# ==========================================
# 2. Error Detection (Adding/Removing CX Gates)
# ==========================================
def test_error_detection(model, device, num_tests=50):
    """
    Test error detection by adding/removing CX gates from circuits.
    
    Based on paper Table II: Detection of Induced Error
    Tests with: Add/Remove 1, 5, 10 CX gates
    
    Args:
        model: Trained FidelityEncoder model
        device: torch device
        num_tests: Number of test circuits to generate
        
    Returns:
        dict: Error detection accuracy for different error types
    """
    print("\n--- Testing Error Detection ---")
    model.eval()
    
    from mpo.circuit_utils import create_random_circuit_with_universal_gates
    from encoder.utils import qasm_to_dgl
    import random
    
    results = {
        'add_1_cx': [],
        'remove_1_cx': [],
        'add_5_cx': [],
        'remove_5_cx': [],
        'add_10_cx': [],
        'remove_10_cx': []
    }
    
    def add_cx_gates(qc, num_gates):
        """Add random CX gates to circuit"""
        modified = qc.copy()
        num_qubits = modified.num_qubits
        for _ in range(num_gates):
            if num_qubits >= 2:
                ctrl = random.randint(0, num_qubits - 2)
                tgt = random.randint(ctrl + 1, num_qubits - 1)
                modified.cx(ctrl, tgt)
        return modified
    
    def remove_cx_gates(qc, num_gates):
        """Remove random CX gates from circuit"""
        modified = qc.copy()
        cx_indices = [i for i, inst in enumerate(modified.data) 
                     if inst.operation.name == 'cx']
        if len(cx_indices) >= num_gates:
            indices_to_remove = random.sample(cx_indices, num_gates)
            new_data = [inst for i, inst in enumerate(modified.data) 
                       if i not in indices_to_remove]
            modified.data = new_data
        return modified
    
    def check_equivalence(orig_qc, mod_qc, threshold=0.8):
        """Check if circuits are equivalent using model"""
        try:
            orig_qasm = qasm2.dumps(orig_qc)
            mod_qasm = qasm2.dumps(mod_qc)
            
            g1, t1, l1 = qasm_to_dgl(orig_qasm)
            g2, t2, l2 = qasm_to_dgl(mod_qasm)
            
            g1, t1, l1 = g1.to(device), t1.to(device), l1.to(device)
            g2, t2, l2 = g2.to(device), t2.to(device), l2.to(device)
            
            with torch.no_grad():
                z1 = model(g1, t1, l1)
                z2 = model(g2, t2, l2)
                
                z1_norm = F.normalize(z1, dim=1)
                z2_norm = F.normalize(z2, dim=1)
                similarity = F.cosine_similarity(z1_norm, z2_norm, dim=1).item()
            
            # Non-equivalent if similarity < threshold
            return similarity < threshold
        except Exception as e:
            print(f"Error in equivalence check: {e}")
            return False
    
    print(f"Generating {num_tests} test circuits...")
    for i in tqdm(range(num_tests), desc="Error Detection Tests"):
        # Generate base circuit
        num_qubits = random.randint(4, 8)
        depth = random.randint(10, 20)
        orig_qc = create_random_circuit_with_universal_gates(num_qubits, depth)
        
        # Test different error types
        error_types = [
            ('add_1_cx', lambda qc: add_cx_gates(qc, 1)),
            ('remove_1_cx', lambda qc: remove_cx_gates(qc, 1)),
            ('add_5_cx', lambda qc: add_cx_gates(qc, 5)),
            ('remove_5_cx', lambda qc: remove_cx_gates(qc, 5)),
            ('add_10_cx', lambda qc: add_cx_gates(qc, 10)),
            ('remove_10_cx', lambda qc: remove_cx_gates(qc, 10)),
        ]
        
        for error_name, error_func in error_types:
            try:
                mod_qc = error_func(orig_qc)
                detected = check_equivalence(orig_qc, mod_qc)
                results[error_name].append(1 if detected else 0)
            except Exception:
                continue
    
    # Compute accuracies
    accuracies = {}
    for error_name, detections in results.items():
        if len(detections) > 0:
            acc = np.mean(detections) * 100
            accuracies[error_name] = acc
            print(f"{error_name.replace('_', ' ').title()}: {acc:.2f}% ({len(detections)} tests)")
        else:
            accuracies[error_name] = 0.0
            print(f"{error_name.replace('_', ' ').title()}: No valid tests")
    
    return accuracies


# ==========================================
# 3. Uncertainty Analysis
# ==========================================
def analyze_uncertainty(model, loader, device):
    """
    Analyze uncertainty distribution for correctly classified vs misclassified instances.
    
    Based on paper Figure 4b: Uncertainty distribution
    
    Args:
        model: Trained FidelityEncoder model
        loader: DataLoader with circuit pairs
        device: torch device
        
    Returns:
        dict: Uncertainty statistics
    """
    print("\n--- Analyzing Uncertainty Distribution ---")
    model.eval()
    
    correct_uncertainties = []
    incorrect_uncertainties = []
    threshold = 0.8
    
    with torch.no_grad():
        for batch_idx, (batch_A, batch_B) in enumerate(loader):
            if batch_idx > 30:
                break
            
            g1, t1, l1 = batch_A
            g2, t2, l2 = batch_B
            
            g1, t1, l1 = g1.to(device), t1.to(device), l1.to(device)
            g2, t2, l2 = g2.to(device), t2.to(device), l2.to(device)
            
            z1 = model(g1, t1, l1)
            z2 = model(g2, t2, l2)
            
            z1_norm = F.normalize(z1, dim=1)
            z2_norm = F.normalize(z2, dim=1)
            similarities = F.cosine_similarity(z1_norm, z2_norm, dim=1)
            
            # Negative uncertainty: lower (more negative) = higher confidence
            # uncertainty = -(similarity - threshold)
            uncertainties = -(similarities - threshold).cpu().numpy()
            
            # Classify as correct (similarity >= threshold) or incorrect
            predictions = (similarities >= threshold).cpu().numpy()
            
            for i, (unc, pred) in enumerate(zip(uncertainties, predictions)):
                if pred:  # Correctly classified as equivalent
                    correct_uncertainties.append(unc)
                else:  # Misclassified
                    incorrect_uncertainties.append(unc)
    
    stats = {
        'correct_mean': np.mean(correct_uncertainties) if correct_uncertainties else 0,
        'correct_std': np.std(correct_uncertainties) if correct_uncertainties else 0,
        'incorrect_mean': np.mean(incorrect_uncertainties) if incorrect_uncertainties else 0,
        'incorrect_std': np.std(incorrect_uncertainties) if incorrect_uncertainties else 0,
        'correct_count': len(correct_uncertainties),
        'incorrect_count': len(incorrect_uncertainties)
    }
    
    print(f"Correctly Classified: {stats['correct_count']} samples")
    print(f"  Mean uncertainty: {stats['correct_mean']:.4f}")
    print(f"  Std uncertainty:  {stats['correct_std']:.4f}")
    print(f"Misclassified: {stats['incorrect_count']} samples")
    print(f"  Mean uncertainty: {stats['incorrect_mean']:.4f}")
    print(f"  Std uncertainty:  {stats['incorrect_std']:.4f}")
    
    # Plot uncertainty distribution
    if correct_uncertainties or incorrect_uncertainties:
        plt.figure(figsize=(10, 6))
        if correct_uncertainties:
            plt.hist(correct_uncertainties, bins=50, alpha=0.7, label='Correctly Classified', 
                    color='green', density=True)
        if incorrect_uncertainties:
            plt.hist(incorrect_uncertainties, bins=50, alpha=0.7, label='Misclassified', 
                    color='red', density=True)
        plt.xlabel('Negative Uncertainty Scores')
        plt.ylabel('Density')
        plt.title('Uncertainty Distribution: Correctly Classified vs Misclassified')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('uncertainty_distribution.png', dpi=150)
        print("Saved uncertainty distribution plot to uncertainty_distribution.png")
    
    return stats, correct_uncertainties, incorrect_uncertainties


# ==========================================
# 4. Runtime and Scalability Analysis
# ==========================================
def analyze_runtime_scalability(model, device, qubit_range=(4, 20), num_circuits=10):
    """
    Analyze runtime and scalability across different qubit counts.
    
    Based on paper Figure 5: Scalability comparison
    
    Args:
        model: Trained FidelityEncoder model
        device: torch device
        qubit_range: Tuple of (min_qubits, max_qubits)
        num_circuits: Number of circuits to test per qubit count
        
    Returns:
        dict: Runtime statistics by qubit count
    """
    print("\n--- Analyzing Runtime and Scalability ---")
    model.eval()
    
    from mpo.circuit_utils import create_random_circuit_with_universal_gates
    from encoder.utils import qasm_to_dgl
    
    results = defaultdict(list)
    
    min_qubits, max_qubits = qubit_range
    
    for num_qubits in range(min_qubits, max_qubits + 1, 2):
        print(f"Testing {num_qubits} qubits...")
        times = []
        
        for _ in range(num_circuits):
            try:
                # Generate test circuit
                qc = create_random_circuit_with_universal_gates(num_qubits, depth=15)
                qasm = qasm2.dumps(qc)
                
                # Convert to graph
                g, t, l = qasm_to_dgl(qasm)
                g, t, l = g.to(device), t.to(device), l.to(device)
                
                # Time the inference
                start_time = time.time()
                with torch.no_grad():
                    z = model(g, t, l)
                end_time = time.time()
                
                inference_time = end_time - start_time
                times.append(inference_time)
                
            except Exception as e:
                print(f"  Error with {num_qubits} qubits: {e}")
                continue
        
        if times:
            avg_time = np.mean(times)
            per_qubit_time = avg_time / num_qubits
            results[num_qubits] = {
                'avg_time': avg_time,
                'per_qubit_time': per_qubit_time,
                'std_time': np.std(times)
            }
            print(f"  {num_qubits} qubits: {avg_time*1000:.2f}ms total, {per_qubit_time*1000:.2f}ms per qubit")
    
    # Plot scalability
    if results:
        qubits = sorted(results.keys())
        total_times = [results[q]['avg_time'] * 1000 for q in qubits]
        per_qubit_times = [results[q]['per_qubit_time'] * 1000 for q in qubits]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(qubits, total_times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Qubits')
        ax1.set_ylabel('Runtime (ms)')
        ax1.set_title('Total Runtime vs Number of Qubits')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(qubits, per_qubit_times, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('Runtime per Qubit (ms)')
        ax2.set_title('Per-Qubit Verification Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scalability_analysis.png', dpi=150)
        print("Saved scalability analysis plot to scalability_analysis.png")
    
    return dict(results)


# ==========================================
# 5. Training Progress Visualization
# ==========================================
def plot_training_metrics(metrics_history=None):
    """
    Plot accuracy, precision, and recall over epochs (if history available).
    
    Based on paper Figure 4a: Accuracy, Precision, and Recall values
    """
    if metrics_history is None:
        print("No training history provided. Skipping training metrics plot.")
        return
    
    # This would require training history - placeholder for now
    print("Training metrics plot would be generated from training history")


# ==========================================
# 6. Main Validation Function
# ==========================================
def run_comprehensive_validation(model_path="fidelity_encoder_ep9.pth", 
                                data_path="data/train.pt",
                                device=None):
    """
    Run comprehensive validation based on paper methodology.
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to dataset
        device: torch device (auto-detected if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("ZXNet-Style Comprehensive Validation")
    print("Based on: ZXNet Paper Evaluation Methodology")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading Dataset from {data_path}...")
    dataset = EncoderDataset(file_path=data_path, size=200, verbose=True)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_quantum_graphs, shuffle=True)
    
    # Initialize model
    print("\nInitializing Model...")
    num_gate_types = len(get_universal_gate_set()['all'])
    model = FidelityEncoder(num_gate_types=num_gate_types, max_qubits=20).to(device)
    
    # Load weights
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model file {model_path} not found! Using random weights.")
    
    # Run all validation tests
    all_results = {}
    
    # 1. Classification Metrics
    metrics, similarities, labels = compute_classification_metrics(model, loader, device)
    all_results['classification'] = metrics
    
    # 2. Error Detection
    error_detection = test_error_detection(model, device, num_tests=30)
    all_results['error_detection'] = error_detection
    
    # 3. Uncertainty Analysis
    uncertainty_stats, correct_unc, incorrect_unc = analyze_uncertainty(model, loader, device)
    all_results['uncertainty'] = uncertainty_stats
    
    # 4. Runtime and Scalability
    scalability = analyze_runtime_scalability(model, device, qubit_range=(4, 16), num_circuits=5)
    all_results['scalability'] = scalability
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
    
    print(f"\nError Detection (Average):")
    avg_error_detection = np.mean(list(error_detection.values()))
    print(f"  Average Detection Accuracy: {avg_error_detection:.2f}%")
    
    print(f"\nScalability:")
    if scalability:
        max_qubits = max(scalability.keys())
        print(f"  Tested up to {max_qubits} qubits")
        avg_per_qubit = np.mean([s['per_qubit_time'] for s in scalability.values()])
        print(f"  Average per-qubit time: {avg_per_qubit*1000:.2f}ms")
    
    print("\n" + "=" * 70)
    print("Validation complete! Check generated plots:")
    print("  - uncertainty_distribution.png")
    print("  - scalability_analysis.png")
    print("=" * 70)
    
    return all_results


# ==========================================
# 7. Main Execution
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive validation based on ZXNet paper")
    parser.add_argument("--model", type=str, default="fidelity_encoder_ep9.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/train.pt",
                       help="Path to dataset")
    
    args = parser.parse_args()
    
    run_comprehensive_validation(model_path=args.model, data_path=args.data)

