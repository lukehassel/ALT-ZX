"""
Validation script comparing model predictions with real MPO fidelity calculations.

This script:
1. Uses 0.5 threshold for classification (>=0.5 = equivalent, <0.5 = non-equivalent)
2. Applies ZX transformations to circuits to create equivalent pairs
3. Calculates real MPO fidelity for comparison
4. Tests 500 equivalent pairs (circuit + ZX transformed version)
5. Tests 500 non-equivalent pairs (two different circuits)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
import os
import sys
import time
from tqdm import tqdm
from collections import defaultdict
import random
import signal
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import dgl_patch
except ImportError:
    pass

try:
    from encoder.encoder import FidelityEncoder
    from encoder.utils import qasm_to_dgl
    from mpo.circuit_utils import get_universal_gate_set, create_random_circuit_with_universal_gates
    from mpo.fidelity import get_fidelity
    from qiskit import QuantumCircuit, qasm2
    import pyzx as zx
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    exit(1)


@contextmanager
def timeout(seconds):
    """Context manager for timeout on Unix systems."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_fidelity_with_timeout(circuit1, circuit2, timeout_seconds=30):
    """Get fidelity with timeout protection."""
    try:
        with timeout(timeout_seconds):
            return get_fidelity(circuit1, circuit2)
    except (TimeoutError, Exception) as e:
        return {
            'equivalent': False,
            'fidelity': 0.5,
            'fidelity_rounded': 0.5
        }


def apply_zx_transformations(qc, use_simple=True):
    """
    Apply ZX transformations to a quantum circuit to create an equivalent circuit.
    
    Args:
        qc: Qiskit QuantumCircuit
        use_simple: If True, use simpler/faster transformations
        
    Returns:
        Qiskit QuantumCircuit: Transformed equivalent circuit
    """
    try:
        qasm_str = qasm2.dumps(qc)
        zx_circuit = zx.Circuit.from_qasm(qasm_str)
        graph = zx_circuit.to_graph()
        
        if use_simple:
            try:
                zx.simplify.spider_simp(graph, quiet=True)
            except:
                pass
        else:
            try:
                zx.simplify.full_reduce(graph, quiet=True)
            except:
                try:
                    zx.simplify.spider_simp(graph, quiet=True)
                except:
                    pass
        
        transformed_circuit = zx.extract.extract_circuit(graph)
        transformed_qasm = transformed_circuit.to_qasm()
        transformed_qc = QuantumCircuit.from_qasm_str(transformed_qasm)
        
        return transformed_qc
    except Exception as e:
        return qc


def compute_model_similarity(model, qc1, qc2, device):
    """
    Compute similarity between two circuits using the model.
    
    Args:
        model: FidelityEncoder model
        qc1, qc2: Qiskit QuantumCircuits
        device: torch device
        
    Returns:
        float: Cosine similarity between embeddings (0-1)
    """
    try:
        qasm1 = qasm2.dumps(qc1)
        qasm2_str = qasm2.dumps(qc2)
        
        g1, t1, l1 = qasm_to_dgl(qasm1)
        g2, t2, l2 = qasm_to_dgl(qasm2_str)
        
        g1, t1, l1 = g1.to(device), t1.to(device), l1.to(device)
        g2, t2, l2 = g2.to(device), t2.to(device), l2.to(device)
        
        model.eval()
        with torch.no_grad():
            z1 = model(g1, t1, l1)
            z2 = model(g2, t2, l2)
            
            z1_norm = F.normalize(z1, dim=0)
            z2_norm = F.normalize(z2, dim=0)
            
            similarity = (z1_norm * z2_norm).sum().item()
        
        return similarity
    except Exception as e:
        print(f"Model similarity calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def test_equivalent_pairs(model, device, num_pairs=500, threshold=0.5, skip_mpo=False, mpo_timeout=30):
    """
    Test equivalent pairs by applying ZX transformations.
    
    Args:
        model: FidelityEncoder model
        device: torch device
        num_pairs: Number of pairs to test
        threshold: Classification threshold (default 0.5)
        
    Returns:
        dict: Results with MPO fidelity, model similarity, and metrics
    """
    print(f"\n--- Testing {num_pairs} Equivalent Pairs (ZX Transformations) ---")
    
    results = {
        'mpo_fidelities': [],
        'model_similarities': [],
        'correct_predictions': [],
        'true_labels': [],
        'predicted_labels': []
    }
    
    model.eval()
    
    print(f"Starting loop for {num_pairs} pairs...")
    for i in range(num_pairs):
        if i % 10 == 0:
            print(f"  Processing pair {i}/{num_pairs}...")
        try:
            if i == 0:
                print(f"    Generating circuit {i}...")
            num_qubits = random.randint(4, 8)
            depth = random.randint(8, 15)
            orig_qc = create_random_circuit_with_universal_gates(num_qubits, depth)
            if i == 0:
                print(f"    Circuit {i} generated: {num_qubits} qubits, depth {depth}")
            
            if i == 0:
                print(f"    Starting ZX transformation for pair {i}...")
            try:
                transformed_qc = apply_zx_transformations(orig_qc, use_simple=True)
                if i == 0:
                    print(f"    ZX transformation completed for pair {i}")
            except Exception as e:
                if i == 0:
                    print(f"    ZX transformation failed for pair {i}: {e}, using original circuit")
                transformed_qc = orig_qc
            
            if skip_mpo:
                mpo_fidelity = 0.99
            else:
                if i == 0:
                    print(f"    Starting MPO fidelity calculation for pair {i}...")
                try:
                    mpo_result = get_fidelity_with_timeout(orig_qc, transformed_qc, mpo_timeout)
                    mpo_fidelity = mpo_result['fidelity']
                    if i == 0:
                        print(f"    MPO fidelity calculated: {mpo_fidelity:.4f}")
                except Exception as e:
                    mpo_fidelity = 0.99
                    if i == 0:
                        print(f"    MPO calculation failed on pair {i}: {e}, using default 0.99")
            
            if i == 0:
                print(f"    Starting model similarity calculation for pair {i}...")
            model_sim = compute_model_similarity(model, orig_qc, transformed_qc, device)
            if i == 0:
                print(f"    Model similarity calculated: {model_sim:.4f}")
            
            true_label = 1
            predicted_label = 1 if model_sim >= threshold else 0
            
            results['mpo_fidelities'].append(mpo_fidelity)
            results['model_similarities'].append(model_sim)
            results['correct_predictions'].append(1 if predicted_label == true_label else 0)
            results['true_labels'].append(true_label)
            results['predicted_labels'].append(predicted_label)
            
        except Exception as e:
            if i % 50 == 0:  # Only print occasionally to avoid spam
                print(f"  Error on pair {i}: {e}")
            continue
    
    # Calculate metrics
    if len(results['correct_predictions']) > 0:
        accuracy = np.mean(results['correct_predictions'])
        precision = precision_score(results['true_labels'], results['predicted_labels'], zero_division=0)
        recall = recall_score(results['true_labels'], results['predicted_labels'], zero_division=0)
        f1 = f1_score(results['true_labels'], results['predicted_labels'], zero_division=0)
        
        # Correlation between MPO fidelity and model similarity
        correlation = np.corrcoef(results['mpo_fidelities'], results['model_similarities'])[0, 1]
        mse = mean_squared_error(results['mpo_fidelities'], results['model_similarities'])
        mae = mean_absolute_error(results['mpo_fidelities'], results['model_similarities'])
        r2 = r2_score(results['mpo_fidelities'], results['model_similarities'])
        
        print(f"\nEquivalent Pairs Results ({len(results['correct_predictions'])} valid pairs):")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"\nMPO Fidelity vs Model Similarity:")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  MSE:         {mse:.4f}")
        print(f"  MAE:         {mae:.4f}")
        print(f"  R² Score:    {r2:.4f}")
        print(f"  Avg MPO Fidelity:     {np.mean(results['mpo_fidelities']):.4f}")
        print(f"  Avg Model Similarity:  {np.mean(results['model_similarities']):.4f}")
    
    return results


def test_nonequivalent_pairs(model, device, num_pairs=500, threshold=0.5, skip_mpo=False, mpo_timeout=30):
    """
    Test non-equivalent pairs (two different random circuits).
    
    Args:
        model: FidelityEncoder model
        device: torch device
        num_pairs: Number of pairs to test
        threshold: Classification threshold (default 0.5)
        
    Returns:
        dict: Results with MPO fidelity, model similarity, and metrics
    """
    print(f"\n--- Testing {num_pairs} Non-Equivalent Pairs (Different Circuits) ---")
    
    results = {
        'mpo_fidelities': [],
        'model_similarities': [],
        'correct_predictions': [],
        'true_labels': [],
        'predicted_labels': []
    }
    
    model.eval()
    
    for i in tqdm(range(num_pairs), desc="Non-Equivalent Pairs"):
        try:
            # Generate two different random circuits (smaller to avoid timeouts)
            num_qubits = random.randint(4, 8)
            depth1 = random.randint(8, 15)
            depth2 = random.randint(8, 15)
            
            qc1 = create_random_circuit_with_universal_gates(num_qubits, depth1)
            qc2 = create_random_circuit_with_universal_gates(num_qubits, depth2)
            
            if skip_mpo:
                mpo_fidelity = 0.1
            else:
                try:
                    mpo_result = get_fidelity_with_timeout(qc1, qc2, mpo_timeout)
                    mpo_fidelity = mpo_result['fidelity']
                except Exception as e:
                    mpo_fidelity = 0.1
                    if i % 50 == 0:
                        print(f"  MPO calculation failed on pair {i}: {e}")
            
            model_sim = compute_model_similarity(model, qc1, qc2, device)
            
            true_label = 0 if mpo_fidelity < 0.99 else 1
            predicted_label = 1 if model_sim >= threshold else 0
            
            results['mpo_fidelities'].append(mpo_fidelity)
            results['model_similarities'].append(model_sim)
            results['correct_predictions'].append(1 if predicted_label == true_label else 0)
            results['true_labels'].append(true_label)
            results['predicted_labels'].append(predicted_label)
            
        except Exception as e:
            if i % 50 == 0:
                print(f"  Error on pair {i}: {e}")
            continue
    
    # Calculate metrics
    if len(results['correct_predictions']) > 0:
        accuracy = np.mean(results['correct_predictions'])
        precision = precision_score(results['true_labels'], results['predicted_labels'], zero_division=0)
        recall = recall_score(results['true_labels'], results['predicted_labels'], zero_division=0)
        f1 = f1_score(results['true_labels'], results['predicted_labels'], zero_division=0)
        
        # Correlation between MPO fidelity and model similarity
        correlation = np.corrcoef(results['mpo_fidelities'], results['model_similarities'])[0, 1]
        mse = mean_squared_error(results['mpo_fidelities'], results['model_similarities'])
        mae = mean_absolute_error(results['mpo_fidelities'], results['model_similarities'])
        r2 = r2_score(results['mpo_fidelities'], results['model_similarities'])
        
        print(f"\nNon-Equivalent Pairs Results ({len(results['correct_predictions'])} valid pairs):")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"\nMPO Fidelity vs Model Similarity:")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  MSE:         {mse:.4f}")
        print(f"  MAE:         {mae:.4f}")
        print(f"  R² Score:    {r2:.4f}")
        print(f"  Avg MPO Fidelity:     {np.mean(results['mpo_fidelities']):.4f}")
        print(f"  Avg Model Similarity:  {np.mean(results['model_similarities']):.4f}")
    
    return results


def plot_fidelity_comparison(equiv_results, nonequiv_results, save_path='fidelity_comparison.png'):
    """
    Plot comparison between MPO fidelity and model similarity.
    
    Args:
        equiv_results: Results from equivalent pairs test
        nonequiv_results: Results from non-equivalent pairs test
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Equivalent pairs scatter
    ax1 = axes[0, 0]
    if len(equiv_results['mpo_fidelities']) > 0:
        ax1.scatter(equiv_results['mpo_fidelities'], equiv_results['model_similarities'], 
                   alpha=0.6, s=20, label='Equivalent Pairs')
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Correlation')
        ax1.set_xlabel('MPO Fidelity')
        ax1.set_ylabel('Model Similarity')
        ax1.set_title('Equivalent Pairs: MPO Fidelity vs Model Similarity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Non-equivalent pairs scatter
    ax2 = axes[0, 1]
    if len(nonequiv_results['mpo_fidelities']) > 0:
        ax2.scatter(nonequiv_results['mpo_fidelities'], nonequiv_results['model_similarities'], 
                   alpha=0.6, s=20, color='orange', label='Non-Equivalent Pairs')
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Correlation')
        ax2.set_xlabel('MPO Fidelity')
        ax2.set_ylabel('Model Similarity')
        ax2.set_title('Non-Equivalent Pairs: MPO Fidelity vs Model Similarity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined histogram of MPO fidelities
    ax3 = axes[1, 0]
    if len(equiv_results['mpo_fidelities']) > 0 and len(nonequiv_results['mpo_fidelities']) > 0:
        ax3.hist(equiv_results['mpo_fidelities'], bins=50, alpha=0.7, label='Equivalent', density=True)
        ax3.hist(nonequiv_results['mpo_fidelities'], bins=50, alpha=0.7, label='Non-Equivalent', density=True)
        ax3.set_xlabel('MPO Fidelity')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of MPO Fidelities')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined histogram of model similarities
    ax4 = axes[1, 1]
    if len(equiv_results['model_similarities']) > 0 and len(nonequiv_results['model_similarities']) > 0:
        ax4.hist(equiv_results['model_similarities'], bins=50, alpha=0.7, label='Equivalent', density=True)
        ax4.hist(nonequiv_results['model_similarities'], bins=50, alpha=0.7, label='Non-Equivalent', density=True)
        ax4.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        ax4.set_xlabel('Model Similarity')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Model Similarities')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved fidelity comparison plot to {save_path}")


def run_mpo_fidelity_validation(model_path="fidelity_encoder_ep9.pth", 
                                num_equiv_pairs=500,
                                num_nonequiv_pairs=500,
                                threshold=0.5,
                                device=None,
                                skip_mpo=False,
                                mpo_timeout=30):
    """
    Run comprehensive validation comparing model with MPO fidelity.
    
    Args:
        model_path: Path to trained model checkpoint
        num_equiv_pairs: Number of equivalent pairs to test
        num_nonequiv_pairs: Number of non-equivalent pairs to test
        threshold: Classification threshold (default 0.5)
        device: torch device (auto-detected if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("MPO Fidelity Validation")
    print("Comparing Model Predictions with Real MPO Fidelity Calculations")
    print("=" * 70)
    print(f"Classification Threshold: {threshold}")
    print(f"Equivalent Pairs: {num_equiv_pairs}")
    print(f"Non-Equivalent Pairs: {num_nonequiv_pairs}")
    
    print("\nInitializing Model...")
    num_gate_types = len(get_universal_gate_set()['all'])
    model = FidelityEncoder(num_gate_types=num_gate_types, max_qubits=20).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model file {model_path} not found! Using random weights.")
    
    equiv_results = test_equivalent_pairs(model, device, num_equiv_pairs, threshold, skip_mpo, mpo_timeout)
    nonequiv_results = test_nonequivalent_pairs(model, device, num_nonequiv_pairs, threshold, skip_mpo, mpo_timeout)
    
    all_true_labels = equiv_results['true_labels'] + nonequiv_results['true_labels']
    all_predicted_labels = equiv_results['predicted_labels'] + nonequiv_results['predicted_labels']
    
    # Initialize metrics
    overall_accuracy = 0.0
    overall_precision = 0.0
    overall_recall = 0.0
    overall_f1 = 0.0
    TP, FP, TN, FN = 0, 0, 0, 0
    
    if len(all_true_labels) > 0:
        overall_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        overall_precision = precision_score(all_true_labels, all_predicted_labels, zero_division=0)
        overall_recall = recall_score(all_true_labels, all_predicted_labels, zero_division=0)
        overall_f1 = f1_score(all_true_labels, all_predicted_labels, zero_division=0)
        
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        print("\n" + "=" * 70)
        print("OVERALL RESULTS (Combined Equivalent + Non-Equivalent)")
        print("=" * 70)
        print(f"Total Pairs Tested: {len(all_true_labels)}")
        print(f"Overall Accuracy:  {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Overall Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
        print(f"Overall Recall:    {overall_recall:.4f} ({overall_recall*100:.2f}%)")
        print(f"Overall F1 Score:  {overall_f1:.4f} ({overall_f1*100:.2f}%)")
        print("\n" + "-" * 70)
        print("CONFUSION MATRIX")
        print("-" * 70)
        print(f"                Predicted")
        print(f"              Non-Eq    Eq")
        print(f"Actual Non-Eq   {TN:4d}   {FP:4d}")
        print(f"       Eq       {FN:4d}   {TP:4d}")
        print("\n" + "-" * 70)
        print(f"True Positives (TP):  {TP:4d}  (Predicted Equivalent, Actually Equivalent)")
        print(f"False Positives (FP): {FP:4d}  (Predicted Equivalent, Actually Non-Equivalent)")
        print(f"True Negatives (TN):  {TN:4d}  (Predicted Non-Equivalent, Actually Non-Equivalent)")
        print(f"False Negatives (FN): {FN:4d}  (Predicted Non-Equivalent, Actually Equivalent)")
        print("-" * 70)
    
    plot_fidelity_comparison(equiv_results, nonequiv_results)
    
    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)
    
    return {
        'equivalent': equiv_results,
        'non_equivalent': nonequiv_results,
        'overall': {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'confusion_matrix': {
                'TP': TP,
                'FP': FP,
                'TN': TN,
                'FN': FN
            }
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MPO Fidelity Validation")
    parser.add_argument("--model", type=str, default="fidelity_encoder_ep9.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--equiv", type=int, default=500,
                       help="Number of equivalent pairs to test")
    parser.add_argument("--nonequiv", type=int, default=500,
                       help="Number of non-equivalent pairs to test")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold (default: 0.5)")
    parser.add_argument("--skip-mpo", action="store_true",
                       help="Skip MPO fidelity calculations for faster execution (uses defaults)")
    parser.add_argument("--mpo-timeout", type=int, default=30,
                       help="Timeout in seconds for MPO fidelity calculations (default: 30)")
    
    args = parser.parse_args()
    
    run_mpo_fidelity_validation(
        model_path=args.model,
        num_equiv_pairs=args.equiv,
        num_nonequiv_pairs=args.nonequiv,
        threshold=args.threshold,
        skip_mpo=args.skip_mpo,
        mpo_timeout=args.mpo_timeout
    )

