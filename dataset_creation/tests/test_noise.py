"""
Tests for the noise application module.
"""

import torch
import numpy as np
from qiskit import QuantumCircuit

import sys
import os

# Ensure project root on sys.path for local imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from encoder.noise import (
    apply_noise_to_circuit,
    GATE_TO_IDX,
    IDX_TO_GATE,
    NUM_GATE_TYPES,
    _SINGLES,
    _PARAM_GATES,
)
from src.model.diffusion import DiscreteDiffusion
from circuit_utils import create_random_circuit_with_universal_gates


def get_diffusion():
    """Create a DiscreteDiffusion instance for testing."""
    marginal = torch.ones(NUM_GATE_TYPES) / NUM_GATE_TYPES
    return DiscreteDiffusion(marginal_list=[marginal], T=100)


def test_empty_circuit_returns_same():
    """Empty circuit should be returned unchanged."""
    diffusion = get_diffusion()
    qc = QuantumCircuit(2)
    result = apply_noise_to_circuit(qc, t_val=10, diffusion=diffusion)
    assert len(result.data) == 0, "Empty circuit should have no gates"
    assert result.num_qubits == qc.num_qubits, "Qubit count should match"
    print("✓ test_empty_circuit_returns_same passed")


def test_output_is_valid_circuit():
    """The output should be a valid QuantumCircuit."""
    diffusion = get_diffusion()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(np.pi / 4, 0)
    
    result = apply_noise_to_circuit(qc, t_val=10, diffusion=diffusion)
    
    assert isinstance(result, QuantumCircuit), "Output should be QuantumCircuit"
    assert result.num_qubits == qc.num_qubits, "Qubit count should match"
    assert len(result.data) == len(qc.data), "Gate count should match"
    print("✓ test_output_is_valid_circuit passed")


def test_topology_constraint_single_qubit():
    """Single-qubit gates should not be swapped with two-qubit gates."""
    diffusion = get_diffusion()
    np.random.seed(42)
    torch.manual_seed(42)
    
    qc = QuantumCircuit(2)
    qc.h(0)  # Single-qubit gate
    
    # Run multiple times to check constraint is maintained
    for _ in range(10):
        result = apply_noise_to_circuit(qc, t_val=50, diffusion=diffusion)
        for inst in result.data:
            gate_name = inst.operation.name
            assert gate_name in _SINGLES or gate_name == 'h', \
                f"Single-qubit gate swapped to non-single: {gate_name}"
    print("✓ test_topology_constraint_single_qubit passed")


def test_topology_constraint_two_qubit():
    """Two-qubit gates should not be swapped with single-qubit gates."""
    diffusion = get_diffusion()
    np.random.seed(42)
    torch.manual_seed(42)
    
    qc = QuantumCircuit(2)
    qc.cx(0, 1)  # Two-qubit gate
    
    for _ in range(10):
        result = apply_noise_to_circuit(qc, t_val=50, diffusion=diffusion)
        for inst in result.data:
            gate_name = inst.operation.name
            assert gate_name not in _SINGLES, \
                f"Two-qubit gate swapped to single-qubit: {gate_name}"
    print("✓ test_topology_constraint_two_qubit passed")


def test_preserves_qubit_count():
    """Circuit qubit count should be preserved."""
    diffusion = get_diffusion()
    for num_qubits in [2, 3, 5]:
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        
        result = apply_noise_to_circuit(qc, t_val=10, diffusion=diffusion)
        assert result.num_qubits == num_qubits, f"Expected {num_qubits} qubits"
    print("✓ test_preserves_qubit_count passed")


def test_preserves_gate_count():
    """Number of gates should be preserved (no additions/removals)."""
    diffusion = get_diffusion()
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(np.pi / 2, 2)
    qc.cx(1, 2)
    
    original_count = len(qc.data)
    result = apply_noise_to_circuit(qc, t_val=10, diffusion=diffusion)
    
    assert len(result.data) == original_count, "Gate count changed"
    print("✓ test_preserves_gate_count passed")


def test_parametric_gate_params_in_range():
    """Parameters of parametric gates should remain in valid range [0, 2π)."""
    diffusion = get_diffusion()
    np.random.seed(42)
    torch.manual_seed(42)
    
    qc = QuantumCircuit(2)
    qc.rx(np.pi / 4, 0)
    qc.ry(np.pi / 2, 1)
    qc.rz(np.pi, 0)
    
    result = apply_noise_to_circuit(qc, t_val=50, diffusion=diffusion)
    
    for inst in result.data:
        if inst.operation.name in _PARAM_GATES and inst.operation.params:
            for param in inst.operation.params:
                param_val = float(param)
                assert 0 <= param_val < 2 * np.pi, f"Parameter {param_val} out of range"
    print("✓ test_parametric_gate_params_in_range passed")


def test_with_random_circuit():
    """Test with randomly generated circuits."""
    diffusion = get_diffusion()
    np.random.seed(42)
    torch.manual_seed(42)
    
    for i in range(5):
        qc = create_random_circuit_with_universal_gates(
            num_qubits=4, depth=5, seed=None
        )
        
        result = apply_noise_to_circuit(qc, t_val=30, diffusion=diffusion)
        
        assert isinstance(result, QuantumCircuit), "Output should be QuantumCircuit"
        assert result.num_qubits == qc.num_qubits, "Qubit count should match"
        assert len(result.data) == len(qc.data), "Gate count should match"
    print("✓ test_with_random_circuit passed")


def test_reproducibility_with_seed():
    """Same seed should produce same results."""
    diffusion = get_diffusion()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(np.pi / 4, 0)
    
    # Run twice with same seeds
    np.random.seed(123)
    torch.manual_seed(123)
    result1 = apply_noise_to_circuit(qc, t_val=50, diffusion=diffusion)
    
    np.random.seed(123)
    torch.manual_seed(123)
    result2 = apply_noise_to_circuit(qc, t_val=50, diffusion=diffusion)
    
    assert len(result1.data) == len(result2.data), "Gate counts should match"
    for inst1, inst2 in zip(result1.data, result2.data):
        assert inst1.operation.name == inst2.operation.name, "Gate names should match"
        assert list(inst1.operation.params) == list(inst2.operation.params), "Params should match"
    print("✓ test_reproducibility_with_seed passed")


def test_gate_mappings():
    """Test gate index mappings are valid."""
    assert len(GATE_TO_IDX) == NUM_GATE_TYPES, "GATE_TO_IDX incomplete"
    assert len(IDX_TO_GATE) == NUM_GATE_TYPES, "IDX_TO_GATE incomplete"
    
    for gate, idx in GATE_TO_IDX.items():
        assert IDX_TO_GATE[idx] == gate, f"Mapping mismatch for {gate}"
    
    assert len(_SINGLES) > 0, "No single-qubit gates defined"
    assert len(_PARAM_GATES) > 0, "No parametric gates defined"
    print("✓ test_gate_mappings passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running noise module tests...")
    print("=" * 60)
    
    tests = [
        test_empty_circuit_returns_same,
        test_output_is_valid_circuit,
        test_topology_constraint_single_qubit,
        test_topology_constraint_two_qubit,
        test_preserves_qubit_count,
        test_preserves_gate_count,
        test_parametric_gate_params_in_range,
        test_with_random_circuit,
        test_reproducibility_with_seed,
        test_gate_mappings,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

