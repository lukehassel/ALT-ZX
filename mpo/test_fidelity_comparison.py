#!/usr/bin/env python3

from qiskit.circuit.random import random_circuit
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from mpo.gpu.mpo_gpu import get_fidelity_gpu
from mpo.fidelity import get_fidelity as get_fidelity_cpu


def test_comparison():
    print("=" * 70)
    print("GPU vs CPU Fidelity Comparison Test")
    print("=" * 70)
    print()
    
    basis_gates = ['cx', 'h', 'rz', 'sx', 'x']
    all_match = True
    
    for num_qubits in [4, 8]:
        print(f"Testing with {num_qubits} qubits:")
        depth = 2
        
        print(f"  Test 1: Different circuits")
        print(f"  {'Seed Pair':<15} {'GPU Fidelity':<15} {'CPU Fidelity':<15} {'Diff':<12} {'Match':<8}")
        
        for seed1, seed2 in [(42, 43), (100, 101)]:
            c1 = random_circuit(num_qubits, depth, seed=seed1)
            c2 = random_circuit(num_qubits, depth, seed=seed2)
            
            coupling = CouplingMap.from_line(num_qubits)
            c1 = transpile(c1, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
            c2 = transpile(c2, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
            
            r_gpu = get_fidelity_gpu(c1, c2, transpile_to_linear=False)
            r_cpu = get_fidelity_cpu(c1, c2)
            
            diff = abs(r_gpu['fidelity'] - r_cpu['fidelity'])
            match = diff < 0.0001
            all_match = all_match and match
            
            print(f"  ({seed1}, {seed2})".ljust(15) + 
                  f"{r_gpu['fidelity']:<15.6f} {r_cpu['fidelity']:<15.6f} {diff:<12.6f} {'✓' if match else '✗':<8}")
        
        print(f"  Test 2: Same circuits (fidelity should be 1.0)")
        print(f"  {'Seed':<10} {'GPU Fidelity':<15} {'CPU Fidelity':<15} {'Both 1.0?':<12}")
        
        for seed in [42, 100]:
            c1 = random_circuit(num_qubits, depth, seed=seed)
            coupling = CouplingMap.from_line(num_qubits)
            c1 = transpile(c1, coupling_map=coupling, basis_gates=basis_gates, optimization_level=0)
            c2 = c1.copy()
            
            r_gpu = get_fidelity_gpu(c1, c2, transpile_to_linear=False)
            r_cpu = get_fidelity_cpu(c1, c2)
            
            both_one = r_gpu['fidelity'] > 0.999 and r_cpu['fidelity'] > 0.999
            all_match = all_match and both_one
            
            print(f"  {seed:<10} {r_gpu['fidelity']:<15.6f} {r_cpu['fidelity']:<15.6f} {'✓' if both_one else '✗':<12}")
        print("-" * 50)

    print()
    print("=" * 70)
    if all_match:
        print("✓ ALL TESTS PASSED - GPU implementation matches CPU")
    else:
        print("✗ SOME TESTS FAILED - Check implementation")
    print("=" * 70)
    
    return all_match


if __name__ == "__main__":
    test_comparison()
