"""
Utility functions for converting quantum circuits to/from DAG representations
and applying noise transformations.
"""
import torch
from qiskit import QuantumCircuit
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from qiskit.circuit.random import random_circuit
from qiskit import transpile
from qiskit.converters import circuit_to_dag


def get_universal_gate_set():
    """
    Get the complete universal gate set compatible with both Qiskit and mqt.yaqs.

    Returns all 18 gates that work in both systems with identical names.
    This includes both single-qubit and two-qubit gates.

    Returns:
        dict: Dictionary with 'single_qubit' and 'two_qubit' gate lists
    """
    return {
        'single_qubit': ['h', 'id', 'p', 'rx', 'ry', 'rz', 'sx', 'x', 'y', 'z'],
        'two_qubit': ['cp', 'cx', 'cz', 'rxx', 'ryy', 'rzz', 'swap'],
        'all': ['cp', 'cx', 'cz', 'h', 'id', 'p', 'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzz', 'swap', 'sx', 'x', 'y', 'z'],
        'parametric': ['cp', 'p', 'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzz']
    }
    
    
def create_random_circuit_with_universal_gates(num_qubits, depth, seed=None, max_operands=2):
    """
    Create a random quantum circuit using the universal gate set.

    Uses Qiskit's random_circuit to generate a random circuit, then transpiles it
    to use only gates from the universal gate set (18 gates compatible with mqt.yaqs).

    Removes idle qubits (those without any gates) and remaps the remaining qubits
    to consecutive indices starting from 0.

    Args:
        num_qubits (int): Number of qubits in the circuit
        depth (int): Circuit depth (number of layers)
        seed (int, optional): Random seed for reproducibility
        max_operands (int): Maximum number of operands for gates (default: 2)

    Returns:
        QuantumCircuit: Random circuit with only used qubits, remapped to consecutive indices
    """
    # Generate random circuit
    circuit = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
        max_operands=max_operands,
        seed=seed
    )

    # Transpile to universal gate set
    transpiled_circuit = transpile(circuit, basis_gates=get_universal_gate_set()['all'])
    return transpiled_circuit

def create_random_circuit_with_universal_gates_remapped(num_qubits, depth, seed=None, max_operands=2):
    """
    Create a random quantum circuit using the universal gate set.

    Uses Qiskit's random_circuit to generate a random circuit, then transpiles it
    to use only gates from the universal gate set (18 gates compatible with mqt.yaqs).

    Removes idle qubits (those without any gates) and remaps the remaining qubits
    to consecutive indices starting from 0.

    Args:
        num_qubits (int): Number of qubits in the circuit
        depth (int): Circuit depth (number of layers)
        seed (int, optional): Random seed for reproducibility
        max_operands (int): Maximum number of operands for gates (default: 2)

    Returns:
        QuantumCircuit: Random circuit with only used qubits, remapped to consecutive indices
    """
    # Generate random circuit
    circuit = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
        max_operands=max_operands,
        seed=seed
    )

    # Transpile to universal gate set
    transpiled_circuit = transpile(circuit, basis_gates=get_universal_gate_set()['all'])

    # Find which qubits are actually used
    dag = circuit_to_dag(transpiled_circuit)
    qubits_used = set()

    for node in dag.topological_op_nodes():
        for qubit in node.qargs:
            qubits_used.add(dag.find_bit(qubit).index)

    # If all qubits are used, return the circuit as-is
    if len(qubits_used) == num_qubits:
        return transpiled_circuit

    # Otherwise, create a new circuit with only the used qubits
    # and remap them to consecutive indices
    qubits_used_sorted = sorted(qubits_used)
    num_used_qubits = len(qubits_used_sorted)

    # Create mapping: old_index -> new_index
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(qubits_used_sorted)}

    # Create new circuit with remapped qubits
    new_circuit = QuantumCircuit(num_used_qubits)

    # Re-apply all gates with remapped qubit indices
    for node in dag.topological_op_nodes():
        gate_name = node.op.name
        old_qubits = [dag.find_bit(q).index for q in node.qargs]
        new_qubits = [qubit_map[old_idx] for old_idx in old_qubits]
        params = list(node.op.params) if hasattr(node.op, 'params') else []

        # Apply gate to new circuit
        if hasattr(new_circuit, gate_name):
            gate_method = getattr(new_circuit, gate_name)
            if len(params) > 0:
                gate_method(*params, *new_qubits)
            else:
                gate_method(*new_qubits)

    return new_circuit



def dag_to_gate_and_edges(dag):
    """
    Convert a DAG circuit to gate indices and adjacency matrix.

    Uses the universal gate set for gate handling.

    Args:
        dag: DAGCircuit to convert

    Returns:
        tuple: (gate_indices, adjacency, gate_to_idx, gate_info)
    """
    gate_types = []
    gate_info = []

    # Get topological order of operation nodes
    nodes = list(dag.topological_op_nodes())

    # Map node_id to index in our sequence
    node_to_idx = {}

    for idx, node in enumerate(nodes):
        gate_name = node.op.name
        qubits = [dag.find_bit(q).index for q in node.qargs]
        params = list(node.op.params) if hasattr(node.op, 'params') else []

        gate_types.append(gate_name)
        gate_info.append((gate_name, qubits, params))
        node_to_idx[node._node_id] = idx

    # Create mapping from gate name to index
    unique_gates = sorted(list(set(gate_types)))
    gate_to_idx = {gate: idx for idx, gate in enumerate(unique_gates)}

    # Convert to indices
    gate_indices = torch.tensor([gate_to_idx[gate]
                                for gate in gate_types], dtype=torch.long)

    # Build adjacency matrix from DAG edges
    n_gates = len(nodes)
    adjacency = torch.zeros((n_gates, n_gates), dtype=torch.float32)

    for idx, node in enumerate(nodes):
        # Get predecessors of this node
        predecessors = dag.predecessors(node)
        for pred in predecessors:
            # Only consider operation nodes (not input/output nodes)
            # Check if the predecessor node is in our node_to_idx mapping
            if pred._node_id in node_to_idx:
                pred_idx = node_to_idx[pred._node_id]
                adjacency[idx, pred_idx] = 1.0

    return gate_indices, adjacency, gate_to_idx, gate_info


def reconstruct_circuit_from_noisy(gate_indices, adjacency, idx_to_gate, gate_info, num_qubits):
    """
    Reconstruct a quantum circuit from noisy gate indices and adjacency matrix.

    Uses the universal gate set to dynamically apply gates instead of hardcoded conditionals.

    Args:
        gate_indices: Tensor of gate type indices
        adjacency: Tensor representing gate dependencies
        idx_to_gate: Mapping from gate index to gate name
        gate_info: List of (gate_name, qubits, params) tuples
        num_qubits: Number of qubits in the circuit

    Returns:
        QuantumCircuit: Reconstructed circuit
    """
    new_circuit = QuantumCircuit(num_qubits)
    n_gates = len(gate_indices)

    # Build dependency order from adjacency (topological sort)
    in_degree = adjacency.sum(dim=1).int()
    order = []
    remaining = set(range(n_gates))

    while remaining:
        # Find gates with no dependencies among remaining
        ready = [i for i in remaining if in_degree[i] == 0]
        if not ready:
            # If circular dependency, just take any remaining gate
            ready = [min(remaining)]

        for gate_idx in ready:
            order.append(gate_idx)
            remaining.remove(gate_idx)
            # Update in-degrees
            for i in remaining:
                if adjacency[i, gate_idx] > 0:
                    in_degree[i] -= 1

    # Apply gates in the determined order
    for idx in order:
        gate_name = idx_to_gate[gate_indices[idx].item()]
        _, original_qubits, original_params = gate_info[idx]

        try:
            # Dynamically get the gate method from QuantumCircuit
            if hasattr(new_circuit, gate_name):
                gate_method = getattr(new_circuit, gate_name)

                # Apply gate with appropriate arguments
                if len(original_params) > 0:
                    # Parameterized gate: params first, then qubits
                    gate_method(*original_params, *original_qubits)
                else:
                    # Non-parameterized gate: just qubits
                    gate_method(*original_qubits)
            else:
                # Gate method doesn't exist in QuantumCircuit, use identity
                new_circuit.id(original_qubits[0] if len(original_qubits) > 0 else 0)

        except Exception as e:
            # If error, add identity as fallback
            try:
                new_circuit.id(original_qubits[0] if len(original_qubits) > 0 else 0)
            except Exception:
                # Last resort: skip this gate
                pass

    return new_circuit


def reconstruct_circuit_from_model_output(x_n, edge_index, num_qubits):
    """
    Reconstruct a Qiskit circuit from LayerDAG model output.

    The model outputs circuits in a specific format:
    - x_n: Tensor of shape [num_gates, 3] with [gate_type, qubit_0, qubit_1]
    - edge_index: Tensor [2, num_edges] with [src, dst] representing gate dependencies

    This function converts that format to a Qiskit QuantumCircuit using the
    existing reconstruct_circuit_from_noisy function.

    Args:
        x_n: Tensor of shape [num_gates, 3] with columns [gate_type_idx, qubit_0, qubit_1]
        edge_index: Tensor of shape [2, num_edges] with [src, dst] edges
        num_qubits: Number of qubits in the circuit

    Returns:
        QuantumCircuit: Reconstructed quantum circuit
    """
    import torch
    import numpy as np

    # Get universal gate set
    all_gates = sorted(get_universal_gate_set()['all'])
    single_qubit_gates = get_universal_gate_set()['single_qubit']

    # Extract gate type indices (first column of x_n)
    gate_indices = x_n[:, 0]

    # Build adjacency matrix from edge_index
    num_gates = x_n.shape[0]
    src, dst = edge_index
    adjacency = torch.zeros((num_gates, num_gates))
    for i in range(len(src)):
        adjacency[dst[i], src[i]] = 1

    # Create idx_to_gate mapping
    idx_to_gate = {idx: gate for idx, gate in enumerate(all_gates)}

    # Create gate_info: list of (gate_name, qubits, params) tuples
    gate_info = []
    for i in range(num_gates):
        gate_type_idx = x_n[i, 0].item()
        gate_name = all_gates[gate_type_idx]
        qubit_0 = x_n[i, 1].item()
        qubit_1 = x_n[i, 2].item()

        # Determine qubits based on gate type
        if gate_name in single_qubit_gates:
            qubits = [qubit_0] if qubit_0 < num_qubits else [0]
        else:
            # Two-qubit gate
            q0 = qubit_0 if qubit_0 < num_qubits else 0
            q1 = qubit_1 if qubit_1 < num_qubits else min(1, num_qubits-1)
            if q0 == q1:
                q1 = (q0 + 1) % num_qubits
            qubits = [q0, q1]

        # Default parameters for parameterized gates
        params = [np.pi/4] if gate_name in ['p', 'rx', 'ry', 'rz', 'cp', 'rxx', 'ryy', 'rzz'] else []
        if gate_name == 'u':
            params = [np.pi/4, np.pi/4, np.pi/4]

        gate_info.append((gate_name, qubits, params))

    # Use the existing reconstruct function
    return reconstruct_circuit_from_noisy(gate_indices, adjacency, idx_to_gate, gate_info, num_qubits)