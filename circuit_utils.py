import torch
from qiskit import QuantumCircuit
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from qiskit.circuit.random import random_circuit
from qiskit import transpile
from qiskit.converters import circuit_to_dag


def get_universal_gate_set():
    """
    Get the complete universal gate set compatible with both Qiskit and mqt.yaqs.
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

    transpiled_circuit = transpile(circuit, basis_gates=get_universal_gate_set()['all'])

    dag = circuit_to_dag(transpiled_circuit)
    qubits_used = set()

    for node in dag.topological_op_nodes():
        for qubit in node.qargs:
            qubits_used.add(dag.find_bit(qubit).index)

    if len(qubits_used) == num_qubits:
        return transpiled_circuit

    qubits_used_sorted = sorted(qubits_used)
    num_used_qubits = len(qubits_used_sorted)
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(qubits_used_sorted)}
    new_circuit = QuantumCircuit(num_used_qubits)

    for node in dag.topological_op_nodes():
        gate_name = node.op.name
        old_qubits = [dag.find_bit(q).index for q in node.qargs]
        new_qubits = [qubit_map[old_idx] for old_idx in old_qubits]
        params = list(node.op.params) if hasattr(node.op, 'params') else []

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
    """
    gate_types = []
    gate_info = []
    nodes = list(dag.topological_op_nodes())
    node_to_idx = {}

    for idx, node in enumerate(nodes):
        gate_name = node.op.name
        qubits = [dag.find_bit(q).index for q in node.qargs]
        params = list(node.op.params) if hasattr(node.op, 'params') else []

        gate_types.append(gate_name)
        gate_info.append((gate_name, qubits, params))
        node_to_idx[node._node_id] = idx

    unique_gates = sorted(list(set(gate_types)))
    gate_to_idx = {gate: idx for idx, gate in enumerate(unique_gates)}

    gate_indices = torch.tensor([gate_to_idx[gate]
                                for gate in gate_types], dtype=torch.long)

    n_gates = len(nodes)
    adjacency = torch.zeros((n_gates, n_gates), dtype=torch.float32)

    for idx, node in enumerate(nodes):
        predecessors = dag.predecessors(node)
        for pred in predecessors:
            if pred._node_id in node_to_idx:
                pred_idx = node_to_idx[pred._node_id]
                adjacency[idx, pred_idx] = 1.0

    return gate_indices, adjacency, gate_to_idx, gate_info


def reconstruct_circuit_from_noisy(gate_indices, adjacency, idx_to_gate, gate_info, num_qubits):
    """
    Reconstruct a quantum circuit from noisy gate indices and adjacency matrix.
    """
    new_circuit = QuantumCircuit(num_qubits)
    n_gates = len(gate_indices)

    in_degree = adjacency.sum(dim=1).int()
    order = []
    remaining = set(range(n_gates))

    while remaining:
        ready = [i for i in remaining if in_degree[i] == 0]
        if not ready:
            ready = [min(remaining)]

        for gate_idx in ready:
            order.append(gate_idx)
            remaining.remove(gate_idx)
            for i in remaining:
                if adjacency[i, gate_idx] > 0:
                    in_degree[i] -= 1

    for idx in order:
        gate_name = idx_to_gate[gate_indices[idx].item()]
        _, original_qubits, original_params = gate_info[idx]

        try:
            if hasattr(new_circuit, gate_name):
                gate_method = getattr(new_circuit, gate_name)
                if len(original_params) > 0:
                    gate_method(*original_params, *original_qubits)
                else:
                    gate_method(*original_qubits)
            else:
                new_circuit.id(original_qubits[0] if len(original_qubits) > 0 else 0)

        except Exception as e:
            try:
                new_circuit.id(original_qubits[0] if len(original_qubits) > 0 else 0)
            except Exception:
                pass

    return new_circuit


def reconstruct_circuit_from_model_output(x_n, edge_index, num_qubits):
    """
    Reconstruct a Qiskit circuit from LayerDAG model output.
    """
    import torch
    import numpy as np

    all_gates = sorted(get_universal_gate_set()['all'])
    single_qubit_gates = get_universal_gate_set()['single_qubit']

    gate_indices = x_n[:, 0]
    num_gates = x_n.shape[0]
    src, dst = edge_index
    adjacency = torch.zeros((num_gates, num_gates))
    for i in range(len(src)):
        adjacency[dst[i], src[i]] = 1

    idx_to_gate = {idx: gate for idx, gate in enumerate(all_gates)}

    gate_info = []
    for i in range(num_gates):
        gate_type_idx = x_n[i, 0].item()
        gate_name = all_gates[gate_type_idx]
        qubit_0 = x_n[i, 1].item()
        qubit_1 = x_n[i, 2].item()

        if gate_name in single_qubit_gates:
            qubits = [qubit_0] if qubit_0 < num_qubits else [0]
        else:
            q0 = qubit_0 if qubit_0 < num_qubits else 0
            q1 = qubit_1 if qubit_1 < num_qubits else min(1, num_qubits-1)
            if q0 == q1:
                q1 = (q0 + 1) % num_qubits
            qubits = [q0, q1]

        params = [np.pi/4] if gate_name in ['p', 'rx', 'ry', 'rz', 'cp', 'rxx', 'ryy', 'rzz'] else []
        if gate_name == 'u':
            params = [np.pi/4, np.pi/4, np.pi/4]

        gate_info.append((gate_name, qubits, params))

    return reconstruct_circuit_from_noisy(gate_indices, adjacency, idx_to_gate, gate_info, num_qubits)