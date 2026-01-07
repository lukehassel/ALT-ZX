from qiskit import QuantumCircuit
import dgl
import torch
from mpo.circuit_utils import get_universal_gate_set

GATE_TO_IDX = {g: i for i, g in enumerate(get_universal_gate_set()['all'])}


def qasm_to_dgl(qasm_str):
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        
        src_nodes = []
        dst_nodes = []
        node_gate_types = []
        node_qubit_indices = []
        
        qubit_map = {q: i for i, q in enumerate(qc.qubits)}
        last_node_on_wire = {i: -1 for i in range(len(qc.qubits))}
        
        current_node_id = 0
        
        for instruction in qc.data:
            op = instruction.operation
            name = op.name
            
            if name in ['barrier', 'measure']:
                continue
                
            g_id = GATE_TO_IDX.get(name, 0)
            q_indices = [qubit_map[q] for q in instruction.qubits]
            primary_qubit = q_indices[0] if q_indices else 0
            
            node_gate_types.append(g_id)
            node_qubit_indices.append(primary_qubit)
            
            for q in q_indices:
                prev_node = last_node_on_wire[q]
                if prev_node != -1:
                    src_nodes.append(prev_node)
                    dst_nodes.append(current_node_id)
                last_node_on_wire[q] = current_node_id
            
            current_node_id += 1
        
        if current_node_id == 0: 
            g = dgl.graph(([], []), num_nodes=1)
            return g, torch.tensor([0]), torch.tensor([0])

        g = dgl.graph((src_nodes, dst_nodes), num_nodes=current_node_id)
        g = dgl.add_self_loop(g)
        
        return g, torch.tensor(node_gate_types), torch.tensor(node_qubit_indices)
        
    except Exception as e:
        print(f"Error parsing QASM: {e}")
        g = dgl.graph(([0], [0]), num_nodes=1)
        return g, torch.tensor([0]), torch.tensor([0])