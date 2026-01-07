import torch
import numpy as np
from qiskit import QuantumCircuit

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from circuit_utils import get_universal_gate_set


_UNIV_SETS = get_universal_gate_set()
GATE_TO_IDX = {g: i for i, g in enumerate(_UNIV_SETS['all'])}
IDX_TO_GATE = {i: g for g, i in GATE_TO_IDX.items()}
NUM_GATE_TYPES = len(_UNIV_SETS['all'])
_SINGLES = frozenset(_UNIV_SETS['single_qubit'])
_PARAM_GATES = frozenset(_UNIV_SETS['parametric'])


def apply_noise_to_circuit(qc, t_val, diffusion):
    if len(qc.data) == 0:
        return qc

    gate_indices = [GATE_TO_IDX.get(inst.operation.name, 0) for inst in qc.data]
    
    x_0 = torch.tensor(gate_indices, dtype=torch.long).unsqueeze(1)
    t = torch.tensor([t_val])
    
    _, x_t = diffusion.apply_noise(x_0, t)
    noisy_indices = x_t.reshape(-1).tolist()
    
    intensity = t.item() / diffusion.T
    drift_prob = intensity
    max_drift = 0.1
    sigma = max_drift * intensity

    noisy_qc = qc.copy_empty_like()
    changes = 0
    
    for i, instruction in enumerate(qc.data):
        op = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        
        orig_name = op.name
        new_idx = noisy_indices[i]
        new_name = IDX_TO_GATE.get(new_idx, orig_name)

        is_orig_single = orig_name in _SINGLES
        is_new_single = new_name in _SINGLES
        
        if is_orig_single != is_new_single:
            final_name = orig_name
        else:
            final_name = new_name

        if final_name != orig_name:
            changes += 1

        if final_name in _PARAM_GATES:
            if final_name == orig_name and op.params:
                if np.random.random() < drift_prob:
                    old_arr = np.array(op.params, dtype=float)
                    perturbation = np.random.normal(0, sigma, size=len(old_arr))
                    new_params = ((old_arr + perturbation) % (2 * np.pi)).tolist()
                else:
                    new_params = list(op.params)
            else:
                num_params_needed = 1 
                new_params = np.random.uniform(0, 2 * np.pi, num_params_needed).tolist()
        else:
            new_params = []

        if final_name != orig_name:
            try:
                if new_params:
                    getattr(noisy_qc, final_name)(*new_params, *qubits)
                else:
                    getattr(noisy_qc, final_name)(*qubits)
            except AttributeError:
                noisy_qc.append(op, qubits, clbits)
        else:
            if new_params != list(op.params):
                new_op = op.copy()
                new_op.params = new_params
                noisy_qc.append(new_op, qubits, clbits)
            else:
                noisy_qc.append(op, qubits, clbits)

    return noisy_qc
