import torch
import numpy as np
from typing import Optional, Tuple, List
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import CouplingMap
import warnings


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = get_device()
DTYPE = torch.complex128

import os
from torch.utils.cpp_extension import load

# Custom CUDA kernel disabled due to NaN issues on H100
MPO_CUDA = None
USE_CUSTOM_CUDA_KERNEL = False
import sys

base_path = os.path.dirname(os.path.abspath(__file__))
if base_path not in sys.path:
    sys.path.append(base_path)

if USE_CUSTOM_CUDA_KERNEL:
    try:
        import mpo_cuda_ext as MPO_CUDA
    except ImportError:
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            MPO_CUDA = load(
                name="mpo_cuda_ext",
                sources=[
                    os.path.join(base_path, "mpo_cuda.cpp"),
                    os.path.join(base_path, "mpo_cuda_kernel.cu"),
                ],
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                verbose=False
            )
        except Exception as e:
            if "Ninja is required" not in str(e):
                print(f"Warning: Failed to load CUDA extension: {e}")
            MPO_CUDA = None


def randomized_svd_gpu(M: torch.Tensor, rank: int, n_iter: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = M.shape
    k = min(rank, m, n)
    
    Omega = torch.randn(n, k, dtype=M.dtype, device=M.device)
    Y = torch.mm(M, Omega)
    
    for _ in range(n_iter):
        Y = torch.mm(M, torch.mm(M.T.conj(), Y))
        
    Q, _ = torch.linalg.qr(Y)
    B = torch.mm(Q.T.conj(), M)
    
    try:
        U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    except RuntimeError:
        U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False, driver='gesvd')
        
    U = torch.mm(Q, U_hat)
    return U, S, Vh


GATE_CACHE = {}


class MPOTensorGPU:
    def __init__(self, num_qubits: int, device=None):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = DEVICE
        self.num_qubits = num_qubits
        self.tensors: List[torch.Tensor] = []
        self._init_identity()
    
    def _init_identity(self):
        # Identity MPO: tensor[i,j,0,0] = delta[i,j]
        for _ in range(self.num_qubits):
            tensor = torch.zeros((2, 2, 1, 1), dtype=DTYPE, device=self.device)
            tensor[0, 0, 0, 0] = 1.0
            tensor[1, 1, 0, 0] = 1.0
            self.tensors.append(tensor)
    
    def to_matrix(self) -> torch.Tensor:
        result = self.tensors[0]
        for i in range(1, len(self.tensors)):
            result = torch.einsum('ijkl,mnlo->ijmnko', result, self.tensors[i])
            shape = result.shape
            result = result.reshape(shape[0]*shape[2], shape[1]*shape[3], shape[4], shape[5])
        return result.squeeze()
    
    def trace_efficient(self) -> torch.Tensor:
        result = None
        for tensor in self.tensors:
            traced = torch.einsum('iijk->jk', tensor).to(DTYPE)
            if result is None:
                result = traced
            else:
                result = torch.matmul(result.to(DTYPE), traced.to(DTYPE))
        return result.squeeze()


def decompose_theta_gpu_batched(theta_batch: torch.Tensor, threshold: float, 
                                max_bond_dim: int = None,
                                original_shapes: List[Tuple[int, int]] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dims = theta_batch.shape
    B = dims[0]
    device = theta_batch.device
    
    theta = theta_batch.permute(0, 1, 3, 4, 2, 5, 6)
    theta_matrix = theta.reshape(B, dims[1]*dims[3]*dims[4], dims[2]*dims[5]*dims[6])
    
    m, n = theta_matrix.shape[1], theta_matrix.shape[2]
    use_rsvd = (max_bond_dim is not None) and (min(m, n) > 2 * max_bond_dim)
    
    U, S, Vh = None, None, None
    if use_rsvd:
        try:
             target_rank = min(max_bond_dim + 16, min(m, n))
             Omega = torch.randn(B, n, target_rank, dtype=DTYPE, device=device)
             Y = torch.bmm(theta_matrix, Omega)
             Q, _ = torch.linalg.qr(Y)
             B_mat = torch.bmm(Q.transpose(1, 2).conj(), theta_matrix)
             U_hat, S, Vh = torch.linalg.svd(B_mat, full_matrices=False)
             U = torch.bmm(Q, U_hat)
        except Exception:
             use_rsvd = False
             
    if not use_rsvd:
        try:
            if device.type == 'cuda':
                U, S, Vh = torch.linalg.svd(theta_matrix, full_matrices=False, driver='gesvd')
            else:
                U, S, Vh = torch.linalg.svd(theta_matrix, full_matrices=False)
        except RuntimeError:
            if device.type == 'cuda':
                U, S, Vh = torch.linalg.svd(theta_matrix, full_matrices=False, driver='gesvd')
            else:
                U, S, Vh = torch.linalg.svd(theta_matrix, full_matrices=False)
        
    U = U.to(DTYPE)
    S = S.to(torch.float32)
    Vh = Vh.to(DTYPE)
    
    results = []
    for i in range(B):
        s_local = S[i]
        mask = s_local > threshold
        num_sv = max(1, min(mask.sum().item(), max_bond_dim or 1000000))
        
        u_local = U[i, :, :num_sv]
        vh_local = Vh[i, :num_sv, :]
        
        if original_shapes:
            dL, dR = original_shapes[i]
        else:
            dL, dR = dims[3], dims[6]
            
        u_tensor = u_local.reshape(dims[1], dims[3], dims[4], num_sv)
        u_tensor = u_tensor[:, :dL, :, :]
        u_tensor = u_tensor.permute(0, 2, 1, 3).contiguous()
        
        m_mat = torch.diag(s_local[:num_sv].to(DTYPE)) @ vh_local
        m_tensor = m_mat.reshape(num_sv, dims[2], dims[5], dims[6])
        m_tensor = m_tensor[:, :, :, :dR]
        m_tensor = m_tensor.permute(1, 2, 0, 3).contiguous()
        
        results.append((u_tensor, m_tensor))
        
    return results


def decompose_theta_gpu(theta: torch.Tensor, threshold: float, 
                        max_bond_dim: int = None,
                        use_randomized_svd: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    dims = theta.shape
    device = theta.device
    
    theta = theta.permute(0, 3, 2, 1, 4, 5)
    theta_matrix = theta.reshape(dims[0] * dims[1] * dims[2], dims[3] * dims[4] * dims[5])
    
    U, S, Vh = None, None, None
    m, n = theta_matrix.shape
    
    effective_rsvd = use_randomized_svd and (max_bond_dim is not None) and (min(m, n) > 2 * max_bond_dim)

    if effective_rsvd:
        try:
             U, S, Vh = randomized_svd_gpu(theta_matrix, rank=max_bond_dim)
        except Exception:
             effective_rsvd = False

    if not effective_rsvd:
        try:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings('always', category=UserWarning)
                    if device.type == 'cuda':
                        U, S, Vh = torch.linalg.svd(theta_matrix, driver='gesvdj')
                    else:
                         U, S, Vh = torch.linalg.svd(theta_matrix)
                    for warning in w:
                        if "failed to converge" in str(warning.message):
                             raise RuntimeError("gesvdj failed to converge")
            except (RuntimeError, torch._C._LinAlgError):
                if device.type == 'cuda':
                    U, S, Vh = torch.linalg.svd(theta_matrix, driver='gesvd')
                else:
                    U, S, Vh = torch.linalg.svd(theta_matrix)
        except Exception as e:
            print(f"!! CRITICAL: GPU SVD Failed ({e}). Falling back to CPU !!!")
            theta_np = theta_matrix.cpu().numpy()
            U_np, S_np, Vh_np = np.linalg.svd(theta_np, full_matrices=False)
            U = torch.tensor(U_np, device=device)
            S = torch.tensor(S_np, device=device)
            Vh = torch.tensor(Vh_np, device=device)

        if U is not None and (torch.isnan(U).any() or torch.isnan(S).any() or torch.isnan(Vh).any()):
             print("!! CRITICAL: SVD returned NaNs! Falling back to CPU !!")
             theta_np = theta_matrix.cpu().numpy()
             U_np, S_np, Vh_np = np.linalg.svd(theta_np, full_matrices=False)
             U = torch.tensor(U_np, device=device)
             S = torch.tensor(S_np, device=device)
             Vh = torch.tensor(Vh_np, device=device)

    U = U.to(DTYPE)
    S = S.to(torch.float64)
    Vh = Vh.to(DTYPE)

    mask = S > threshold
    num_sv = max(1, mask.sum().item())
    if max_bond_dim is not None:
        num_sv = min(num_sv, max_bond_dim)
    
    U = U[:, :num_sv]
    S = S[:num_sv]
    Vh = Vh[:num_sv, :]
    
    U_tensor = U.reshape(dims[0], dims[1], dims[2], num_sv)
    
    S_complex = S.to(DTYPE)
    M_mat = torch.diag(S_complex) @ Vh
    M_tensor = M_mat.reshape(num_sv, dims[3], dims[4], dims[5])
    M_tensor = M_tensor.permute(1, 2, 0, 3)
    
    return U_tensor, M_tensor


def gate_to_tensor_gpu(gate_matrix: np.ndarray, device=None) -> torch.Tensor:
    device = device or DEVICE
    key = (gate_matrix.tobytes(), device)
    
    if key in GATE_CACHE:
        return GATE_CACHE[key]
    
    if len(GATE_CACHE) > 1024:
        GATE_CACHE.clear()

    tensor = torch.tensor(gate_matrix, dtype=DTYPE, device=device)
    GATE_CACHE[key] = tensor
    return tensor


def apply_single_gate_gpu(theta: torch.Tensor, gate: torch.Tensor, site: int, 
                          conjugate: bool = False) -> torch.Tensor:
    if conjugate:
        gate = torch.conj(gate)
    
    if site == 0:
        theta = torch.einsum('ij,jklmno->iklmno', gate, theta)
    else:
        theta = torch.einsum('ij,kjlmno->kilmno', gate, theta)
    
    return theta


def apply_two_qubit_gate_gpu(theta: torch.Tensor, gate: torch.Tensor,
                              conjugate: bool = False) -> torch.Tensor:
    if conjugate:
        gate = torch.conj(gate)
    
    theta = torch.einsum('ijkl,klmnop->ijmnop', gate, theta)
    return theta


def update_mpo_gpu(mpo: MPOTensorGPU, gates1: List[Tuple], gates2: List[Tuple],
                   site: int, threshold: float):
    n = site
    
    # Contract neighboring MPO tensors: theta = T_n ⊗ T_{n+1}
    theta = torch.einsum('abcd,efdg->aecbfg', mpo.tensors[n], mpo.tensors[n + 1])
    
    for gate_matrix, target in gates1:
        gate = gate_to_tensor_gpu(gate_matrix, mpo.device)
        if gate_matrix.shape == (2, 2):
            local_site = 0 if target == n else 1
            theta = apply_single_gate_gpu(theta, gate, local_site, conjugate=False)
        elif gate_matrix.shape == (4, 4):
            gate_tensor = gate.reshape(2, 2, 2, 2)
            theta = apply_two_qubit_gate_gpu(theta, gate_tensor, conjugate=False)
    
    for gate_matrix, target in gates2:
        gate = gate_to_tensor_gpu(gate_matrix, mpo.device)
        theta = theta.permute(3, 4, 2, 0, 1, 5)
        if gate_matrix.shape == (2, 2):
            local_site = 0 if target == n else 1
            theta = apply_single_gate_gpu(theta, gate, local_site, conjugate=True)
        elif gate_matrix.shape == (4, 4):
            gate_tensor = gate.reshape(2, 2, 2, 2)
            theta = apply_two_qubit_gate_gpu(theta, gate_tensor, conjugate=True)
        theta = theta.permute(3, 4, 2, 0, 1, 5)
    
    mpo.tensors[n], mpo.tensors[n + 1] = decompose_theta_gpu(theta, threshold)


def compute_theta_gpu(mpo: MPOTensorGPU, gates1: List[Tuple], gates2: List[Tuple], site: int):
    n = site
    
    if MPO_CUDA is not None and mpo.device.type == 'cuda':
        g1 = get_consolidated_gate_gpu(gates1, n, mpo.device, conjugate=False)
        g2_conj = get_consolidated_gate_gpu(gates2, n, mpo.device, conjugate=True)
        
        theta = MPO_CUDA.update_theta(
            mpo.tensors[n].contiguous(), 
            mpo.tensors[n+1].contiguous(), 
            g1.contiguous(), 
            g2_conj.contiguous()
        )
    else:
        theta = torch.einsum('abcd,efdg->aecbfg', mpo.tensors[n], mpo.tensors[n + 1])
        for gate_data in gates1:
            theta = apply_gate_to_theta_gpu(theta, gate_data, n, n + 1, conjugate=False, device=mpo.device)
        for gate_data in gates2:
            theta = apply_gate_to_theta_gpu(theta, gate_data, n, n + 1, conjugate=True, device=mpo.device)
            
    return theta


def select_starting_point_gpu(num_qubits: int, dag: DAGCircuit) -> Tuple[range, range]:
    first_iterator = range(0, num_qubits - 1, 2)
    second_iterator = range(1, num_qubits - 1, 2)
    
    first_layer = next(dag.layers(), None)
    if first_layer is not None:
        layer_circuit = dag_to_circuit(first_layer["graph"])
        for gate in layer_circuit.data:
            if gate.operation.num_qubits == 2:
                if gate.qubits[0]._index % 2 != 0:
                    first_iterator = range(1, num_qubits - 1, 2)
                    second_iterator = range(0, num_qubits - 1, 2)
                break
    
    return first_iterator, second_iterator


def iterate_gpu(mpo: MPOTensorGPU, circuit1: QuantumCircuit, circuit2: QuantumCircuit,
                threshold: float = 1e-13, use_randomized_svd: bool = False,
                use_batched_svd: bool = False, max_bond_dim: int = None) -> None:
    num_qubits = mpo.num_qubits
    
    dag1 = circuit_to_dag(circuit1)
    dag2 = circuit_to_dag(circuit2)
    
    if list(dag1.op_nodes()):
        first_iterator, second_iterator = select_starting_point_gpu(num_qubits, dag1)
    else:
        first_iterator, second_iterator = select_starting_point_gpu(num_qubits, dag2)
    
    max_iterations = circuit1.depth() + circuit2.depth() + 20
    iter_gate_batches = []
    
    for _ in range(max_iterations):
        if not list(dag1.op_nodes()) and not list(dag2.op_nodes()):
            break
        for iterator in [first_iterator, second_iterator]:
            batch_tasks = []
            for n in iterator:
                gates1 = extract_gates_from_dag(dag1, n, n + 1)
                gates2 = extract_gates_from_dag(dag2, n, n + 1)
                if gates1 or gates2:
                    batch_tasks.append((n, gates1, gates2))
            if batch_tasks:
                iter_gate_batches.append(batch_tasks)

    if use_batched_svd and mpo.device.type == 'cuda':
        for batch_tasks in iter_gate_batches:
            thetas = []
            for n, g1, g2 in batch_tasks:
                theta = compute_theta_gpu(mpo, g1, g2, n)
                thetas.append((n, theta))
            if not thetas: continue
            max_dL = max(t[1].shape[2] for t in thetas)
            max_dR = max(t[1].shape[5] for t in thetas)
            padded_thetas, site_indices, original_shapes = [], [], []
            for n, theta in thetas:
                dL, dR = theta.shape[2], theta.shape[5]
                original_shapes.append((dL, dR))
                padding = (0, max_dR - dR, 0, 0, 0, 0, 0, max_dL - dL)
                padded_thetas.append(torch.nn.functional.pad(theta, padding))
                site_indices.append(n)
            results = decompose_theta_gpu_batched(torch.stack(padded_thetas), threshold, 
                                                  max_bond_dim=max_bond_dim,
                                                  original_shapes=original_shapes)
            for i, (u, m) in enumerate(results):
                site_idx = site_indices[i]
                mpo.tensors[site_idx], mpo.tensors[site_idx+1] = u, m
    elif mpo.device.type == 'cuda':
        if not hasattr(mpo, 'streams'):
             mpo.streams = [torch.cuda.Stream(device=mpo.device) for _ in range(8)]
        for batch_tasks in iter_gate_batches:
            for idx, (n, g1, g2) in enumerate(batch_tasks):
                stream = mpo.streams[idx % len(mpo.streams)]
                with torch.cuda.stream(stream):
                    update_mpo_sites_gpu(mpo, g1, g2, n, threshold, use_randomized_svd, max_bond_dim=max_bond_dim)
            torch.cuda.synchronize(mpo.device)
    else:
        for batch_tasks in iter_gate_batches:
            for n, g1, g2 in batch_tasks:
                update_mpo_sites_gpu(mpo, g1, g2, n, threshold, use_randomized_svd, max_bond_dim=max_bond_dim)


def extract_gates_from_dag(dag: DAGCircuit, site0: int, site1: int) -> List[Tuple]:
    from mqt.yaqs.core.libraries.gate_library import GateLibrary
    
    gates = []
    qubits_to_check = {dag.qubits[site0], dag.qubits[site1]}
    
    layers = list(dag.multigraph_layers())
    
    for layer in layers:
        for node in layer:
            if isinstance(node, DAGOpNode):
                qubit_set = set(node.qargs)
                
                if qubit_set <= qubits_to_check:
                    if node.op.name in {"measure", "barrier"}:
                        dag.remove_op_node(node)
                        continue
                    
                    try:
                        name = node.op.name
                        attr = getattr(GateLibrary, name)
                        gate_obj = attr(node.op.params) if node.op.params else attr()
                        
                        qubit_indices = [q._index for q in node.qargs]
                        gate_obj.set_sites(*qubit_indices)
                        
                        if gate_obj.interaction == 1:
                            matrix = gate_obj.matrix
                        else:
                            matrix = gate_obj.tensor
                        
                        target = qubit_indices[0]
                        gates.append((matrix, target, gate_obj.interaction, qubit_indices))
                        dag.remove_op_node(node)
                    except Exception:
                        dag.remove_op_node(node)
                        continue
                else:
                    if node.op.name in {"measure", "barrier"}:
                        dag.remove_op_node(node)
                        continue
                    for item in qubit_set & qubits_to_check:
                        qubits_to_check.discard(item)
        
        if len(qubits_to_check) == 0:
            break
    
    return gates


def apply_gate_to_theta_gpu(theta: torch.Tensor, gate_data: Tuple,
                            site0: int, site1: int,
                            conjugate: bool, device) -> torch.Tensor:
    matrix, target, interaction, qubit_indices = gate_data
    gate = torch.tensor(matrix, dtype=DTYPE, device=device)
    
    if conjugate:
        theta = theta.permute(3, 4, 2, 0, 1, 5)
    
    if interaction == 1:
        gate_site = qubit_indices[0]
        if conjugate:
            gate = torch.conj(gate)
        if gate_site == site0:
            theta = torch.einsum('ij,jklmno->iklmno', gate, theta)
        elif gate_site == site1:
            theta = torch.einsum('ij,kjlmno->kilmno', gate, theta)
    elif interaction == 2:
        if conjugate:
            gate = torch.conj(gate)
        theta = torch.einsum('ijkl,klmnop->ijmnop', gate, theta)
    
    if conjugate:
        theta = theta.permute(3, 4, 2, 0, 1, 5)
    
    return theta


def get_consolidated_gate_gpu(gates: List[Tuple], site0: int, device: torch.device, conjugate: bool = False) -> torch.Tensor:
    I = torch.eye(2, dtype=DTYPE, device=device)
    total_gate = torch.eye(4, dtype=DTYPE, device=device)
    
    for gate_data in gates:
        matrix, target, interaction, qubit_indices = gate_data
        gate = torch.tensor(matrix, dtype=DTYPE, device=device)
        if conjugate:
            gate = torch.conj(gate)
        
        if interaction == 1:
            if target == site0:
                g4 = torch.kron(gate, I)
            else:
                g4 = torch.kron(I, gate)
        else:
            g4 = gate.reshape(4, 4)
            
        total_gate = g4 @ total_gate
        
    return total_gate


def apply_consolidated_gate(theta: torch.Tensor, gate_matrix: torch.Tensor, site0: int, site1: int, conjugate: bool = False) -> torch.Tensor:
    gate_tensor = gate_matrix.reshape(2, 2, 2, 2)
    
    if not conjugate:
        res = torch.einsum('abcd,cefdgh->abefgh', gate_tensor, theta)
        return res.permute(0, 2, 3, 1, 4, 5)
    else:
        res = torch.einsum('abcd,ecfgdh->eabfgh', gate_tensor, theta)
        return res.permute(0, 1, 3, 4, 2, 5)


def update_mpo_sites_gpu(mpo: MPOTensorGPU, gates1: List[Tuple], gates2: List[Tuple],
                         site: int, threshold: float, use_randomized_svd: bool = False,
                         max_bond_dim: int = None):
    n = site
    
    if MPO_CUDA is not None and mpo.device.type == 'cuda':
        g1 = get_consolidated_gate_gpu(gates1, n, mpo.device, conjugate=False)
        g2_conj = get_consolidated_gate_gpu(gates2, n, mpo.device, conjugate=True)
        
        if torch.isnan(mpo.tensors[n]).any() or torch.isnan(mpo.tensors[n+1]).any():
            raise RuntimeError("Input MPO tensors contain NaNs")
        if torch.isnan(g1).any() or torch.isnan(g2_conj).any():
            raise RuntimeError("Consolidated gates contain NaNs")
            
        theta = MPO_CUDA.update_theta(
            mpo.tensors[n].contiguous(), 
            mpo.tensors[n+1].contiguous(), 
            g1.contiguous(), 
            g2_conj.contiguous()
        )
        
        if torch.isnan(theta).any() or torch.isinf(theta).any():
            # Fallback to einsum
            theta = torch.einsum('abcd,efdg->aecbfg', mpo.tensors[n], mpo.tensors[n + 1])
            gate_tensor = get_consolidated_gate_gpu(gates1, n, mpo.device, conjugate=False)
            theta = apply_consolidated_gate(theta, gate_tensor, n, n+1, conjugate=False)
            gate_tensor_2 = get_consolidated_gate_gpu(gates2, n, mpo.device, conjugate=True)
            theta = apply_consolidated_gate(theta, gate_tensor_2, n, n+1, conjugate=True)
            
            if torch.isnan(theta).any():
                 # CPU rescue
                 t1_cpu = mpo.tensors[n].cpu().double()
                 t2_cpu = mpo.tensors[n+1].cpu().double()
                 theta_cpu = torch.einsum('abcd,efdg->aecbfg', t1_cpu, t2_cpu)
                 g1_cpu = get_consolidated_gate_gpu(gates1, n, torch.device('cpu'), conjugate=False).double()
                 g2_cpu = get_consolidated_gate_gpu(gates2, n, torch.device('cpu'), conjugate=True).double()
                 theta_cpu = apply_consolidated_gate(theta_cpu, g1_cpu, n, n+1, conjugate=False)
                 theta_cpu = apply_consolidated_gate(theta_cpu, g2_cpu, n, n+1, conjugate=True)
                 theta = theta_cpu.to(mpo.device)
                 if torch.isnan(theta).any():
                      raise RuntimeError("CPU rescue failed: Theta still contains NaNs")
    else:
        theta = torch.einsum('abcd,efdg->aecbfg', mpo.tensors[n], mpo.tensors[n + 1])
        
        for gate_data in gates1:
            theta = apply_gate_to_theta_gpu(theta, gate_data, n, n + 1, 
                                            conjugate=False, device=mpo.device)
        
        for gate_data in gates2:
            theta = apply_gate_to_theta_gpu(theta, gate_data, n, n + 1,
                                            conjugate=True, device=mpo.device)
    
    mpo.tensors[n], mpo.tensors[n + 1] = decompose_theta_gpu(theta, threshold, 
                                                             use_randomized_svd=use_randomized_svd,
                                                             max_bond_dim=max_bond_dim)


def get_fidelity_gpu(circuit1: QuantumCircuit, circuit2: QuantumCircuit,
                     threshold: float = 1e-13,
                     transpile_to_linear: bool = True,
                     device = None,
                     use_randomized_svd: bool = False,
                     use_batched_svd: bool = False,
                     max_bond_dim: int = None) -> dict:
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    
    assert circuit1.num_qubits == circuit2.num_qubits
    
    num_qubits = circuit1.num_qubits
    
    if device is None:
        device = get_device()
    
    basis = ['cx', 'h', 'rz', 'sx', 'x']
    
    if transpile_to_linear:
        coupling = CouplingMap.from_line(num_qubits)
        circuit1 = transpile(circuit1, coupling_map=coupling, basis_gates=basis, optimization_level=0)
        circuit2 = transpile(circuit2, coupling_map=coupling, basis_gates=basis, optimization_level=0)
    else:
        circuit1 = transpile(circuit1, basis_gates=basis, optimization_level=0)
        circuit2 = transpile(circuit2, basis_gates=basis, optimization_level=0)
    
    mpo = MPOTensorGPU(num_qubits, device)
    
    iterate_gpu(mpo, circuit1, circuit2, threshold, use_randomized_svd=use_randomized_svd,
                use_batched_svd=use_batched_svd, max_bond_dim=max_bond_dim)
    
    trace = mpo.trace_efficient()
    dimension = 2 ** num_qubits
    fidelity = torch.abs(trace).item() / dimension
    
    return {
        'fidelity': fidelity,
        'equivalent': fidelity > 0.9999,
        'device': str(device)
    }


def get_fidelity_multi_gpu(circuit1: QuantumCircuit, circuit2: QuantumCircuit,
                           threshold: float = 1e-13,
                           devices: List[str] = None,
                           use_randomized_svd: bool = False) -> dict:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if devices is None:
        devices = ['cuda:1', 'cuda:3']
    
    n_gpus = len(devices)
    num_qubits = circuit1.num_qubits
    
    basis = ['cx', 'h', 'rz', 'sx', 'x']
    coupling_map = CouplingMap.from_line(num_qubits)
    circuit1 = transpile(circuit1, coupling_map=coupling_map, basis_gates=basis, optimization_level=0)
    circuit2 = transpile(circuit2, coupling_map=coupling_map, basis_gates=basis, optimization_level=0)
    
    dag1 = circuit_to_dag(circuit1)
    dag2 = circuit_to_dag(circuit2)
    
    if list(dag1.op_nodes()):
        first_iterator, second_iterator = select_starting_point_gpu(num_qubits, dag1)
    else:
        first_iterator, second_iterator = select_starting_point_gpu(num_qubits, dag2)
    
    sites_per_gpu = num_qubits // n_gpus
    gpu_ranges = [i * sites_per_gpu for i in range(n_gpus)] + [num_qubits]
    remainder = num_qubits % n_gpus
    for i in range(1, n_gpus + 1):
        gpu_ranges[i] += min(i, remainder)
    
    def get_gpu_for_site(site: int) -> int:
        for gpu_idx in range(n_gpus):
            if gpu_ranges[gpu_idx] <= site < gpu_ranges[gpu_idx + 1]:
                return gpu_idx
        return n_gpus - 1
    
    mpo_tensors = []
    for i in range(num_qubits):
        gpu_idx = get_gpu_for_site(i)
        device = torch.device(devices[gpu_idx])
        tensor = torch.zeros((2, 2, 1, 1), dtype=DTYPE, device=device)
        tensor[0, 0, 0, 0] = 1.0
        tensor[1, 1, 0, 0] = 1.0
        mpo_tensors.append(tensor)
    
    max_iterations = circuit1.depth() + circuit2.depth() + 10
    
    dag_lock = threading.Lock()
    
    def process_gpu_sites(gpu_idx: int, sites: List[int]):
        for n in sites:
            with dag_lock:
                gates1 = extract_gates_from_dag(dag1, n, n + 1)
                gates2 = extract_gates_from_dag(dag2, n, n + 1)
            
            if not gates1 and not gates2:
                continue
            
            device = torch.device(devices[gpu_idx])
            t1 = mpo_tensors[n].to(device)
            t2 = mpo_tensors[n + 1].to(device)
            
            theta = torch.einsum('abcd,efdg->aecbfg', t1, t2)
            
            for gate_data in gates1:
                theta = apply_gate_to_theta_gpu(theta, gate_data, n, n + 1,
                                                conjugate=False, device=device)
            
            for gate_data in gates2:
                theta = apply_gate_to_theta_gpu(theta, gate_data, n, n + 1,
                                                conjugate=True, device=device)
            
            u, m = decompose_theta_gpu(theta, threshold, use_randomized_svd=use_randomized_svd)
            
            mpo_tensors[n] = u.to(devices[get_gpu_for_site(n)])
            mpo_tensors[n + 1] = m.to(devices[get_gpu_for_site(n + 1)])
        
        torch.cuda.synchronize(devices[gpu_idx])
    
    for iteration in range(max_iterations):
        if not list(dag1.op_nodes()) and not list(dag2.op_nodes()):
            break
        
        even_sites_by_gpu = [[] for _ in range(n_gpus)]
        for n in first_iterator:
            gpu_idx = get_gpu_for_site(n)
            even_sites_by_gpu[gpu_idx].append(n)
        
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = []
            for gpu_idx in range(n_gpus):
                if even_sites_by_gpu[gpu_idx]:
                    futures.append(executor.submit(process_gpu_sites, gpu_idx, even_sites_by_gpu[gpu_idx]))
            for f in futures:
                f.result()
        
        for dev in devices:
            torch.cuda.synchronize(dev)
        
        odd_sites_by_gpu = [[] for _ in range(n_gpus)]
        for n in second_iterator:
            gpu_idx = get_gpu_for_site(n)
            odd_sites_by_gpu[gpu_idx].append(n)
        
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = []
            for gpu_idx in range(n_gpus):
                if odd_sites_by_gpu[gpu_idx]:
                    futures.append(executor.submit(process_gpu_sites, gpu_idx, odd_sites_by_gpu[gpu_idx]))
            for f in futures:
                f.result()
        
        for dev in devices:
            torch.cuda.synchronize(dev)
    
    device0 = torch.device(devices[0])
    tensors_on_gpu0 = [t.to(device0) for t in mpo_tensors]
    
    result = None
    for tensor in tensors_on_gpu0:
        traced = torch.einsum('iijk->jk', tensor).to(DTYPE)
        if result is None:
            result = traced
        else:
            result = torch.matmul(result.to(DTYPE), traced.to(DTYPE))
    
    trace = result.squeeze()
    dimension = 2 ** num_qubits
    fidelity = torch.abs(trace).item() / dimension
    
    return {
        'fidelity': fidelity,
        'equivalent': fidelity > 0.9999,
        'devices': devices
    }


def get_fidelity_multi_gpu_v2(circuit1: QuantumCircuit, circuit2: QuantumCircuit,
                              threshold: float = 1e-13,
                              devices: List[str] = None,
                              use_randomized_svd: bool = False) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from copy import deepcopy
    
    if devices is None:
        devices = ['cuda:0', 'cuda:2', 'cuda:3']
    
    n_gpus = len(devices)
    num_qubits = circuit1.num_qubits
    
    basis = ['cx', 'h', 'rz', 'sx', 'x']
    coupling_map = CouplingMap.from_line(num_qubits)
    circuit1 = transpile(circuit1, coupling_map=coupling_map, basis_gates=basis, optimization_level=0)
    circuit2 = transpile(circuit2, coupling_map=coupling_map, basis_gates=basis, optimization_level=0)
    
    dag1 = circuit_to_dag(circuit1)
    dag2 = circuit_to_dag(circuit2)
    
    if list(dag1.op_nodes()):
        first_iterator, second_iterator = select_starting_point_gpu(num_qubits, dag1)
    else:
        first_iterator, second_iterator = select_starting_point_gpu(num_qubits, dag2)
    
    # Pre-extract all gates before parallel execution
    max_iterations = circuit1.depth() + circuit2.depth() + 10
    all_gates_per_iteration = []
    
    for iteration in range(max_iterations):
        if not list(dag1.op_nodes()) and not list(dag2.op_nodes()):
            break
        
        iteration_gates = {}
        
        for n in first_iterator:
            gates1 = extract_gates_from_dag(dag1, n, n + 1)
            gates2 = extract_gates_from_dag(dag2, n, n + 1)
            if gates1 or gates2:
                iteration_gates[('even', n)] = (gates1, gates2)
        
        for n in second_iterator:
            gates1 = extract_gates_from_dag(dag1, n, n + 1)
            gates2 = extract_gates_from_dag(dag2, n, n + 1)
            if gates1 or gates2:
                iteration_gates[('odd', n)] = (gates1, gates2)
        
        if iteration_gates:
            all_gates_per_iteration.append(iteration_gates)
    
    sites_per_gpu = num_qubits // n_gpus
    gpu_ranges = [i * sites_per_gpu for i in range(n_gpus)] + [num_qubits]
    remainder = num_qubits % n_gpus
    for i in range(1, n_gpus + 1):
        gpu_ranges[i] += min(i, remainder)
    
    def get_gpu_for_site(site: int) -> int:
        for gpu_idx in range(n_gpus):
            if gpu_ranges[gpu_idx] <= site < gpu_ranges[gpu_idx + 1]:
                return gpu_idx
        return n_gpus - 1
    
    mpo_tensors = []
    for i in range(num_qubits):
        gpu_idx = get_gpu_for_site(i)
        device = torch.device(devices[gpu_idx])
        tensor = torch.zeros((2, 2, 1, 1), dtype=DTYPE, device=device)
        tensor[0, 0, 0, 0] = 1.0
        tensor[1, 1, 0, 0] = 1.0
        mpo_tensors.append(tensor)
    
    def process_site_with_gates(n: int, gates1: List, gates2: List, gpu_idx: int):
        device = torch.device(devices[gpu_idx])
        
        t1 = mpo_tensors[n].to(device)
        t2 = mpo_tensors[n + 1].to(device)
        
        theta = torch.einsum('abcd,efdg->aecbfg', t1, t2)
        
        for gate_data in gates1:
            theta = apply_gate_to_theta_gpu(theta, gate_data, n, n + 1,
                                            conjugate=False, device=device)
        
        for gate_data in gates2:
            theta = apply_gate_to_theta_gpu(theta, gate_data, n, n + 1,
                                            conjugate=True, device=device)
        
        u, m = decompose_theta_gpu(theta, threshold, use_randomized_svd=use_randomized_svd)
        
        mpo_tensors[n] = u.to(devices[get_gpu_for_site(n)])
        mpo_tensors[n + 1] = m.to(devices[get_gpu_for_site(n + 1)])
    
    def process_gpu_sites_parallel(gpu_idx: int, sites_with_gates: List):
        for n, gates1, gates2 in sites_with_gates:
            process_site_with_gates(n, gates1, gates2, gpu_idx)
        torch.cuda.synchronize(devices[gpu_idx])
    
    for iteration_gates in all_gates_per_iteration:
        even_by_gpu = [[] for _ in range(n_gpus)]
        for key, (gates1, gates2) in iteration_gates.items():
            if key[0] == 'even':
                n = key[1]
                gpu_idx = get_gpu_for_site(n)
                even_by_gpu[gpu_idx].append((n, gates1, gates2))
        
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = [executor.submit(process_gpu_sites_parallel, gpu_idx, even_by_gpu[gpu_idx])
                       for gpu_idx in range(n_gpus) if even_by_gpu[gpu_idx]]
            for f in futures:
                f.result()
        
        for dev in devices:
            torch.cuda.synchronize(dev)
        
        odd_by_gpu = [[] for _ in range(n_gpus)]
        for key, (gates1, gates2) in iteration_gates.items():
            if key[0] == 'odd':
                n = key[1]
                gpu_idx = get_gpu_for_site(n)
                odd_by_gpu[gpu_idx].append((n, gates1, gates2))
        
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = [executor.submit(process_gpu_sites_parallel, gpu_idx, odd_by_gpu[gpu_idx])
                       for gpu_idx in range(n_gpus) if odd_by_gpu[gpu_idx]]
            for f in futures:
                f.result()
        
        for dev in devices:
            torch.cuda.synchronize(dev)
    
    device0 = torch.device(devices[0])
    tensors_on_gpu0 = [t.to(device0) for t in mpo_tensors]
    
    result = None
    for tensor in tensors_on_gpu0:
        traced = torch.einsum('iijk->jk', tensor).to(DTYPE)
        if result is None:
            result = traced
        else:
            result = torch.matmul(result.to(DTYPE), traced.to(DTYPE))
    
    trace = result.squeeze()
    dimension = 2 ** num_qubits
    fidelity = torch.abs(trace).item() / dimension
    
    return {
        'fidelity': fidelity,
        'equivalent': fidelity > 0.9999,
        'devices': devices
    }


if __name__ == "__main__":
    from qiskit.circuit.random import random_circuit
    from qiskit import transpile
    import time
    
    BASIS_GATES = ['cx', 'h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
    
    print(f"Device: {get_device()}")
    print()
    
    print("Testing GPU MPO implementation:")
    print(f"{'Qubits':<8} {'Time (s)':<12} {'Fidelity':<12}")
    print("-" * 35)
    
    for n in [3, 4, 5, 6, 7, 8]:
        c1 = transpile(random_circuit(n, 5, seed=42), basis_gates=BASIS_GATES, optimization_level=0)
        c2 = transpile(random_circuit(n, 5, seed=43), basis_gates=BASIS_GATES, optimization_level=0)
        
        t0 = time.time()
        try:
            result = get_fidelity_gpu(c1, c2)
            elapsed = time.time() - t0
            print(f"{n:<8} {elapsed:<12.3f} {result['fidelity']:<12.6f}")
        except Exception as e:
            print(f"{n:<8} ERROR: {e}")
