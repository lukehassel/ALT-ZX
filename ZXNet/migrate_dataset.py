import os
import sys
import torch
from qiskit import QuantumCircuit, transpile

# Allow importing zx_loader from the repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from zx_loader import circuit_to_pyg  # noqa: E402


DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "test.pt")
OUTPUT_PATH = os.path.join(DATA_DIR, "train_pyg.pt")

# Basic gates that PyZX can parse
PYZX_BASIS_GATES = ['cx', 'cz', 'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'ccx', 'swap']


def qasm_to_circuit_transpiled(qasm_str: str) -> QuantumCircuit:
    """Load QASM string and transpile to basic gates PyZX understands."""
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    # Transpile to basic gate set
    qc_transpiled = transpile(qc, basis_gates=PYZX_BASIS_GATES, optimization_level=0)
    return qc_transpiled


def migrate_dataset(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
    """Convert QASM-based train.pt into PyG pair tuples compatible with ZXNet generation."""
    raw = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"Expected list/tuple in {input_path}, got {type(raw)}")

    pairs = []
    skipped = 0

    print(f"Migrating {len(raw)} items from {input_path}", flush=True)

    for idx, item in enumerate(raw):
        try:
            qasm_1 = item["circuit_1"]
            qasm_2 = item["circuit_2"]
            fidelity = float(item.get("fidelity", 0.5))

            # Load and transpile to basic gates
            qc1 = qasm_to_circuit_transpiled(qasm_1)
            qc2 = qasm_to_circuit_transpiled(qasm_2)

            data1 = circuit_to_pyg(qc1)
            data2 = circuit_to_pyg(qc2)

            # Store qubit counts on the PyG objects for downstream use
            data1.num_qubits = qc1.num_qubits
            data2.num_qubits = qc2.num_qubits

            label = torch.tensor(fidelity, dtype=torch.float32)
            pairs.append((data1, data2, label))

            if (idx + 1) % 25 == 0 or idx == len(raw) - 1:
                print(f"  processed {idx + 1}/{len(raw)}", flush=True)

        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"  Skip idx {idx}: {exc}", flush=True)

    torch.save(pairs, output_path)
    print(
        f"Done. Saved {len(pairs)} pairs to {output_path} (skipped {skipped}).",
        flush=True,
    )

    return pairs


if __name__ == "__main__":
    migrate_dataset()
