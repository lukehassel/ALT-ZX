
import argparse
import torch
import numpy as np
import pyzx as zx
from tqdm import tqdm
from pathlib import Path

from model import ZXVGAE
from zx_loader import pyzx_to_pyg, qasm_to_pyg, pyg_to_pyzx


def check_gflow(graph: zx.Graph) -> bool:
    try:
        if hasattr(zx, 'gflow') and hasattr(zx.gflow, 'gflow'):
            return zx.gflow.gflow(graph)
        else:
            circuit = zx.extract.extract_circuit(graph)
            return True
    except:
        return False


def extract_circuit_gate_count(graph: zx.Graph) -> Tuple[Optional[zx.Circuit], int]:
    try:
        circuit = zx.extract.extract_circuit(graph)
        gate_count = len(circuit.gates)
        return circuit, gate_count
    except Exception as e:
        return None, float('inf')