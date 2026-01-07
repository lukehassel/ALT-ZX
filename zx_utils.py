import argparse
import torch
import numpy as np
import pyzx as zx
from tqdm import tqdm
from pathlib import Path
import random
from typing import Optional, Tuple
from zx_loader import pyzx_graph_to_pyg, zxnet_qasm_to_pyg, pyg_to_pyzx


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


def apply_random_zx_transform(graph: zx.Graph) -> zx.Graph:
    g = graph.copy()
    vertices = list(g.vertices())
    inputs_set = set(g.inputs())
    outputs_set = set(g.outputs())
    internal = [v for v in vertices if v not in inputs_set and v not in outputs_set]
    
    if len(internal) < 1:
        return g
    
    transform_type = random.choice(['unfuse_spider', 'insert_identity', 'insert_identity', 'insert_identity'])
    
    try:
        if transform_type == 'unfuse_spider':
            zx_spiders = [v for v in internal if g.type(v) in (zx.VertexType.Z, zx.VertexType.X)]
            candidates = [v for v in zx_spiders if len(list(g.neighbors(v))) >= 3]
            if candidates:
                v = random.choice(candidates)
                neighbors = list(g.neighbors(v))
                split_point = max(1, len(neighbors) // 2)
                new_v = g.add_vertex(g.type(v), qubit=g.qubit(v), 
                                    row=g.row(v) + 0.1, phase=0)
                for n in neighbors[:split_point]:
                    edge_type = g.edge_type(g.edge(v, n))
                    g.remove_edge(g.edge(v, n))
                    g.add_edge((new_v, n), edge_type)
                g.add_edge((v, new_v))
                
        elif transform_type == 'insert_identity':
            edges = list(g.edges())
            internal_edges = []
            for e in edges:
                s, t = g.edge_st(e)
                if s not in inputs_set and s not in outputs_set and t not in inputs_set and t not in outputs_set:
                    if g.type(s) in (zx.VertexType.Z, zx.VertexType.X) and \
                       g.type(t) in (zx.VertexType.Z, zx.VertexType.X):
                        internal_edges.append(e)
            if internal_edges:
                e = random.choice(internal_edges)
                s, t = g.edge_st(e)
                edge_type = g.edge_type(e)
                new_v = g.add_vertex(g.type(s), 
                                    qubit=(g.qubit(s) + g.qubit(t)) / 2,
                                    row=(g.row(s) + g.row(t)) / 2, 
                                    phase=0)
                g.remove_edge(e)
                g.add_edge((s, new_v), edge_type)
                g.add_edge((new_v, t), edge_type)
                
    except Exception:
        pass
    
    return g