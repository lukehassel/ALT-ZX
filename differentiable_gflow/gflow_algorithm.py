import networkx as nx
from graphix.opengraph import OpenGraph
from graphix.fundamentals import Plane

def find_gflow(n, edges, inputs, outputs):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    
    outputs_set = set(outputs)
    measurements = {i: Plane.XY for i in range(n) if i not in outputs_set}
    
    try:
        og = OpenGraph(G, list(inputs), list(outputs), measurements)
    except Exception:
        return None
        
    return og.find_gflow()
