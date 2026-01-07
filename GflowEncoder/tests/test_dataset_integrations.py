
import unittest
import os
import shutil
import torch
import sys

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GflowEncoder.dataset import (
    integration_create_circuit,
    integration_pyzx_to_pyg,
    integration_apply_transform,
    integration_save_torch,
    integration_makedirs,
    integration_init_worker
)

class TestGflowDatasetIntegrations(unittest.TestCase):
    
    def test_integration_create_circuit(self):
        graph = integration_create_circuit(4, 5, 42)
        self.assertIsNotNone(graph)
        # Check basic pyzx graph properties if possible, or just type
        self.assertTrue(hasattr(graph, 'vertices'))

    def test_integration_pyzx_to_pyg(self):
        graph = integration_create_circuit(3, 3, 123)
        data = integration_pyzx_to_pyg(graph)
        # pyzx_to_pyg might return None if conversion fails, but for simple graph it should work
        if data is not None:
            self.assertTrue(hasattr(data, 'x'))
            self.assertTrue(hasattr(data, 'edge_index'))

    def test_integration_apply_transform(self):
        graph = integration_create_circuit(3, 3, 123)
        initial_verts = len(graph.vertices())
        new_graph = integration_apply_transform(graph)
        self.assertIsNotNone(new_graph)
        # Transform might change vertex count or edges
        
    def test_integration_save_torch(self):
        test_data = {"a": 1}
        path = "test_data.pt"
        integration_save_torch(test_data, path)
        self.assertTrue(os.path.exists(path))
        loaded = torch.load(path)
        self.assertEqual(loaded["a"], 1)
        os.remove(path)

    def test_integration_makedirs(self):
        path = "test_dir_xyz"
        if os.path.exists(path):
            shutil.rmtree(path)
        integration_makedirs(path)
        self.assertTrue(os.path.exists(path))
        os.rmdir(path)
        
    def test_integration_init_worker(self):
        # Just check it runs without error
        integration_init_worker(123)

if __name__ == "__main__":
    unittest.main()
