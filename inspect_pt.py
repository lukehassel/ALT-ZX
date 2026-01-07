#!/usr/bin/env python
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _print_item_detail(item):
    if hasattr(item, 'x') and hasattr(item, 'edge_index'):
        num_nodes = item.x.shape[0] if item.x is not None else (item.num_nodes if hasattr(item, 'num_nodes') else '?')
        num_edges = item.edge_index.shape[1] if item.edge_index is not None else 0
        print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
        for key in item.keys():
            val = item[key]
            if hasattr(val, 'shape'):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            elif hasattr(val, '__len__') and not isinstance(val, str):
                print(f"  {key}: len={len(val)}")
            else:
                print(f"  {key}: {val}")
    elif hasattr(item, 'shape'):
        print(f"  shape={item.shape}, dtype={item.dtype}")
    elif isinstance(item, dict):
        for k, v in item.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}")
            else:
                print(f"  {k}: {type(v).__name__}")
    else:
        print(f"  {item}")


def draw_pyg_graph(data, output_path, title=""):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    try:
        from zx_loader import reconstruct_pyzx_from_6feat
        import pyzx as zx
        
        graph = reconstruct_pyzx_from_6feat(data)
        
        if graph is None:
            print(f"  Warning: Could not convert graph for {output_path}")
            return False
        
        fig = zx.draw_matplotlib(graph, figsize=(14, 6), labels=False)
        if fig is not None:
            fig.suptitle(title, fontsize=14, fontweight='bold')
            fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
        return True
        
    except Exception as e:
        print(f"  Warning: Failed to draw graph: {e}")
        return False


def draw_comparison(data1, data2, output_path, title1="Before", title2="After"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    try:
        from zx_loader import reconstruct_pyzx_from_6feat
        import pyzx as zx
        
        graph1 = reconstruct_pyzx_from_6feat(data1)
        graph2 = reconstruct_pyzx_from_6feat(data2)
        
        if graph1 is None or graph2 is None:
            print(f"  Warning: Could not convert graphs for {output_path}")
            return False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        fig1 = zx.draw_matplotlib(graph1, figsize=(10, 8), labels=False)
        fig2 = zx.draw_matplotlib(graph2, figsize=(10, 8), labels=False)
        
        plt.close('all')
        
        fig1 = zx.draw_matplotlib(graph1, figsize=(12, 6), labels=False)
        if fig1:
            fig1.suptitle(title1, fontsize=14, fontweight='bold')
            path1 = output_path.replace('.png', '_before.png')
            fig1.savefig(path1, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig1)
            print(f"  Created: {path1}")
        
        fig2 = zx.draw_matplotlib(graph2, figsize=(12, 6), labels=False)
        if fig2:
            fig2.suptitle(title2, fontsize=14, fontweight='bold')
            path2 = output_path.replace('.png', '_after.png')
            fig2.savefig(path2, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig2)
            print(f"  Created: {path2}")
        
        return True
        
    except Exception as e:
        print(f"  Warning: Failed to draw comparison: {e}")
        return False


def create_dataset_images(data, output_dir="images", num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating images for first {num_samples} samples...")
    print(f"Output directory: {output_dir}")
    
    for i in range(min(num_samples, len(data))):
        item = data[i]
        
        if isinstance(item, tuple) and len(item) == 2 and hasattr(item[0], 'x') and hasattr(item[1], 'x'):
            corrupted, original = item
            
            orig_path = os.path.join(output_dir, f"sample_{i:02d}_original.png")
            orig_title = f"Sample {i} - Original (nodes={original.x.shape[0]})"
            if draw_pyg_graph(original, orig_path, orig_title):
                print(f"  Created: {orig_path}")
            
            corr_path = os.path.join(output_dir, f"sample_{i:02d}_corrupted.png")
            corr_title = f"Sample {i} - Corrupted (nodes={corrupted.x.shape[0]})"
            if draw_pyg_graph(corrupted, corr_path, corr_title):
                print(f"  Created: {corr_path}")
        
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
            graph_data, label = item
            label_str = "has_gflow" if label == 1 else "no_gflow"
            
            output_path = os.path.join(output_dir, f"sample_{i:02d}_{label_str}.png")
            title = f"Sample {i}: {label_str} (nodes={graph_data.x.shape[0]}, edges={graph_data.edge_index.shape[1]})"
            
            if draw_pyg_graph(graph_data, output_path, title):
                print(f"  Created: {output_path}")
        
        elif hasattr(item, 'x') and hasattr(item, 'edge_index'):
            output_path = os.path.join(output_dir, f"sample_{i:02d}.png")
            title = f"Sample {i} (nodes={item.x.shape[0]}, edges={item.edge_index.shape[1]})"
            if draw_pyg_graph(item, output_path, title):
                print(f"  Created: {output_path}")
        
        else:
            print(f"  Skipping sample {i}: unknown format {type(item)}")
    
    print(f"\nDone! Images saved to {output_dir}/")


def inspect_pt(filepath, key=None, create_images=False):
    print(f"=== {filepath} ===")
    print("Loading file (this may take a moment for large files)...")
    data = None
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"torch.load failed: {e}")
        print("Trying pickle fallback...")
        try:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e2:
            print(f"pickle.load also failed: {e2}")
            return
    
    if data is None:
        print("Failed to load data")
        return
    
    print(f"Type: {type(data).__name__}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        if key is not None:
            if key in data:
                print(f"\n['{key}']:")
                print(data[key])
            else:
                print(f"Key '{key}' not found. Available: {list(data.keys())}")
        else:
            for k, v in data.items():
                print(f"  '{k}': {type(v).__name__}", end="")
                if hasattr(v, 'shape'):
                    print(f" shape={v.shape}", end="")
                elif hasattr(v, '__len__'):
                    print(f" len={len(v)}", end="")
                print()
    
    elif isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        if key is not None:
            idx = int(key)
            if 0 <= idx < len(data):
                print(f"\n[{idx}]:")
                item = data[idx]
                _print_item_detail(item)
            else:
                print(f"Index {idx} out of range [0, {len(data)-1}]")
        else:
            num_to_show = min(5, len(data))
            for i in range(num_to_show):
                item = data[i]
                print(f"\n--- [{i}] {type(item).__name__} ---")
                _print_item_detail(item)
            if len(data) > num_to_show:
                print(f"\n... and {len(data)-num_to_show} more items")
        
        if create_images:
            create_dataset_images(data)
    
    elif isinstance(data, torch.Tensor):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Range: [{data.min():.4f}, {data.max():.4f}]")
        if key is not None:
            idx = int(key)
            print(f"\n[{idx}]:")
            print(data[idx])
        else:
            print(f"Data:\n{data}")
    
    else:
        print(f"Value: {data}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pt.py <file.pt> [key|index] [--images]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    key = None
    create_images = False
    
    for arg in sys.argv[2:]:
        if arg == '--images':
            create_images = True
        else:
            key = arg
    
    inspect_pt(filepath, key, create_images)
