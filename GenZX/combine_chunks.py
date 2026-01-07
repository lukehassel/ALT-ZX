import os
import glob
import torch


def main():
    chunk_dir = "GenZX"
    pattern = os.path.join(chunk_dir, "genzx_dataset_chunk_*.pt")
    chunk_files = sorted(glob.glob(pattern))
    
    if not chunk_files:
        print(f"No chunk files found matching {pattern}")
        return
    
    print(f"Found {len(chunk_files)} chunk files")
    
    all_data = []
    for chunk_path in chunk_files:
        print(f"Loading {chunk_path}...")
        data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        all_data.extend(data)
        print(f"  Added {len(data)} samples (total: {len(all_data)})")
    
    output_path = os.path.join(chunk_dir, "genzx_dataset.pt")
    print(f"\nSaving combined dataset to {output_path}...")
    torch.save(all_data, output_path)
    print(f"Done! Total samples: {len(all_data)}")


if __name__ == "__main__":
    main()
