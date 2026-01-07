import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

DEFAULT_DATASET_PATH = "GflowEncoder/dataset.pt"


def integration_load_dataset(filepath):
    return torch.load(filepath, weights_only=False)


class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]


def triplet_collate_fn(batch):
    # Batch (anchor, positive, negative) triplets into PyG Batch objects
    anchors = [item[0] for item in batch]
    positives = [item[1] for item in batch]
    negatives = [item[2] for item in batch]
    
    return (
        Batch.from_data_list(anchors),
        Batch.from_data_list(positives),
        Batch.from_data_list(negatives)
    )


def create_data_loader(dataset_path=DEFAULT_DATASET_PATH, batch_size=64, shuffle=True, num_workers=0):
    print(f"Loading dataset from {dataset_path}...")
    triplets = integration_load_dataset(dataset_path)
    print(f"Loaded {len(triplets)} triplets")
    
    dataset = TripletDataset(triplets)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=triplet_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader


if __name__ == "__main__":
    loader = create_data_loader(batch_size=32)
    print(f"Created loader with {len(loader)} batches")
    
    for anchor, pos, neg in loader:
        print(f"Anchor: {anchor.x.shape}, Pos: {pos.x.shape}, Neg: {neg.x.shape}")
        break
