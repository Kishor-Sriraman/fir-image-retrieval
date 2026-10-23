import json
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


def load_json_image_list(json_path, image_dir):
    """Loads images listed in a JSON file from disk."""
    json_path, image_dir = Path(json_path), Path(image_dir)
    
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract image filenames (handles both dict and list structures)
    filenames = data.get("im_paths") or data.get("images") if isinstance(data, dict) else data

    images = []
    for fn in filenames:
        # Load directly from image directory using the base filename
        img_path = image_dir / Path(fn).name
        img = cv2.imread(str(img_path))
        
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            print(f"[WARNING] Could not load image: {img_path}")

    if not images:
        raise ValueError(f"[ERROR] Failed to load any valid images from {json_path}")

    print(f"[DATASET] Loaded {len(images)} images from {json_path.name}")
    return images


def load_ground_truth_h5(filepath, key="fov"):
    """Loads ground truth matrix from an HDF5 file."""
    with h5py.File(filepath, "r") as f:
        return np.array(f[key], dtype=np.float32)


class SpatialWordDataset(Dataset):
    """PyTorch Dataset wrapper for spatial grids."""
    def __init__(self, grids):
        self.grids = grids

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return torch.from_numpy(self.grids[idx]).float(), idx


class TripletDataset(Dataset):
    """Constructs (Anchor, Positive, Negative) triplets based on FOV overlap."""
    def __init__(self, q_grids, db_grids, fov_gt_matrix, pos_thresh=0.20, neg_thresh=0.05, max_triplets_per_query=10):
        self.q_grids = q_grids
        self.db_grids = db_grids
        self.triplets = []
        np.random.seed(42)
        is_self_match = (q_grids.shape[0] == db_grids.shape[0])

        # For each query index, see which database images are positive or negative
        for q_idx, fov_row in enumerate(fov_gt_matrix):
            pos_indices = np.where(fov_row >= pos_thresh)[0]
            neg_indices = np.where(fov_row <= neg_thresh)[0]

            if is_self_match:
                pos_indices = pos_indices[pos_indices != q_idx]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            sampled_positives = np.random.choice(
                pos_indices, 
                size=min(len(pos_indices), max_triplets_per_query), 
                replace=False
            )

            for p_idx in sampled_positives:
                n_idx = np.random.choice(neg_indices)
                self.triplets.append((q_idx, p_idx, n_idx))

        print(f"[TripletDataset] Generated {len(self.triplets)} triplets.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        q_idx, p_idx, n_idx = self.triplets[idx]
        return (
            torch.from_numpy(self.q_grids[q_idx]).float(),
            torch.from_numpy(self.db_grids[p_idx]).float(),
            torch.from_numpy(self.db_grids[n_idx]).float(),
        )