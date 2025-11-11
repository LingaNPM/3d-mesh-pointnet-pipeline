# triplet_loader.py

import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from torch.utils.data import Sampler
import numpy as np
import os
from utils.mesh_to_pointnet import mesh_to_pointnet_input

# --------------------------
# Dataset Loader
# --------------------------
class PointCloudDataset(Dataset):
    def __init__(self, root, output_dir:str, split="train"):
        self.paths, self.labels = [], []
        self.classes = sorted(os.listdir(root))

        for label_idx, cls in enumerate(self.classes):
            split_dir = os.path.join(root, cls, split)
            if not os.path.isdir(split_dir):
                continue

            for f in os.listdir(split_dir):
                obj_path = os.path.join(split_dir, f)

                if 'partial' in f.lower() or not f.lower().endswith('.obj'):
                    print(f"[INFO] Skipping non-OBJ or partial file: {obj_path}")
                    continue  # skip partial or non-OBJ files
                pc_path, _ = mesh_to_pointnet_input(obj_path, output_dir=output_dir)  # ALWAYS tuple

                if pc_path is None:   # skip invalid files
                    continue

                self.paths.append(pc_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pts = np.load(self.paths[idx]).astype(np.float32)

        if not np.isfinite(pts).all():
            print(f"[ERROR] Invalid points in {self.paths[idx]}, regenerating...")
            pc_path = mesh_to_pointnet_input(self.original_obj_paths[idx])
            pts = np.load(pc_path).astype(np.float32)

        return torch.tensor(pts, dtype=torch.float32), self.labels[idx]

