import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh, random
import math
import sys
from pathlib import Path
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import trimesh

import os
import json
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm

# dataset_precomputed.py

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json


class PrecomputedMeshDataset(Dataset):
    def __init__(self, precomputed_dir, index_path):
        self.precomputed_dir = Path(precomputed_dir)
        
        with open(index_path, "r") as f:
            self.entries = json.load(f)

        # Create class-to-index mapping
        class_names = sorted(list({entry["label"] for entry in self.entries}))
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        # Populate labels and filenames
        self.labels = [self.class_to_idx[entry["label"]] for entry in self.entries]
        self.filenames = [f"{entry['id']}.off" for entry in self.entries]  # or keep just ID if you prefer


    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        mesh_id = entry["id"]
        label = self.class_to_idx[entry["label"]]

        # Load precomputed files
        feats = np.load(self.precomputed_dir / f"{mesh_id}_feats.npy")
        nbrs = np.load(self.precomputed_dir / f"{mesh_id}_nbrs.npy")

        # Convert to torch tensors
        feats = torch.tensor(feats, dtype=torch.float32)
        return feats, nbrs, mesh_id, label


class BalancedTripletMeshDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split  # points to train/ or test/
        self.transform = transform

        self.label_to_paths = {}
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                paths = list(class_dir.glob("*.off"))
                if len(paths) >= 1:
                    self.label_to_paths[class_dir.name] = paths

        self.samples = [(p, label) for label, paths in self.label_to_paths.items() for p in paths]

        if len(self.label_to_paths) < 2:
            raise ValueError("Triplet sampling requires at least 2 classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, anchor_label = self.samples[idx]

        # Positive sample (same class, different path)
        pos_candidates = [p for p in self.label_to_paths[anchor_label] if p != anchor_path]
        if not pos_candidates:
            raise ValueError(f"No positive match for label '{anchor_label}'")
        positive_path = random.choice(pos_candidates)

        # Negative sample (different class)
        neg_labels = [label for label in self.label_to_paths if label != anchor_label]
        if not neg_labels:
            raise ValueError(f"No negative labels available for anchor label '{anchor_label}'")
        neg_label = random.choice(neg_labels)
        negative_path = random.choice(self.label_to_paths[neg_label])

        # Load meshes
        anchor_data = self.load_mesh(anchor_path)
        positive_data = self.load_mesh(positive_path)
        negative_data = self.load_mesh(negative_path)

        return anchor_data, positive_data, negative_data

    def load_mesh(self, path):
        # Replace this with your actual mesh loader (e.g., trimesh, MeshCNN)
        return str(path)  # Placeholder: just returns file path for now


class MeshTripletDataset(Dataset):
    def __init__(self, file_paths, labels, load_mesh_fn=None):
        """
        file_paths: List of mesh file paths
        labels:     List of class labels (same length as file_paths)
        load_mesh_fn: Optional custom mesh loader (returns (edge_feats, edge_neighbors))
        """
        self.samples = list(zip(file_paths, labels))
        self.label_to_paths = defaultdict(list)
        for path, label in self.samples:
            self.label_to_paths[label].append(path)

        self.load_mesh = load_mesh_fn if load_mesh_fn else self.default_load_mesh

    def __len__(self):
        return len(self.samples)

    def __getitem_cpy__(self, idx):
        anchor_path, anchor_label = self.samples[idx]

        # Sample a positive example (same class, different file)
        pos_candidates = [p for p in self.label_to_paths[anchor_label] if p != anchor_path]
        if not pos_candidates:
            raise ValueError(f"No positive match found for label '{anchor_label}'")
        positive_path = random.choice(pos_candidates)

        # Sample a negative example (different class)
        neg_labels = list(self.label_to_paths.keys())
        neg_labels.remove(anchor_label)
        neg_label = random.choice(neg_labels)
        negative_path = random.choice(self.label_to_paths[neg_label])

        # Load mesh data
        anchor_data = self.load_mesh(anchor_path)
        positive_data = self.load_mesh(positive_path)
        negative_data = self.load_mesh(negative_path)

        return anchor_data, positive_data, negative_data
    
    def __getitem__(self, idx):
        for _ in range(5):  # retry a few times
            anchor_path, anchor_label = self.samples[idx]

            # Positive sample
            pos_candidates = [p for p in self.label_to_paths[anchor_label] if p != anchor_path]
            if not pos_candidates:
                return None
            positive_path = random.choice(pos_candidates)

            # Negative sample
            neg_labels = [l for l in self.label_to_paths if l != anchor_label]
            if not neg_labels:
                return None
            neg_label = random.choice(neg_labels)
            negative_path = random.choice(self.label_to_paths[neg_label])

            # Load mesh
            anchor_data = self.load_mesh(anchor_path)
            positive_data = self.load_mesh(positive_path)
            negative_data = self.load_mesh(negative_path)

            return (
                (anchor_data[0], anchor_data[1], anchor_path, anchor_label),
                (positive_data[0], positive_data[1], positive_path, anchor_label),
                (negative_data[0], negative_data[1], negative_path, neg_label)
            )

    def default_load_mesh(self, mesh_path):
        name = Path(mesh_path).stem
        feats = np.load(f"precomputed/{name}_feats.npy")
        nbrs = np.load(f"precomputed/{name}_nbrs.npy", allow_pickle=True)
        return torch.tensor(feats, dtype=torch.float32), nbrs.tolist()



class ModelNet40MeshDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.data = []
        self.class_to_idx = {}
        self.mesh_paths = []

        all_classes = sorted(os.listdir(root_dir))
        for class_idx, class_name in enumerate(all_classes):

            self.class_to_idx[class_name] = class_idx
            split_dir = os.path.join(root_dir, class_name, split)
            files = glob.glob(os.path.join(split_dir, '*.off'))
            for f in files:
                self.mesh_paths.append((f, class_idx))

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        path, label = self.mesh_paths[idx]
        mesh = trimesh.load(path, process=False)

        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        edge_feats, edge_neighbors = extract_edge_features(vertices, faces)

        edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        return edge_feats, edge_neighbors, label


from torch.utils.data import Dataset

class MeshDataset(Dataset):
    """
    PyTorch Dataset for loading mesh files and returning MeshCNN-style edge features.
    """
    def __init__(self, file_paths, labels=None, transform=None):
        """
        Args:
            file_paths (List[str]): List of paths to mesh files (.off or .obj).
            labels (List[Any], optional): Optional list of labels for each mesh (for similarity learning).
            transform (callable, optional): Optional transform to apply to edge_feats.
        """
        self.file_paths = file_paths
        self.labels = labels if labels is not None else [None] * len(file_paths)
        if labels is not None and len(labels) != len(file_paths):
            raise ValueError("Length of labels must match length of file_paths")
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the mesh and compute edge features
        mesh_path = self.file_paths[idx]
        edge_feats, edge_neighbors = load_mesh(mesh_path)
        # Apply transform to features if provided (e.g., normalization or augmentation)
        if self.transform:
            edge_feats = self.transform(edge_feats)
        if self.labels[idx] is not None:
            return edge_feats, edge_neighbors, self.labels[idx]
        else:
            return edge_feats, edge_neighbors


import torch
from torch.utils.data import Dataset
import random

class MeshPairDataset(Dataset):
    def __init__(self, file_paths, labels):
        """
        Args:
            file_paths (List[str]): List of mesh file paths (.off or .obj)
            labels (List[str|int]): Class label or ID for each mesh
        """
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_paths = {}
        for path, label in zip(file_paths, labels):
            self.label_to_paths.setdefault(label, []).append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        anchor_path = self.file_paths[idx]
        anchor_label = self.labels[idx]

        # Decide whether to create a positive or negative pair
        if random.random() > 0.5:
            # Positive pair: same label
            candidates = [p for p in self.label_to_paths[anchor_label] if p != anchor_path]
            if not candidates:
                # fallback if only one sample in class
                candidates = [anchor_path]
            other_path = random.choice(candidates)
            label = 1
        else:
            # Negative pair: different label
            other_label = random.choice([l for l in self.label_to_paths if l != anchor_label])
            other_path = random.choice(self.label_to_paths[other_label])
            label = 0

        # Load both meshes
        anchor_feats, anchor_neighbors = load_mesh(anchor_path)
        other_feats, other_neighbors = load_mesh(other_path)

        return (anchor_feats, anchor_neighbors), (other_feats, other_neighbors), label
