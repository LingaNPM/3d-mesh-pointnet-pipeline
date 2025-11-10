# triplet_loader.py

import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from torch.utils.data import Sampler
import re
import random

class MeshTripletDataset(Dataset):
    def __init__(self, mesh_entries):
        """
        mesh_entries: List of (feat_tensor, nbrs_array, path_or_id, class_label)
        """
        self.meshes = mesh_entries

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        return self.meshes[idx]

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples, num_batches):
        self.labels = labels
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.labels_set = list(set(labels))
        self.batch_size = self.n_classes * self.n_samples
        self.num_batches = num_batches

        if len(self.labels_set) < self.n_classes:
            raise ValueError(
                f"Cannot sample {self.n_classes} classes per batch — only {len(self.labels_set)} unique labels in dataset."
            )

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_labels = random.sample(self.labels_set, self.n_classes)
            batch = []
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.n_samples:
                    batch.extend(random.sample(indices, k=self.n_samples))
                else:
                    batch.extend(random.choices(indices, k=self.n_samples))  # allow repeats if not enough
            yield batch

    def __len__(self):
        return self.num_batches

class VariantAwareBalancedSampler(Sampler):
    def __init__(self, labels, filenames, n_classes, n_samples, num_batches):
        """
        labels     : list of class labels (same length as dataset)
        filenames  : list of mesh file paths or names (same length as dataset)
        """
        self.labels = labels
        self.filenames = filenames
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.num_batches = num_batches

        # Group: {class → {base_id → [indices]}}
        self.class_to_baseid_indices = defaultdict(lambda: defaultdict(list))
        for idx, (label, fname) in enumerate(zip(labels, filenames)):
            base_id = self.extract_base_id(fname)
            self.class_to_baseid_indices[label][base_id].append(idx)

        self.labels_set = list(self.class_to_baseid_indices.keys())

        if len(self.labels_set) < self.n_classes:
            raise ValueError(f"Only {len(self.labels_set)} classes available.")

    def extract_base_id(self, filename):
        # E.g., "nut_0045_partial_top_half.off" → "nut_0045"
        match = re.match(r"^.*_\d+", filename)
        if match:
            return match.group(0)
        else:
            raise ValueError(f"Filename '{filename}' does not match expected pattern.")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            selected_classes = random.sample(self.labels_set, self.n_classes)
            for cls in selected_classes:
                base_to_indices = self.class_to_baseid_indices[cls]
                selected_base_ids = random.sample(
                    list(base_to_indices.keys()), k=self.n_samples
                )
                for base_id in selected_base_ids:
                    idx_list = base_to_indices[base_id]
                    # Choose one of the variants (or full)
                    idx = random.choice(idx_list)
                    batch.append(idx)
            yield batch

    def __len__(self):
        return self.num_batches



def mesh_collate_fn(batch):
    feats = [b[0] for b in batch]
    nbrs = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    labels = [b[3] for b in batch]
    return feats, nbrs, ids, labels


def get_triplets_semi_hard(embeddings, labels, margin=0.2):
    triplets = []
    batch_size = embeddings.size(0)

    for i in range(batch_size):
        anchor = embeddings[i]
        anchor_label = labels[i]

        pos_indices = (labels == anchor_label).nonzero(as_tuple=False).flatten()
        pos_indices = pos_indices[pos_indices != i]
        if len(pos_indices) == 0:
            continue
        pos_idx = random.choice(pos_indices)
        positive = embeddings[pos_idx]
        pos_dist = torch.norm(anchor - positive).item()

        neg_indices = (labels != anchor_label).nonzero(as_tuple=False).flatten()
        semi_hard_neg = None
        for j in neg_indices:
            neg = embeddings[j]
            neg_dist = torch.norm(anchor - neg).item()
            if pos_dist < neg_dist < pos_dist + margin:
                semi_hard_neg = j
                break
        if semi_hard_neg is None and len(neg_indices) > 0:
            semi_hard_neg = random.choice(neg_indices)

        if semi_hard_neg is not None:
            triplets.append((i, pos_idx.item(), semi_hard_neg.item()))

    return triplets
