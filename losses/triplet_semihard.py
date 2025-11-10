# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiHardTripletLoss(nn.Module):
    """
    Implements Semi-Hard Triplet Loss from FaceNet:
    - Anchor–Positive distance is small
    - Negative is *further than positive* but still within margin (semi-hard)

    triplet: (a, p, n) where:
        d(a,p) < d(a,n) < d(a,p) + margin
    """

    def __init__(self, margin=0.2, distance='cosine'):
        super().__init__()
        self.margin = margin
        assert distance in ["cosine", "euclidean"]
        self.distance = distance

    def pairwise_distance(self, x):
        """
        Computes pairwise distance matrix (batch_size x batch_size)
        """
        if self.distance == "euclidean":
            # squared Euclidean
            x2 = torch.sum(x * x, dim=1, keepdim=True)
            dist = x2 + x2.t() - 2 * (x @ x.t())
            return torch.clamp(dist, min=0.0)

        elif self.distance == "cosine":
            # cosine distance (1 - cosine similarity)
            x_norm = F.normalize(x, p=2, dim=1)
            sim = torch.matmul(x_norm, x_norm.t())  # [-1..1]
            return 1.0 - sim

    def forward(self, embeddings, labels):
        """
        embeddings: (B, D)
        labels: (B,)
        """
        batch_size = embeddings.shape[0]
        dist_matrix = self.pairwise_distance(embeddings)

        triplet_losses = []

        for i in range(batch_size):
            anchor_label = labels[i]

            # Positive indices (same class, exclude anchor itself)
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=labels.device) != i)
            neg_mask = labels != anchor_label

            pos_dists = dist_matrix[i][pos_mask]
            neg_dists = dist_matrix[i][neg_mask]

            if len(pos_dists) == 0 or len(neg_dists) == 0:
                # ✅ cannot form a meaningful triplet for this anchor
                continue

            # Semi-hard negative: d(an) > d(ap) but closest possible
            d_ap = pos_dists.max()  # hardest positive
            semi_neg = neg_dists[neg_dists > d_ap]

            if len(semi_neg) > 0:
                d_an = semi_neg.min()
            else:
                # ✅ fallback: pick hardest negative, but ONLY if exists
                d_an = neg_dists.min()

            # triplet loss
            loss = torch.relu(d_ap - d_an + self.margin)
            triplet_losses.append(loss)

        if len(triplet_losses) == 0:
            # ✅ nothing to compute this batch
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(triplet_losses).mean()
