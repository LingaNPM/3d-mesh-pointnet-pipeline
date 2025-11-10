# PointNet Pipeline — Triplet / Semi-Hard Contrastive Embedding Learning  
Real-time 3D Part Similarity Descriptor Training (Point Cloud)

---

## Overview

This repository contains the **point-cloud based training pipeline** used to compare against our
MeshCNN / Mesh-GAT based mesh-descriptor model.

Instead of edges and mesh topology (MeshCNN),
this pipeline converts each `.obj` mesh into a **normalized point cloud** (PointNet-style input),
and trains a **global embedding network** using:

- Minimal PointNet Encoder (Conv1d + Global Max Pool)
- Semi-Hard Triplet Loss (FaceNet-style)
- Fixed descriptor dimensionality (default: **256-D**)
- t-SNE visualization per epoch (saved for paper figures)

This model produces **shape descriptors** that can be compared using cosine similarity.

> This repo only handles **training + embedding generation** for point clouds.  
> Our real-time similarity search engine (Metal GPU) lives in another repo.

---

## Why this repository exists

We use this to benchmark our **MeshCNN descriptor model** against a **PointNet-based baseline**,
on the same dataset of **20 mechanical part classes** (custom industrial dataset: MCB_ASSETS).

| Model | Input | Descriptor | Preserves structure? | Best at |
|--------|-------|-------------|----------------------|---------|
| MeshCNN (ours) | Edges + topology | Patch-aware | Handles small local structures | complex mechanical parts |
| PointNet (baseline) | Point cloud (xyz only) | Global (256-D) | loses fine structure | large smooth shapes |

This comparison appears in our paper submission.

---

## Repository Structure

pointnet-pipeline/
│
├── train_pointnet.py # Main training script
├── losses/
│ └── triplet_semihard.py # Semi-hard triplet loss
├── utils/
│ ├── mesh_to_pointnet.py # OBJ → point cloud conversion + sanitizing
│ └── tsne_plot.py # PCA + TSNE per epoch
│
├── pointnet_data/ # Generated pointcloud npy files (ignored via .gitignore)
├── tsne_plots/ # Embedding visualizations saved per epoch
│
└── README.md # (this file)