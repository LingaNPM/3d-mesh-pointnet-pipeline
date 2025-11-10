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

## Loss: Semi-Hard Triplet Loss

Same as FaceNet:

minimize:  max( d(a,p) − d(a,n) + margin , 0 )


where

a = anchor

p = positive (same class)

n = negative (different class)

Semi-hard negative definition:

d(a,p) < d(a,n) < margin + d(a,p)


## Dataset Format

Dataset structure:

data/
   ├── hinge/
   │    ├── train/
   │    │     ├── hinge_0001.obj
   │    │     ├── hinge_0002.obj
   │    └── test/
   │          ├── hinge_0099.obj
   ├── motor_mount/
   │    ├── train/
   └── ...


The script automatically:

Loads each .obj

Samples a 2048-point cloud

Normalizes to a unit sphere

Saves as <name>_pc.npy under pointnet_data/


## Training

python train_pointnet.py


Outputs include:

pointnet_data/*.npy → cached point clouds

checkpoints/pointnet_epoch_XX.pt

tsne_plots/epoch_XX.png → used in publication figures


## t-SNE Embedding Visualization

After every epoch, embeddings are visualized via PCA → t-SNE:

<p align="center"> <img src="tsne_plots/epoch_032.png" width="320"> &nbsp;&nbsp;&nbsp; <img src="tsne_plots/epoch_048.png" width="320"> </p>

Left = poor separation (early), Right = clearer clustering.


## Evaluation / Descriptor Export

Use the trained checkpoint to export descriptors:

python export_descriptors.py


Outputs:

descriptors/
   ├── hinge_001.npy
   ├── hinge_002.npy
   └── ...


Each file contains a 256-D L2-normalized vector.

These can be fed into your Metal / CUDA search engine.