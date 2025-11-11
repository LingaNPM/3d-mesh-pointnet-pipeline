# PointNet Pipeline — Triplet / Semi-Hard Contrastive Embedding Learning  
Real-time 3D Part Similarity Descriptor Training (Point Cloud)

<p align="center">
  <img src="https://img.shields.io/badge/3D%20Deep%20Learning-PointNet%20Pipeline-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Triplet%20Loss-Semi--Hard%20Mining-green?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Embeddings-256D-orange?style=for-the-badge" />
</p>

<h1 align="center">🔷 PointNet-Pipeline</h1>
<p align="center">Point cloud based shape embedding + triplet loss training for mechanical parts</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/LingaNPM/3d-mesh-pointnet-pipeline?style=flat-square" />
  <img src="https://img.shields.io/github/forks/LingaNPM/3d-mesh-pointnet-pipeline?style=flat-square" />
  <img src="https://img.shields.io/github/license/LingaNPM/3d-mesh-pointnet-pipeline?style=flat-square" />
</p>

<p align="center">
  <b>Train PointNet + Triplet loss to generate 256-D shape descriptors</b><br/>
  Includes automatic mesh → point cloud conversion, t-SNE visualization, and embedding extraction
</p>



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
> Triaining was done in NVIDIA GPU using CUDA 13.0 version driver.



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

Origin of the dataset: [A Large-Scale Annotated Mechanical Components Benchmark](https://engineering.purdue.edu/cdesign/wp/a-large-scale-annotated-mechanical-components-benchmark-for-classification-and-retrieval-tasks-with-deep-neural-networks/)

We aim to use real-world mechanical part models from the referenced dataset for training and evaluation. For benchmarking, we selected 20 distinct classes from this dataset to represent a diverse set of mechanical parts.

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

Our setup used NVIDIA GPU using CUDA 13.0 version driver.

For 50 epoches, the training took around 7 mins.

## Results — Shape Embedding & Clustering (t-SNE Visualization)

After every epoch, embeddings are visualized via PCA → t-SNE:

<p align="center"> <img src="https://github.com/LingaNPM/pointnet-pipeline/blob/main/tsne_plots/tsne_epoch_001.png" width="320"> &nbsp;&nbsp;&nbsp; <img src="https://github.com/LingaNPM/pointnet-pipeline/blob/main/tsne_plots/tsne_epoch_024.png" width="320"> &nbsp;&nbsp;&nbsp; <img src="https://github.com/LingaNPM/pointnet-pipeline/blob/main/tsne_plots/tsne_epoch_048.png" width="320"> </p>

Epoches 1, 24, 48 are shown here.


After training PointNet on 20 mechanical component categories (MCB dataset — bearings, bushings, clamps, gears, hinges, etc.), we visualized the learned embeddings using t-SNE.

<p align="center"> <img src="https://github.com/LingaNPM/pointnet-pipeline/blob/main/tsne_plots/tsne_epoch_048.png" width="600"> </p>

Figure: t-SNE projection of 256-D PointNet embeddings (epoch 48).


# What the visualization shows

Samples belonging to the same mechanical part class form visible clusters.

PointNet successfully learns a global shape signature from point clouds.

Training with Triplet Loss (semi-hard negative mining) encourages separation between classes.

# Why clusters form

PointNet works in two stages:

| Stage              | What happens                                          |
| ------------------ | ----------------------------------------------------- |
| Point-wise MLP     | Learns per-point geometric features (`xyz → feature`) |
| Global Max Pooling | Aggregates them into a single descriptor vector       |

Because of max pooling, all point features are reduced into a single global feature, representing the overall silhouette of the object.

This is why parts that look globally different cluster well.

# Limitation observed

PointNet does not encode mesh topology or edge relationships.

| Issue observed                                                  | Why it happens                             |
| --------------------------------------------------------------- | ------------------------------------------ |
| Classes with similar silhouettes overlap in the embedding space | Max pooling erases local geometric details |
| Small structural differences are ignored                        | No neighborhood / connectivity awareness   |

Example cases where PointNet tends to confuse:

hinge ↔ fork-joint

washer ↔ disc

stud ↔ pin

Even though these parts differ in mechanical purpose, their global geometry (cylindrical or planar distribution of points) may appear similar.

PointNet = excellent global similarity
→ but weak for fine-grained mechanical part discrimination
→ because it treats points independently and lacks edge or topological reasoning.

# Key Takeaways

PointNet embeddings successfully cluster different mechanical part types.

Clusters are meaningful, but not tightly separated for look-alike categories.

Motivates using mesh-based or graph-based models where geometry relationships (edges, curvature) matter.

# Implementation Details

| Component      | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| Model          | PointNet Encoder (global feature extractor)                    |
| Embedding size | `256-dim`                                                      |
| Loss           | `Triplet Loss (semi-hard negative mining)`                     |
| Dataset        | Mechanical part dataset (20 classes)                           |
| Input          | 2048 sampled points from each mesh (normalized to unit sphere) |

# Commands to Reproduce

# create point cloud npy files automatically
python train_pointnet.py \
    --data_dir data/MCB_dataset_root \
    --output_dir output_pointnet \
    --epoch 50

# During training, the repository saves:
output_pointnet/
 ├── model_checkpoint.pth
 ├── tsne_plots/
 │    ├── tsne_epoch_01.png
 │    ├── tsne_epoch_10.png
 │    ├── ...
 │    └── tsne_epoch_48.png  ← shown above

# Result Summary

PointNet learns useful global representations of 3D mechanical parts.
But it lacks structural discrimination because it does not model edges, adjacency, or topology.

This establishes a baseline and motivates exploring mesh-aware models for fine-grained similarity search.

# Training enforced grouping

We use Semi-Hard Triplet Loss, which pulls embeddings of same-class samples together and pushes different classes apart only when they are too close.
This encourages clustering behavior, but if two objects are structurally similar:

The network learns "these are close, but different",
yet does not learn clear boundary separation.

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