import os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from models.pointnet import PointNetEncoder
from losses.triplet_semihard import SemiHardTripletLoss
import trimesh
import torch.nn.functional as F
from utils.plot_tsne import plot_and_save_tsne

import os, re
import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, Optional


def normalize_points_to_unit_sphere(pts: np.ndarray) -> np.ndarray:
    # center and scale to unit sphere
    pts = pts - pts.mean(axis=0, keepdims=True)
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm < 1e-12:
        return pts.astype(np.float32)
    return (pts / max_norm).astype(np.float32)

def sanitize_filename(name: str) -> str:
    """Remove/replace invalid filesystem chars."""
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def mesh_to_pointnet_input(
    obj_path: str,
    output_dir: str = "pointnet_data",
    num_points: int = 2048,
    seed: Optional[int] = None,
    force_regenerate: bool = False
) -> Tuple[Optional[str], Optional[np.ndarray]]:

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(obj_path):
        print(f"[mesh_to_pointnet_input] OBJ missing: {obj_path}")
        return None, None

    stem = sanitize_filename(Path(obj_path).stem)
    output_path = os.path.join(output_dir, f"{stem}_pc.npy")

    # ───────────────────────────
    # Try loading existing .npy
    # ───────────────────────────
    if os.path.exists(output_path) and not force_regenerate:
        try:
            pts = np.load(output_path)
            if pts.shape == (num_points, 3) and np.isfinite(pts).all():
                return output_path, pts
        except Exception:
            pass  # corrupted file → regenerate

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def sanitize(points: np.ndarray, n: int) -> np.ndarray:
        """Remove NaN / Inf, enforce exact shape, guarantee non-empty output."""
        points = np.asarray(points, dtype=np.float32)

        # Remove invalid rows
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]

        # If no valid rows, generate fallback
        if points.size == 0:
            print("    [fallback] generating synthetic sphere point cloud")
            phi = np.random.rand(n) * np.pi * 2
            costheta = np.random.rand(n) * 2 - 1
            theta = np.arccos(costheta)
            r = np.ones(n)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            return np.vstack([x, y, z]).T.astype(np.float32)

        # Normal resampling / padding
        if points.shape[0] >= n:
            idx = np.random.choice(points.shape[0], n, replace=False)
            points = points[idx]
        else:
            pad = np.zeros((n, 3), dtype=np.float32)
            pad[: points.shape[0]] = points
            points = pad

        return points
    def sample_from_mesh(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, n)
        except Exception:
            pts = np.asarray(mesh.vertices)

        return sanitize(pts, n)

    # ───────────────────────────
    # Load mesh
    # ───────────────────────────
    try:
        mesh = trimesh.load(obj_path, process=False, force="mesh")
    except Exception as e:
        print(f"[mesh_to_pointnet_input] trimesh.load failed for {obj_path}: {e}")
        return None, None

    # If Scene, merge meshes
    if isinstance(mesh, trimesh.Scene):
        try:
            geoms = list(mesh.geometry.values())
            if len(geoms) == 0:
                print(f"[mesh_to_pointnet_input] Scene contains no geometry: {obj_path}")
                return None, None
            mesh = trimesh.util.concatenate(geoms)
        except Exception as e:
            print(f"[mesh_to_pointnet_input] Failed to concatenate scene: {e}")
            return None, None

    # ───────────────────────────
    # Point cloud extraction
    # ───────────────────────────
    try:
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            print(f"[mesh_to_pointnet_input] No faces: sampling vertices {obj_path}")
            if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
                print(f"[mesh_to_pointnet_input] No faces: sampling vertices {obj_path}")
                points = sanitize(mesh.vertices, num_points)
            else:
                try:
                    pts, _ = trimesh.sample.sample_surface(mesh, num_points)
                    points = sanitize(pts, num_points)
                except Exception:
                    print(f"[mesh_to_pointnet_input] sample_surface failed, using vertices: {obj_path}")
                    points = sanitize(mesh.vertices, num_points)

        else:
            points = sample_from_mesh(mesh, num_points)
    except Exception as e:
        print(f"[mesh_to_pointnet_input] Sampling failed for {obj_path}: {e}")
        return None, None

    # ───────────────────────────
    # Normalize to unit sphere
    # ───────────────────────────
    try:
        points = normalize_points_to_unit_sphere(points)
        if not np.isfinite(points).all():
            raise ValueError("Normalization produced NaN/Inf")
    except Exception as e:
        print(f"[mesh_to_pointnet_input] Normalization failed: {e}")
        return None, None

    # ───────────────────────────
    # Save atomically
    # ───────────────────────────
    tmp = output_path + ".tmp"
    try:
        np.save(output_path, points)
#        os.replace(tmp, output_path)
    except Exception as e:
        print(f"[mesh_to_pointnet_input] Failed saving {output_path}: {e}")
        return None, None

    print(f"[mesh_to_pointnet_input] saved: {output_path}")
    return output_path, points

