import os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from models.pointnet import PointNetEncoder
from losses.triplet_semihard import SemiHardTripletLoss
import trimesh
import torch.nn.functional as F

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



# --------------------------
# t-SNE + Save Function
# --------------------------
def plot_and_save_tsne(embeddings, labels, class_names, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[DEBUG] t-SNE called → embeddings={embeddings.shape}")

    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    row_norms = np.linalg.norm(embeddings, axis=1)
    valid_idx = row_norms > 1e-12

    embeddings = embeddings[valid_idx]
    labels = labels[valid_idx]

    print(f"[DEBUG] Valid embeddings after cleaning = {embeddings.shape}")

    if embeddings.shape[0] == 0:
        print(f"[WARN] Epoch {epoch}: No valid embeddings for TSNE. Skipping.")
        return


    # Stable TSNE config
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, embeddings.shape[0] // 3),
        max_iter=1000,
        init="random",        # 🚀 avoids divide-by-zero PCA init
        learning_rate="auto"
    )

    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for cls in np.unique(labels):
        mask = labels == cls
        plt.scatter(reduced[mask, 0], reduced[mask, 1], s=5, label=class_names[cls])
    plt.title(f"t-SNE at epoch {epoch}")
    plt.legend(markerscale=3)
    plt.savefig(f"{out_dir}/tsne_epoch_{epoch:03d}.png")
    plt.close()



# --------------------------
# Extract Embeddings Function
# --------------------------
def extract_embeddings(model, dataloader, device):
    model.eval()
    all_emb, all_labels = [], []
    with torch.no_grad():
        for pts, lbl in dataloader:
            pts = pts.to(device)
            emb = model(pts).cpu().numpy()
            all_emb.append(emb)
            all_labels.extend(lbl)
    return np.concatenate(all_emb, axis=0), np.array(all_labels)

# --------------------------
# Training Loop
# --------------------------
def train_pointnet(data_dir: str, output_dir: str, epochs: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_ds = PointCloudDataset(data_dir, output_dir=output_dir, split="train")
    test_ds = PointCloudDataset(data_dir, output_dir=output_dir, split="test")

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    model = PointNetEncoder(feature_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = SemiHardTripletLoss()

    tsne_output_dir = os.path.join(output_dir, "tsne_plots")

    os.makedirs(tsne_output_dir, exist_ok=True)

    for epoch in range(1, epochs): 
        model.train()
        for pts, lbl in train_dl:
            pts, lbl = pts.to(device), lbl.to(device)
            out = model(pts)
            out = F.normalize(out, p=2, dim=1)   # normalize AFTER model output
            loss = criterion(out, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

        # Extract embeddings on test set
        emb, lbls = extract_embeddings(model, test_dl, device)
        print("embedding mean:", emb.mean().item(), "std:", emb.std().item())

        # Save + plot t-SNE
        plot_and_save_tsne(emb, lbls, train_ds.classes, epoch, tsne_output_dir)

    torch.save(model.state_dict(), "pointnet_triplet.pth")
    print("Training complete — model saved.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PointNet with Triplet Loss")
    parser.add_argument("--data_dir", required=True, help="Path to root directory with train/ and test/ folders.")
    parser.add_argument("--output_dir", required=True, help="Save directory.")
    parser.add_argument("--epoch", required=True, help="Number of training epochs.", type=int, default=50)

    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    train_pointnet(args.data_dir, output_dir=args.output_dir, epochs=args.epoch)