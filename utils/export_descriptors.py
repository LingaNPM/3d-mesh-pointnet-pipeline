import os, sys
import numpy as np
import json
import trimesh
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mesh_loader import extract_edge_features
from mesh_loader import load_mesh 
from models.meschcnn_model import MeshCNNEncoder
import math



# def export_meshcnn_encoder():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load trained encoder
#     model = MeshCNNEncoder()
#     model.load_state_dict(torch.load("meshcnn_encoder.pth", map_location=device))
#     model.eval().to(device)

#     mesh_dir = "data/ModelNet40/ModelNet40/cone/train"
#     out_dir = "descriptors"
#     os.makedirs(out_dir, exist_ok=True)

#     index = {}

#     for fname in os.listdir(mesh_dir):
#         if not fname.endswith(".off") and not fname.endswith(".obj"):
#             continue

#         path = os.path.join(mesh_dir, fname)
#         mesh = trimesh.load(path, process=False)
#         vertices = np.array(mesh.vertices)
#         faces = np.array(mesh.faces)
#         edge_feats_np, edge_neighbors = extract_edge_features(vertices, faces)
#         edge_feats = torch.tensor(edge_feats_np).float().to(device)

#         with torch.no_grad():
#             vec = model(edge_feats, edge_neighbors)
#         desc = vec.squeeze().cpu().numpy()

#         name = os.path.splitext(fname)[0]
#         np.save(os.path.join(out_dir, f"{name}.npy"), desc)
#         index[name] = path

#     # Save index
#     with open(os.path.join(out_dir, "index.json"), "w") as f:
#         json.dump(index, f, indent=2)

#     print("Descriptors saved to:", out_dir)


TARGET_DIM = 256  # expected descriptor dimension for Metal

def pad_to_256(desc: np.ndarray) -> np.ndarray:
    if desc.size == TARGET_DIM:
        return desc
    elif desc.size < TARGET_DIM:
        return np.pad(desc, (0, TARGET_DIM - desc.size), mode='constant')
    else:
        print(f"Skipping descriptor with shape {desc.shape}: exceeds 256")
        return None

def export_descriptors_for_metal(folder: str, output_path: str):
    all_descriptors = []
    filenames = []

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.npy'):
            continue

        if any(s in fname for s in ["_cropped", "_rotated", "_noise", "_smoothed", "_partial", "_simplified"]):
            continue

        path = os.path.join(folder, fname)
        try:
            desc = np.load(path, allow_pickle=True).astype(np.float32).flatten()
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue

        if desc.ndim != 1 or desc.size < 8 or desc.size > 2048:
            print(f"Skipping {fname}: unexpected shape {desc.shape}")
            continue

        desc = pad_to_256(desc)
        if desc is None:
            continue

        all_descriptors.append(desc)
        filenames.append(fname)

    if not all_descriptors:
        raise RuntimeError("No valid descriptors found.")

    stacked = np.vstack(all_descriptors).astype(np.float32)
    stacked.tofile(output_path)

    print(f"Exported {len(all_descriptors)} descriptors to {output_path}, shape: {stacked.shape}")


    # Save metadata
    with open(output_path.replace(".bin", "_filenames.json"), "w") as f:
        json.dump(filenames, f)
    with open(output_path.replace(".bin", "_shape.json"), "w") as f:
        json.dump({
            "count": len(all_descriptors),
            "dimension": stacked.shape[1]
        }, f)

    print(f"Exported {stacked.shape[0]} descriptors with padded dim {stacked.shape[1]}")


# def export_descriptors(model_path, feats_dir, save_dir, index_out="index.json", in_feats=5, device=None):
#     """
#     Compute and export mean descriptors for all meshes in feats_dir.
    
#     Args:
#         model_path: path to trained .pth file
#         feats_dir: directory containing *_feats.npy and *_nbrs.npy
#         save_dir: output dir for descriptors and index.json
#         index_out: name of index file (default: index.json)
#         in_feats: number of input features per edge (default: 5)
#         device: torch.device (default: auto-detect)
#     """
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(save_dir, exist_ok=True)
#     index = {}

#     # Load model
#     print(f"Loading model from {model_path}")
#     model = MeshCNNEncoder(in_feats=in_feats, with_mean=False).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     # Get mesh IDs from *_feats.npy files
#     feats_files = list(Path(feats_dir).glob("*_feats.npy"))
#     print(f"Found {len(feats_files)} meshes to export.")

#     for feats_file in tqdm(feats_files, desc="Exporting descriptors"):
#         mesh_id = feats_file.stem.replace("_feats", "")
#         nbrs_file = feats_file.parent / f"{mesh_id}_nbrs.npy"

#         try:
#             edge_feats = np.load(feats_file)
#             edge_nbrs = np.load(nbrs_file)

#             edge_feats = torch.tensor(edge_feats, dtype=torch.float32).to(device)
#             edge_nbrs = torch.tensor(edge_nbrs, dtype=torch.long).to(device)

#             with torch.no_grad():
#                 emb = model(edge_feats, edge_nbrs)  # [num_edges, emb_dim]
#                 descriptor = emb.mean(dim=0)
#                 descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=0)
#                 descriptor_np = descriptor.cpu().numpy()

#             # Save descriptor
#             out_path = Path(save_dir) / f"{mesh_id}.npy"
#             np.save(out_path, descriptor_np)

#             # Update index
#             index[mesh_id] = f"{mesh_id}"

#         except Exception as e:
#             print(f"[SKIP] {mesh_id}: {e}")
#             continue

#     # Save index.json
#     with open(Path(save_dir) / index_out, "w") as f:
#         json.dump(index, f, indent=2)

#     print(f"\nExported {len(index)} descriptors to {save_dir}")
#     print(f"Index saved to {Path(save_dir) / index_out}")

# Example usage
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Compress descriptors to single npy/bin file.")
    parser.add_argument("--desc_dir", required=True, help="descriptors directory.")
    parser.add_argument("--output_file", required=True, help="output npy file path.")
    parser.add_argument("--platform", required=True, help="metal / cuda.")
    args = parser.parse_args()
    if args.platform == "metal":
        export_descriptors_for_metal(args.desc_dir,
                                      output_path=args.output_file)