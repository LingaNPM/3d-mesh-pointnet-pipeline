import os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from models.pointnet import PointNetEncoder
from losses.triplet_semihard import SemiHardTripletLoss
import torch.nn.functional as F
from utils.plot_tsne import plot_and_save_tsne
from utils.mesh_to_pointnet import mesh_to_pointnet_input
from datasets.point_cloud_dataset import PointCloudDataset


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