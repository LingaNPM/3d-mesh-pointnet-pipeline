import plotly.express as px
import pandas as pd
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path

import os
import json
import torch
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.manifold import TSNE

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
    