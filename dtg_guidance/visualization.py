"""Visualization utilities for DTG-Diff experiments."""
from __future__ import annotations

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_compare(real: np.ndarray, synth: np.ndarray, save_path: str, n: int = 5):
    """Plot real vs synthetic samples (first channel)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(n, len(real), len(synth))
    fig, axes = plt.subplots(n, 2, figsize=(8, 2 * n))
    if n == 1:
        axes = np.array([axes])
    for i in range(n):
        axes[i, 0].plot(real[i, 0])
        axes[i, 0].set_title("Real")
        axes[i, 1].plot(synth[i, 0])
        axes[i, 1].set_title("Synthetic")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_embedding(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    method: str = "tsne",
):
    """Project embeddings to 2D with t-SNE or UMAP."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if method.lower() == "umap":
        try:
            import umap
        except Exception as exc:
            raise ImportError("UMAP not installed. Install umap-learn.") from exc
        reducer = umap.UMAP(n_components=2, random_state=42)
        proj = reducer.fit_transform(embeddings)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, init="pca")
        proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="coolwarm", s=8)
    ax.set_title(f"Embedding ({method})")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
