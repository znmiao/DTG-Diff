"""Dataset processing utilities for DTG-Diff.

Provides dataset loaders consistent with CutAddPaste preprocessing and
windowing logic. Outputs tensors in [N, C, T] format and optional
per-window anomaly masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import subsequences
from dataloader.data_preprocessing import swat as load_swat
from dataloader.data_preprocessing import wadi as load_wadi
from dataloader.data_preprocessing import synthetic as load_synthetic
from dataloader.data_preprocessing import other_datasets, norm as norm_data
try:
    from ts_datasets.ts_datasets.anomaly import get_dataset
except ImportError:
    get_dataset = None


@dataclass
class WindowConfig:
    window_size: int = 100
    time_step: int = 1
    few_shot_count: int = 50
    few_shot_ratio: Optional[float] = None
    use_train_anomalies: bool = True
    val_ratio: float = 0.2
    seed: int = 42


def _window_labels(label_seq: np.ndarray, window_size: int, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return window-level labels and per-window masks.

    labels: [T]
    returns:
      window_labels: [N]
      window_masks: [N, window_size]
    """
    label_windows = subsequences(label_seq, window_size, time_step)
    if label_windows.ndim != 2:
        raise ValueError("Expected label windows to be 2D.")
    window_labels = (label_windows.sum(axis=1) > 0).astype(np.int64)
    window_masks = (label_windows > 0).astype(np.float32)
    return window_labels, window_masks


def _window_data(data: np.ndarray, window_size: int, time_step: int) -> np.ndarray:
    """Create data windows in shape [N, C, T]."""
    windows = subsequences(data, window_size, time_step)
    if windows.ndim != 3:
        raise ValueError("Expected data windows to be 3D.")
    return windows.transpose(0, 2, 1)


def load_dataset(name: str, root: Optional[str] = None, series_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset with CutAddPaste-consistent preprocessing.

    Returns: train_data, test_data, train_labels, test_labels
    """
    name_upper = name.upper()
    if name_upper in {"SYNTHETIC", "TOY"}:
        return load_synthetic(seed=series_index)
    if name_upper == "SWAT":
        return load_swat(root)
    if name_upper == "WADI":
        return load_wadi(root)
    if name_upper == "PSM":
        return load_psm(root)

    if get_dataset is None:
        raise ImportError("ts_datasets is required for datasets outside SWaT, WADI, PSM, and synthetic.")
    dataset = get_dataset(name, rootdir=root)
    time_series, meta = dataset[series_index]
    train_data, test_data, train_labels, test_labels = other_datasets(time_series, meta)
    return train_data, test_data, train_labels.squeeze(), test_labels.squeeze()


def _read_psm_csv(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    label = None
    for col in ["label", "Label", "labels", "anomaly"]:
        if col in df.columns:
            label = df[col].to_numpy()
            df = df.drop(columns=[col])
            break
    data = df.to_numpy()
    return data, label


def load_psm(root: Optional[str] = None, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = root or os.path.join("data", "psm")
    train_path = os.path.join(root, "train.csv")
    test_path = os.path.join(root, "test.csv")
    test_label_path = os.path.join(root, "test_label.csv")

    train_data, train_label = _read_psm_csv(train_path)
    test_data, test_label = _read_psm_csv(test_path)

    if test_label is None and os.path.exists(test_label_path):
        test_label = pd.read_csv(test_label_path).values.squeeze()
    if train_label is None:
        train_label = np.zeros(len(train_data))
    if test_label is None:
        test_label = np.zeros(len(test_data))

    if normalize:
        train_data, test_data = norm_data(train_data, test_data)
    return train_data, test_data, train_label, test_label


def build_windows(
    train_data: np.ndarray,
    test_data: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    cfg: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate windows and masks for train/test."""
    train_x = _window_data(train_data, cfg.window_size, cfg.time_step)
    test_x = _window_data(test_data, cfg.window_size, cfg.time_step)

    train_y, train_mask = _window_labels(train_labels, cfg.window_size, cfg.time_step)
    test_y, test_mask = _window_labels(test_labels, cfg.window_size, cfg.time_step)
    return train_x, train_y, train_mask, test_x, test_y, test_mask


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    return x[train_idx], y[train_idx], mask[train_idx], x[val_idx], y[val_idx], mask[val_idx]


def _select_few_shot(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cfg: WindowConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.where(y == 1)[0]
    if len(idx) == 0:
        return np.empty((0,) + x.shape[1:], dtype=x.dtype), np.empty((0,) + mask.shape[1:], dtype=mask.dtype)

    if cfg.few_shot_ratio is not None:
        count = max(1, int(len(idx) * cfg.few_shot_ratio))
    else:
        count = min(cfg.few_shot_count, len(idx))

    rng = rng or np.random.default_rng(cfg.seed)
    choice = rng.choice(idx, size=count, replace=False)
    return x[choice], mask[choice]


def prepare_dtg_arrays(
    name: str,
    cfg: WindowConfig,
    root: Optional[str] = None,
    series_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Prepare arrays for DTG-Diff training.

    Returns:
      normal_windows: [N, C, T]
      few_windows: [M, C, T]
      few_masks: [M, T]
    """
    train_data, test_data, train_labels, test_labels = load_dataset(name, root=root, series_index=series_index)
    train_x, train_y, train_mask, test_x, test_y, test_mask = build_windows(
        train_data, test_data, train_labels, test_labels, cfg
    )

    normal_windows = train_x[train_y == 0]

    rng = np.random.default_rng(cfg.seed)
    if cfg.use_train_anomalies:
        few_x, few_mask = _select_few_shot(train_x, train_y, train_mask, cfg, rng=rng)
        if few_x.shape[0] == 0:
            few_x, few_mask = _select_few_shot(test_x, test_y, test_mask, cfg, rng=rng)
    else:
        few_x, few_mask = _select_few_shot(test_x, test_y, test_mask, cfg, rng=rng)

    return normal_windows, few_x, few_mask


def prepare_splits(
    name: str,
    cfg: WindowConfig,
    root: Optional[str] = None,
    series_index: int = 0,
) -> Dict[str, np.ndarray]:
    train_data, test_data, train_labels, test_labels = load_dataset(name, root=root, series_index=series_index)
    train_x, train_y, train_mask, test_x, test_y, test_mask = build_windows(
        train_data, test_data, train_labels, test_labels, cfg
    )
    tr_x, tr_y, tr_m, val_x, val_y, val_m = split_train_val(
        train_x, train_y, train_mask, cfg.val_ratio, cfg.seed
    )
    return {
        "train_x": tr_x,
        "train_y": tr_y,
        "train_mask": tr_m,
        "val_x": val_x,
        "val_y": val_y,
        "val_mask": val_m,
        "test_x": test_x,
        "test_y": test_y,
        "test_mask": test_mask,
    }


def compute_stats(splits: Dict[str, np.ndarray], few_x: np.ndarray) -> Dict[str, float]:
    stats = {}
    for split in ["train", "val", "test"]:
        y = splits[f"{split}_y"]
        stats[f"{split}_count"] = float(len(y))
        stats[f"{split}_anom_ratio"] = float(y.mean()) if len(y) else 0.0
    total_anom = float((splits["train_y"] == 1).sum() + (splits["val_y"] == 1).sum())
    stats["few_shot_count"] = float(len(few_x))
    stats["few_shot_coverage"] = float(len(few_x) / max(1.0, total_anom))
    return stats


def plot_stats(stats: Dict[str, float], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = ["train", "val", "test"]
    ratios = [stats[f"{k}_anom_ratio"] for k in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, ratios, color="#4C72B0")
    axes[0].set_ylabel("Anomaly Ratio")
    axes[0].set_ylim(0, max(ratios + [0.05]))
    axes[0].set_title("Split Anomaly Ratios")

    coverage = stats.get("few_shot_coverage", 0.0)
    axes[1].bar(["few-shot"], [coverage], color="#55A868")
    axes[1].set_ylim(0, max(coverage + 0.05, 0.1))
    axes[1].set_title("Few-shot Coverage")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_npz(path: str, normal: np.ndarray, few: np.ndarray, mask: Optional[np.ndarray]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mask is None:
        np.savez(path, normal=normal, few=few)
    else:
        np.savez(path, normal=normal, few=few, mask=mask)


def prepare_and_save(
    name: str,
    output_path: str,
    cfg: WindowConfig,
    root: Optional[str] = None,
    series_index: int = 0,
):
    normal, few, mask = prepare_dtg_arrays(name, cfg, root=root, series_index=series_index)
    save_npz(output_path, normal, few, mask)
    return output_path
