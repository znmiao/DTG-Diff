from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def norm(train_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_data - mean) / std, (test_data - mean) / std


def synthetic(
    length_train: int = 512,
    length_test: int = 256,
    channels: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = length_train + length_test
    t = np.linspace(0, 16 * np.pi, total, dtype=np.float32)
    data = []
    for c in range(channels):
        base = np.sin(t * (0.4 + 0.15 * c)) + 0.35 * np.cos(t * (0.9 + 0.1 * c))
        data.append(base)
    values = np.stack(data, axis=1)
    values += rng.normal(0, 0.08, size=values.shape).astype(np.float32)
    labels = np.zeros(total, dtype=np.int64)

    spans = [(120, 145), (300, 326), (590, 616), (690, 716)]
    for start, end in spans:
        end = min(end, total)
        if start >= total:
            continue
        labels[start:end] = 1
        width = end - start
        pulse = np.hanning(max(width, 3))[:width].reshape(-1, 1)
        values[start:end] += pulse * rng.normal(2.2, 0.25, size=(1, channels))

    train_data, test_data = values[:length_train], values[length_train:]
    train_labels, test_labels = labels[:length_train], labels[length_train:]
    return norm(train_data, test_data) + (train_labels, test_labels)


def _read_csv_pair(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_path = os.path.join(root, "train.csv")
    test_path = os.path.join(root, "test.csv")
    label_path = os.path.join(root, "test_label.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected train.csv and test.csv under {root}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_label = _pop_label(train_df)
    test_label = _pop_label(test_df)
    if test_label is None and os.path.exists(label_path):
        test_label = pd.read_csv(label_path).values.squeeze()
    if train_label is None:
        train_label = np.zeros(len(train_df), dtype=np.int64)
    if test_label is None:
        test_label = np.zeros(len(test_df), dtype=np.int64)
    train_data, test_data = norm(train_df.to_numpy(np.float32), test_df.to_numpy(np.float32))
    return train_data, test_data, train_label.astype(np.int64), test_label.astype(np.int64)


def _pop_label(df: pd.DataFrame) -> Optional[np.ndarray]:
    for col in ["label", "Label", "labels", "anomaly", "attack"]:
        if col in df.columns:
            values = df[col].to_numpy()
            df.drop(columns=[col], inplace=True)
            return values
    return None


def swat(root: Optional[str] = None):
    if root is None:
        raise FileNotFoundError("SWaT dataset root is required. Pass --root with train.csv and test.csv.")
    return _read_csv_pair(root)


def wadi(root: Optional[str] = None):
    if root is None:
        raise FileNotFoundError("WADI dataset root is required. Pass --root with train.csv and test.csv.")
    return _read_csv_pair(root)


def psm(root: Optional[str] = None):
    root = root or os.path.join("data", "psm")
    return _read_csv_pair(root)


def other_datasets(time_series: np.ndarray, meta: Dict[str, Any]):
    labels = np.asarray(meta.get("labels", meta.get("label", np.zeros(len(time_series)))))
    split = int(meta.get("train_split", len(time_series) // 2))
    train_data = np.asarray(time_series[:split], dtype=np.float32)
    test_data = np.asarray(time_series[split:], dtype=np.float32)
    train_labels = labels[:split].astype(np.int64)
    test_labels = labels[split:].astype(np.int64)
    train_data, test_data = norm(train_data, test_data)
    return train_data, test_data, train_labels, test_labels
