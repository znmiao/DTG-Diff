from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class _Metric:
    value: float

    def f1(self, score_type=None) -> float:
        return self.value


def _point_adjust(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    adjusted = preds.copy()
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] != 1:
            i += 1
            continue
        j = i
        while j < n and labels[j] == 1:
            j += 1
        if preds[i:j].any():
            adjusted[i:j] = 1
        i = j
    return adjusted


def _revised_point_adjust(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    adjusted = preds.copy()
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] != 1:
            i += 1
            continue
        j = i
        while j < n and labels[j] == 1:
            j += 1
        hits = np.flatnonzero(preds[i:j] == 1)
        if hits.size:
            adjusted[i + hits[0]:j] = 1
        i = j
    return adjusted


def _f1(labels: np.ndarray, preds: np.ndarray) -> float:
    tp = float(((labels == 1) & (preds == 1)).sum())
    fp = float(((labels == 0) & (preds == 1)).sum())
    fn = float(((labels == 1) & (preds == 0)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def _threshold(scores: np.ndarray, mode: str, nu: float, threshold: Optional[float]) -> float:
    if threshold is not None or mode == "fixed":
        if threshold is None:
            raise ValueError("threshold is required when threshold_mode='fixed'")
        return float(threshold)
    if mode in {"floating", "quantile"}:
        q = min(max(1.0 - float(nu), 0.0), 1.0)
        return float(np.quantile(scores, q))
    if mode in {"median", "mad"}:
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        return med + 3.0 * max(mad, 1e-8)
    raise ValueError(f"Unsupported threshold mode: {mode}")


def ad_predict(targets, scores, threshold_mode: str = "floating", nu: float = 0.001,
               threshold: Optional[float] = None) -> Tuple[float, _Metric, _Metric, np.ndarray, np.ndarray]:
    labels = np.asarray(targets).astype(np.int64)
    values = np.asarray(scores).astype(np.float64)
    thr = _threshold(values, threshold_mode, nu, threshold)
    preds = (values >= thr).astype(np.int64)
    pa_preds = _point_adjust(labels, preds)
    rpa_preds = _revised_point_adjust(labels, preds)
    return thr, _Metric(_f1(labels, rpa_preds)), _Metric(_f1(labels, pa_preds)), values, preds
