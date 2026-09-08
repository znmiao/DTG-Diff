"""Evaluation utilities for detector outputs and protocols (TSTR/TSRTR)."""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score

from models.anomaly_predict import ad_predict
try:
    from merlion.evaluate.anomaly import ScoreType
except ImportError:
    ScoreType = None


def collect_scores(model, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.long().to(device)
            logits = model(x)
            prob = F.softmax(logits, dim=1)[:, 1]
            all_scores.append(prob.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())
    return np.concatenate(all_scores), np.concatenate(all_targets)


def compute_metrics(targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    f1 = f1_score(targets, preds, zero_division=0)
    aupr = average_precision_score(targets, scores) if targets.max() > 0 else 0.0
    return {"F1": float(f1), "AUPR": float(aupr)}


def compute_rpa_rap(targets: np.ndarray, scores: np.ndarray, threshold_mode: str, nu: float) -> Dict[str, float]:
    _, rpa_score, pa_score, _, _ = ad_predict(targets, scores, threshold_mode, nu)
    return {
        "RPA": float(rpa_score.f1(getattr(ScoreType, "RevisedPointAdjusted", None))),
        "RAP": float(pa_score.f1(getattr(ScoreType, "PointAdjusted", None))),
    }


def evaluate_detector(model, loader, device, threshold_mode: str, nu: float) -> Dict[str, float]:
    scores, targets = collect_scores(model, loader, device)
    metrics = compute_metrics(targets, scores)
    metrics.update(compute_rpa_rap(targets, scores, threshold_mode, nu))
    return metrics
