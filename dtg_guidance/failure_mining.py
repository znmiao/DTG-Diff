"""Failure set mining for downstream-guided generation."""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.anomaly_predict import ad_predict


@dataclass
class FailureSet:
    """Container for failure anchors and normal anchors."""
    false_negatives: Optional[torch.Tensor]
    false_positives: Optional[torch.Tensor]
    normals: Optional[torch.Tensor]

    def build_failure_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Build a loader for failure anchors with labels.

        FN samples are labeled 1 (anomaly), FP samples are labeled 0 (normal).
        """
        xs = []
        ys = []
        if self.false_negatives is not None:
            xs.append(self.false_negatives)
            ys.append(torch.ones(self.false_negatives.size(0), dtype=torch.long))
        if self.false_positives is not None:
            xs.append(self.false_positives)
            ys.append(torch.zeros(self.false_positives.size(0), dtype=torch.long))
        if not xs:
            raise ValueError("FailureSet is empty; cannot build loader.")
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _concat_or_none(chunks):
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def predict_labels_from_logits(logits: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert detector logits to anomaly probabilities and binary predictions."""
    prob = F.softmax(logits, dim=1)[:, 1]
    pred = (prob >= threshold).long()
    return prob, pred


def predict_from_model(model, x: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model forward and obtain probabilities + predictions via a dedicated function."""
    logits = model(x)
    return predict_labels_from_logits(logits, threshold)


def mine_failure_sets(
    model,
    dataloader,
    device,
    threshold: float = 0.5,
    threshold_mode: Optional[str] = None,
    nu: float = 0.001,
    predict_fn: Optional[Callable[[torch.nn.Module, torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> FailureSet:
    """Identify false negatives, false positives, and normals from a detector.

    Args:
        model: downstream detector (e.g., CutAddPaste base model).
        dataloader: yields (x, y) with y in {0,1}.
        device: torch device.
        threshold: anomaly probability threshold for predictions (when threshold_mode is None).
        threshold_mode: if provided, uses ad_predict to determine predictions (e.g., \"floating\").
        nu: quantile parameter for thresholding when using ad_predict.
        predict_fn: optional callable that returns (prob, pred). If None, uses `predict_from_model`.

    Returns:
        FailureSet with tensors or None if empty.
    """
    model.eval()
    fns, fps, normals = [], [], []
    if predict_fn is None:
        predict_fn = predict_from_model

    all_x, all_y, all_scores = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            y = y.long().to(device)
            if threshold_mode is None:
                _, pred = predict_fn(model, x, threshold)
                fn_mask = (y == 1) & (pred == 0)
                fp_mask = (y == 0) & (pred == 1)
                normal_mask = y == 0

                if fn_mask.any():
                    fns.append(x[fn_mask].detach().cpu())
                if fp_mask.any():
                    fps.append(x[fp_mask].detach().cpu())
                if normal_mask.any():
                    normals.append(x[normal_mask].detach().cpu())
            else:
                logits = model(x)
                prob = F.softmax(logits, dim=1)[:, 1]
                all_scores.append(prob.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
                all_x.append(x.detach().cpu())

    if threshold_mode is not None:
        scores = np.concatenate(all_scores) if all_scores else np.array([])
        targets = np.concatenate(all_y) if all_y else np.array([])
        if scores.size == 0:
            return FailureSet(None, None, None)
        _, _, _, _, predict = ad_predict(targets, scores, threshold_mode, nu)
        pred = torch.from_numpy(predict).long()
        y = torch.from_numpy(targets).long()
        x = torch.cat(all_x, dim=0)

        fn_mask = (y == 1) & (pred == 0)
        fp_mask = (y == 0) & (pred == 1)
        normal_mask = y == 0
        if fn_mask.any():
            fns.append(x[fn_mask])
        if fp_mask.any():
            fps.append(x[fp_mask])
        if normal_mask.any():
            normals.append(x[normal_mask])

    return FailureSet(
        false_negatives=_concat_or_none(fns),
        false_positives=_concat_or_none(fps),
        normals=_concat_or_none(normals),
    )
