"""Detector training with evaluation and best-checkpoint saving."""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.CutAddPaste.network.model import base_Model
from dtg_guidance.evaluation import evaluate_detector


def train_detector(
    cfg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    max_epochs: Optional[int] = None,
    checkpoint_manager=None,
    log_fn=None,
) -> Tuple[base_Model, Dict[str, float]]:
    model = base_Model(cfg, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight)
    max_epochs = max_epochs or cfg.num_epoch
    best_metric = None
    best_metrics = {}

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.long().to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        val_metrics = evaluate_detector(model, val_loader, device, cfg.threshold_determine, cfg.detect_nu)
        metric_key = "RPA"
        metric_val = val_metrics.get(metric_key, 0.0)
        if checkpoint_manager is not None:
            best_metric, path = checkpoint_manager.save_best(
                f"detector_best", model, metric_val, best_metric
            )
            if path is not None:
                best_metrics = val_metrics
        if log_fn is not None:
            log_fn({
                "epoch": epoch,
                "train_loss": total_loss / max(1, total),
                "val_metrics": val_metrics,
            })

    test_metrics = evaluate_detector(model, test_loader, device, cfg.threshold_determine, cfg.detect_nu)
    return model, {"val": best_metrics, "test": test_metrics}
