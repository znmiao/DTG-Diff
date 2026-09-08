"""Checkpoint utilities for detector/generator/synthetic artifacts."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CheckpointManager:
    root: str

    def __post_init__(self):
        os.makedirs(self.root, exist_ok=True)

    def save_model(self, name: str, model: torch.nn.Module, extra: Optional[dict] = None):
        path = os.path.join(self.root, f"{name}.pt")
        payload = {"model": model.state_dict()}
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)
        return path

    def save_best(self, name: str, model: torch.nn.Module, metric: float, best_metric: Optional[float]):
        if best_metric is None or metric > best_metric:
            path = self.save_model(name, model, extra={"metric": metric})
            return metric, path
        return best_metric, None

    def save_numpy(self, name: str, array):
        import numpy as np
        path = os.path.join(self.root, f"{name}.npz")
        np.savez(path, data=array)
        return path

    def load_model(self, path: str, model: torch.nn.Module, map_location: Optional[str] = None):
        payload = torch.load(path, map_location=map_location or "cpu")
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
        return payload
