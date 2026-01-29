"""Utilities for aggregating experiment metrics."""
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key))
        else:
            try:
                out[key] = float(v)
            except Exception:
                pass
    return out


def aggregate_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    flat = [flatten_dict(r) for r in runs]
    keys = sorted({k for d in flat for k in d.keys()})
    agg = {}
    for k in keys:
        vals = [d[k] for d in flat if k in d]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        agg[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
    return agg
