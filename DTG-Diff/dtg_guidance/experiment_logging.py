"""Lightweight JSON logging and metadata capture for experiments."""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def _git_commit(root: Optional[str] = None) -> Optional[str]:
    root = root or os.getcwd()
    git_head = os.path.join(root, ".git", "HEAD")
    if not os.path.exists(git_head):
        return None
    with open(git_head, "r", encoding="utf-8") as f:
        ref = f.read().strip()
    if ref.startswith("ref:"):
        ref_path = ref.split(" ", 1)[1]
        ref_file = os.path.join(root, ".git", ref_path)
        if os.path.exists(ref_file):
            with open(ref_file, "r", encoding="utf-8") as rf:
                return rf.read().strip()
        return None
    return ref


def collect_env_info() -> Dict[str, Any]:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except Exception:
        info["torch"] = None
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except Exception:
        info["numpy"] = None
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        info["sklearn"] = None
    return info


class ExperimentLogger:
    def __init__(self, log_dir: str, run_name: str = "run"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.run_name = run_name
        self.log_path = os.path.join(log_dir, f"{run_name}.jsonl")
        self.meta_path = os.path.join(log_dir, f"{run_name}_meta.json")

    def log(self, data: Dict[str, Any]):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=True) + "\n")

    def save_metadata(self, params: Dict[str, Any], seed: int, root: Optional[str] = None):
        meta = {
            "params": params,
            "seed": seed,
            "env": collect_env_info(),
            "git_commit": _git_commit(root),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

    def path(self) -> str:
        return self.log_path
