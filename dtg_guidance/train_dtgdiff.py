"""End-to-end training pipeline for DTG-Diff generator.

Stages:
1) Base pretraining on normal data (freeze nothing)
2) Few-shot adaptation on anomalies (freeze backbone, train prototype pool + FiLM)

This script is intentionally minimal and dataset-agnostic. It expects tensors in
shape [N, C, T]. Use --data_path to load .npz files with keys:
  - normal: normal samples
  - few: few-shot anomaly samples
  - mask: (optional) binary mask for anomaly segments, same shape as samples
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dtg_guidance import (
    DiffusionBackboneConfig,
    DTGDiff,
    DiffusionSchedule,
    GaussianDiffusion1D,
    WindowConfig,
    prepare_dtg_arrays,
    prepare_splits,
    compute_stats,
    plot_stats,
    get_dataset_config,
)


@dataclass
class TrainConfig:
    device: str = "cpu"
    seed: int = 42
    batch_size: int = 64
    base_epochs: int = 10
    adapt_epochs: int = 5
    base_lr: float = 1e-4
    adapt_lr: float = 5e-5
    timesteps: int = 100
    ortho_weight: float = 1e-2
    grad_clip: Optional[float] = 1.0


class TensorDatasetWithMask(Dataset):
    def __init__(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.x = x
        self.mask = mask

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        if self.mask is None:
            return self.x[idx]
        return self.x[idx], self.mask[idx]


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _load_npz(path: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    data = np.load(path)
    if "normal" not in data or "few" not in data:
        raise ValueError(".npz must contain 'normal' and 'few' arrays")
    normal = torch.from_numpy(data["normal"]).float()
    few = torch.from_numpy(data["few"]).float()
    mask = torch.from_numpy(data["mask"]).float() if "mask" in data else None
    return normal, few, mask


def _make_loader(x: torch.Tensor, mask: Optional[torch.Tensor], batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDatasetWithMask(x, mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_base(generator: DTGDiff, diffusion: GaussianDiffusion1D, loader: DataLoader,
                cfg: TrainConfig, device: torch.device):
    generator.set_base_training()
    optim = torch.optim.Adam(generator.parameters(), lr=cfg.base_lr)

    for epoch in range(cfg.base_epochs):
        generator.train()
        total = 0.0
        for batch in loader:
            x = batch.float().to(device)
            bsz = x.size(0)
            t = torch.randint(0, diffusion.schedule.timesteps, (bsz,), device=device)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)

            loss = generator.base_loss(x_t, t, noise)
            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip)
            optim.step()
            total += loss.item() * bsz
        print(f"[Base] Epoch {epoch+1}/{cfg.base_epochs} | Loss {total / len(loader.dataset):.6f}")


def _train_adapt(generator: DTGDiff, diffusion: GaussianDiffusion1D, loader: DataLoader,
                 cfg: TrainConfig, device: torch.device):
    generator.set_adaptation_training()
    optim = torch.optim.Adam(generator.trainable_parameters(), lr=cfg.adapt_lr)

    for epoch in range(cfg.adapt_epochs):
        generator.train()
        total = 0.0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, mask = batch
                mask = mask.to(device)
            else:
                x, mask = batch, None
            x = x.float().to(device)
            bsz = x.size(0)
            t = torch.randint(0, diffusion.schedule.timesteps, (bsz,), device=device)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)

            loss = generator.adaptation_loss(x_t, t, noise, x_few=x, mask=mask)
            loss = loss + cfg.ortho_weight * generator.orthogonality_loss()

            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(generator.trainable_parameters(), cfg.grad_clip)
            optim.step()
            total += loss.item() * bsz
        print(f"[Adapt] Epoch {epoch+1}/{cfg.adapt_epochs} | Loss {total / len(loader.dataset):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to .npz with keys: normal, few, (optional) mask")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., SWaT, WADI, SMAP, SMD, MSL, NAB, IOpsCompetition)")
    parser.add_argument("--root", type=str, default=None, help="Dataset root dir for ts_datasets")
    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--time_step", type=int, default=1)
    parser.add_argument("--few_shot_count", type=int, default=50)
    parser.add_argument("--few_shot_ratio", type=float, default=None)
    parser.add_argument("--use_config_defaults", action="store_true",
                        help="Use dataset-specific defaults when available")
    parser.add_argument("--stats_path", type=str, default=None,
                        help="Path to save split anomaly ratio plot")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_epochs", type=int, default=10)
    parser.add_argument("--adapt_epochs", type=int, default=5)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--adapt_lr", type=float, default=5e-5)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--ortho_weight", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default="dtgdiff_ckpt.pt")
    args = parser.parse_args()

    cfg = TrainConfig(
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        base_epochs=args.base_epochs,
        adapt_epochs=args.adapt_epochs,
        base_lr=args.base_lr,
        adapt_lr=args.adapt_lr,
        timesteps=args.timesteps,
        ortho_weight=args.ortho_weight,
        grad_clip=args.grad_clip,
    )

    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    if args.data_path is not None:
        normal, few, mask = _load_npz(args.data_path)
    elif args.dataset is not None:
        if args.use_config_defaults:
            cfg_ds = get_dataset_config(args.dataset)
            win_cfg = WindowConfig(
                window_size=cfg_ds.window_size,
                time_step=cfg_ds.time_step,
                few_shot_count=cfg_ds.few_shot_count,
                few_shot_ratio=args.few_shot_ratio,
            )
        else:
            win_cfg = WindowConfig(
                window_size=args.window_size,
                time_step=args.time_step,
                few_shot_count=args.few_shot_count,
                few_shot_ratio=args.few_shot_ratio,
            )
        normal, few, mask = prepare_dtg_arrays(args.dataset, win_cfg, root=args.root, series_index=args.series_index)
        splits = prepare_splits(args.dataset, win_cfg, root=args.root, series_index=args.series_index)
        stats = compute_stats(splits, few)
        print("Split stats:", stats)
        if args.stats_path is not None:
            plot_stats(stats, args.stats_path)
    else:
        raise ValueError("Either --data_path or --dataset must be provided.")
    if normal.dim() != 3 or few.dim() != 3:
        raise ValueError("Expected normal/few shapes [N, C, T]")

    base_loader = _make_loader(normal, None, cfg.batch_size)
    adapt_loader = _make_loader(few, mask, cfg.batch_size)

    backbone_cfg = DiffusionBackboneConfig(in_channels=normal.size(1))
    generator = DTGDiff(backbone_cfg, num_prototypes=10, proto_dim=128, topk=3).to(device)
    diffusion = GaussianDiffusion1D(generator, DiffusionSchedule(timesteps=cfg.timesteps), device=device)

    _train_base(generator, diffusion, base_loader, cfg, device)
    _train_adapt(generator, diffusion, adapt_loader, cfg, device)

    torch.save({"model": generator.state_dict(), "config": backbone_cfg.__dict__}, args.save_path)
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
