"""Train a CutAddPaste detector with DTG-Diff dataset splits."""
from __future__ import annotations

import argparse
import torch

from dataloader.dataset import data_generator
from dataloader.data_preprocessing import swat as load_swat
from dataloader.data_preprocessing import wadi as load_wadi
from dataloader.data_preprocessing import other_datasets
from models.CutAddPaste.network.model import base_Model
from models.CutAddPaste.trainer.trainer import Trainer
from ts_datasets.ts_datasets.anomaly import get_dataset

from dtg_guidance import WindowConfig, prepare_splits, prepare_dtg_arrays, compute_stats, plot_stats
from dtg_guidance.configs import get_config as get_dataset_config, to_cutaddpaste_config


def _load_raw_dataset(name: str, root: str = None, series_index: int = 0):
    key = name.upper()
    if key == "SWAT":
        return load_swat()
    if key == "WADI":
        return load_wadi()
    dataset = get_dataset(name, rootdir=root)
    time_series, meta = dataset[series_index]
    return other_datasets(time_series, meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg_ds = get_dataset_config(args.dataset)
    cfg = to_cutaddpaste_config(cfg_ds)

    train_data, test_data, train_labels, test_labels = _load_raw_dataset(
        args.dataset, root=args.root, series_index=args.series_index
    )

    train_dl, val_dl, test_dl, _ = data_generator(
        train_data,
        test_data,
        train_labels,
        test_labels,
        args.seed,
        cfg,
    )

    win_cfg = WindowConfig(
        window_size=cfg_ds.window_size,
        time_step=cfg_ds.time_step,
        few_shot_count=cfg_ds.few_shot_count,
        seed=args.seed,
    )
    splits = prepare_splits(args.dataset, win_cfg, root=args.root, series_index=args.series_index)
    _, few_x, _ = prepare_dtg_arrays(args.dataset, win_cfg, root=args.root, series_index=args.series_index)
    stats = compute_stats(splits, few_x)
    print("Split stats:", stats)
    if args.stats_path is not None:
        plot_stats(stats, args.stats_path)

    device = torch.device(args.device)
    model = base_Model(cfg, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight)

    Trainer(model, optimizer, train_dl, val_dl, test_dl, device, cfg, idx=0)


if __name__ == "__main__":
    main()
