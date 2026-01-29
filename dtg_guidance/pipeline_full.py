"""Paper-grade, one-click reproduction pipeline for DTG-Diff.

Stages:
1) Data split + stats + logging
2) Detector training (baseline)
3) Blind spot mining (FN/FP)
4) DTG generator training (base + adaptation)
5) Dual-guided synthesis
6) Retrain detector (TSRTR) or train on synthetic only (TSTR)
7) Evaluation + checkpointing + visualization
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dtg_guidance import (
    WindowConfig,
    prepare_splits,
    prepare_dtg_arrays,
    compute_stats,
    plot_stats,
    get_dataset_config,
    to_cutaddpaste_config,
    DiffusionBackboneConfig,
    DTGDiff,
    DiffusionSchedule,
    GaussianDiffusion1D,
    TaskGuidanceConfig,
    DiscriminativeGuidanceConfig,
    TaskGradientGuidance,
    DiscriminativeSpaceGuidance,
    DualGuidanceScheduler,
    CutAddPasteEncoder,
    PhaseScheduleConfig,
)
from dtg_guidance.failure_mining import mine_failure_sets
from dtg_guidance.detector_training import train_detector as train_detector_with_eval
from dtg_guidance.experiment_logging import ExperimentLogger
from dtg_guidance.checkpoints import CheckpointManager
from dtg_guidance.visualization import plot_sample_compare, plot_embedding


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = WindowDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def build_loaders(cfg, splits):
    train_dl = make_loader(splits["train_x"], splits["train_y"], cfg.batch_size, shuffle=True, drop_last=cfg.drop_last)
    val_dl = make_loader(splits["val_x"], splits["val_y"], cfg.batch_size, shuffle=False, drop_last=False)
    test_dl = make_loader(splits["test_x"], splits["test_y"], cfg.batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_dtg_generator(
    normal_x: np.ndarray,
    few_x: np.ndarray,
    few_mask: Optional[np.ndarray],
    in_channels: int,
    timesteps: int,
    device: torch.device,
    base_epochs: int,
    adapt_epochs: int,
    num_prototypes: int,
    proto_dim: int,
    topk: int,
    base_lr: float,
    adapt_lr: float,
    base_batch: int,
    adapt_batch: int,
):
    backbone_cfg = DiffusionBackboneConfig(in_channels=in_channels)
    generator = DTGDiff(
        backbone_cfg,
        num_prototypes=num_prototypes,
        proto_dim=proto_dim,
        topk=topk,
    ).to(device)
    diffusion = GaussianDiffusion1D(generator, DiffusionSchedule(timesteps=timesteps), device=device)

    # Base pretrain on normal data
    generator.set_base_training()
    base_loader = DataLoader(torch.from_numpy(normal_x).float(), batch_size=base_batch, shuffle=True)
    base_opt = torch.optim.Adam(generator.parameters(), lr=base_lr)
    for epoch in range(base_epochs):
        total = 0.0
        for x in base_loader:
            x = x.to(device)
            bsz = x.size(0)
            t = torch.randint(0, timesteps, (bsz,), device=device)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            loss = generator.base_loss(x_t, t, noise)
            base_opt.zero_grad()
            loss.backward()
            base_opt.step()
            total += loss.item() * bsz
        print(f"[DTG Base] Epoch {epoch+1}/{base_epochs} | Loss {total / len(base_loader.dataset):.6f}")

    # Few-shot adaptation
    generator.set_adaptation_training()
    adapt_opt = torch.optim.Adam(generator.trainable_parameters(), lr=adapt_lr)
    few_tensor = torch.from_numpy(few_x).float()
    few_mask_tensor = torch.from_numpy(few_mask).float() if few_mask is not None else None
    adapt_loader = DataLoader(
        list(zip(few_tensor, few_mask_tensor)) if few_mask_tensor is not None else few_tensor,
        batch_size=adapt_batch,
        shuffle=True,
    )

    for epoch in range(adapt_epochs):
        total = 0.0
        for batch in adapt_loader:
            if isinstance(batch, (list, tuple)):
                x, mask = batch
                mask = mask.to(device)
            else:
                x, mask = batch, None
            x = x.to(device)
            bsz = x.size(0)
            t = torch.randint(0, timesteps, (bsz,), device=device)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            loss = generator.adaptation_loss(x_t, t, noise, x_few=x, mask=mask)
            loss = loss + 1e-2 * generator.orthogonality_loss()
            adapt_opt.zero_grad()
            loss.backward()
            adapt_opt.step()
            total += loss.item() * bsz
        print(f"[DTG Adapt] Epoch {epoch+1}/{adapt_epochs} | Loss {total / len(adapt_loader.dataset):.6f}")

    generator.freeze_all()
    return generator, diffusion


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_pipeline(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    cfg_ds = get_dataset_config(args.dataset)
    cfg = to_cutaddpaste_config(cfg_ds)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join("outputs", "runs", args.dataset, timestamp)
    ensure_dir(run_dir)
    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    vis_dir = os.path.join(run_dir, "vis")
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    # Data splits & stats
    win_cfg = WindowConfig(
        window_size=cfg_ds.window_size,
        time_step=cfg_ds.time_step,
        few_shot_count=cfg_ds.few_shot_count,
        seed=args.seed,
    )
    splits = prepare_splits(args.dataset, win_cfg, root=args.root, series_index=args.series_index)
    normal_x, few_x, few_mask = prepare_dtg_arrays(args.dataset, win_cfg, root=args.root, series_index=args.series_index)
    stats = compute_stats(splits, few_x)
    save_json(os.path.join(run_dir, "split_stats.json"), stats)
    if args.stats_plot:
        plot_stats(stats, os.path.join(vis_dir, "split_stats.png"))

    # Logging + checkpoints
    logger = ExperimentLogger(log_dir, run_name="experiment")
    logger.save_metadata(vars(args), seed=args.seed)
    logger.log({"event": "split_stats", "stats": stats})
    ckpt = CheckpointManager(ckpt_dir)

    # 1) Train detector on original data
    train_dl, val_dl, test_dl = build_loaders(cfg, splits)
    detector, det_metrics = train_detector_with_eval(
        cfg,
        train_dl,
        val_dl,
        test_dl,
        device,
        checkpoint_manager=ckpt,
        log_fn=lambda d: logger.log({"event": "detector_train", **d}),
    )
    logger.log({"event": "detector_metrics", "metrics": det_metrics})

    # 2) Blind spot detection (use detector thresholding policy)
    failure_set = mine_failure_sets(
        detector,
        val_dl,
        device,
        threshold_mode=cfg.threshold_determine,
        nu=cfg.detect_nu,
    )
    if args.ablation == "fn-only":
        failure_set.false_positives = None
    elif args.ablation == "fp-only":
        failure_set.false_negatives = None

    # 3) Train DTG generator
    if few_x.shape[0] == 0:
        raise ValueError("Few-shot anomalies are empty; cannot adapt generator.")
    generator, diffusion = train_dtg_generator(
        normal_x,
        few_x,
        few_mask,
        in_channels=cfg_ds.input_channels,
        timesteps=args.timesteps,
        device=device,
        base_epochs=args.base_epochs,
        adapt_epochs=args.adapt_epochs,
        num_prototypes=args.num_prototypes,
        proto_dim=args.proto_dim,
        topk=args.topk,
        base_lr=args.base_lr,
        adapt_lr=args.adapt_lr,
        base_batch=args.base_batch,
        adapt_batch=args.adapt_batch,
    )
    ckpt.save_model("generator", generator)

    # 4) Dual-guided generation
    task_cfg = TaskGuidanceConfig(alpha=2.0, use_caga=True, per_sample_grads=True)
    disc_cfg = DiscriminativeGuidanceConfig(margin=0.5, k_neighbors=5, lambda_=1.0)

    task_guidance = TaskGradientGuidance(detector, nn.CrossEntropyLoss(), task_cfg, device)
    if args.ablation != "no-task":
        if failure_set.false_negatives is not None or failure_set.false_positives is not None:
            failure_loader = failure_set.build_failure_loader(batch_size=32)
            task_guidance.update_failure_grad(failure_loader)

    encoder = CutAddPasteEncoder(detector)
    disc_guidance = DiscriminativeSpaceGuidance(encoder, disc_cfg, device)
    if args.ablation != "no-disc":
        if failure_set.false_negatives is not None and failure_set.normals is not None:
            disc_guidance.set_anchor_banks(failure_set.false_negatives, failure_set.normals)

    phase_cfg = PhaseScheduleConfig(
        t_switch=args.t_switch,
        use_energy_ratio=args.use_energy_ratio,
        energy_ratio_threshold=args.energy_ratio_threshold,
        energy_cutoff=args.energy_cutoff,
    )
    scheduler = DualGuidanceScheduler(
        t_switch=args.t_switch,
        alpha=task_cfg.alpha,
        lambda_=disc_cfg.lambda_,
        phase_cfg=phase_cfg,
    )

    def guidance_fn(mu, x_t, x0_pred, t_tensor):
        t_int = int(t_tensor[0].item())
        grad_disc = None
        grad_task = None
        if args.ablation != "no-disc" and failure_set.false_negatives is not None and failure_set.normals is not None:
            grad_disc = disc_guidance.contrastive_gradient(x_t)
        if args.ablation != "no-task":
            target = torch.ones(x0_pred.size(0), dtype=torch.long, device=x0_pred.device)
            grad_task = task_guidance.influence_gradient(x0_pred, target, task_guidance.failure_grad)
        return scheduler.update_mean(mu, t_int, diffusion.schedule.timesteps, grad_disc, grad_task, x_t=x_t)

    num_anom = int((splits["train_y"] == 1).sum())
    synth_count = max(1, int(args.synth_ratio * max(1, num_anom)))
    x_few_t = torch.from_numpy(few_x).float().to(device)
    synth = diffusion.sample(
        batch_size=synth_count,
        shape=(cfg_ds.input_channels, cfg_ds.window_size),
        x_few=x_few_t,
        guidance_fn=guidance_fn,
    ).detach().cpu().numpy()
    ckpt.save_numpy("synthetic", synth)

    if args.save_visuals:
        plot_sample_compare(few_x, synth, save_path=os.path.join(vis_dir, "samples.png"))

    # 5) Retrain detector with synthetic anomalies
    if args.protocol.upper() == "TSTR":
        normal_train = splits["train_x"][splits["train_y"] == 0]
        train_x = np.concatenate([normal_train, synth], axis=0)
        train_y = np.concatenate([
            np.zeros(len(normal_train)),
            np.ones(len(synth)),
        ])
    else:  # TSRTR
        train_x = np.concatenate([splits["train_x"], synth], axis=0)
        train_y = np.concatenate([splits["train_y"], np.ones(len(synth))], axis=0)

    aug_splits = dict(splits)
    aug_splits["train_x"] = train_x
    aug_splits["train_y"] = train_y
    train_dl2, val_dl2, test_dl2 = build_loaders(cfg, aug_splits)
    detector2, det2_metrics = train_detector_with_eval(
        cfg,
        train_dl2,
        val_dl2,
        test_dl2,
        device,
        checkpoint_manager=ckpt,
        log_fn=lambda d: logger.log({"event": "detector_retrain", **d}),
    )
    logger.log({"event": "detector_retrain_metrics", "metrics": det2_metrics})

    if args.save_visuals:
        encoder = CutAddPasteEncoder(detector2)
        with torch.no_grad():
            emb_real = encoder(torch.from_numpy(few_x).float().to(device)).cpu().numpy()
            emb_synth = encoder(torch.from_numpy(synth).float().to(device)).cpu().numpy()
        labels = np.concatenate([np.zeros(len(emb_real)), np.ones(len(emb_synth))])
        emb = np.concatenate([emb_real, emb_synth], axis=0)
        plot_embedding(emb, labels, save_path=os.path.join(vis_dir, "embed_tsne.png"), method="tsne")

    summary = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "ablation": args.ablation,
        "detector": det_metrics,
        "detector_retrain": det2_metrics,
        "run_dir": run_dir,
    }
    save_json(os.path.join(run_dir, "summary.json"), summary)
    return summary


def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_args(args, cfg: dict):
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--protocol", type=str, default="TSRTR", choices=["TSTR", "TSRTR"])
    parser.add_argument("--stats_plot", action="store_true")
    parser.add_argument("--save_visuals", action="store_true")

    # DTG training
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--base_epochs", type=int, default=5)
    parser.add_argument("--adapt_epochs", type=int, default=3)
    parser.add_argument("--synth_ratio", type=float, default=1.0)
    parser.add_argument("--num_prototypes", type=int, default=10)
    parser.add_argument("--proto_dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--adapt_lr", type=float, default=5e-5)
    parser.add_argument("--base_batch", type=int, default=64)
    parser.add_argument("--adapt_batch", type=int, default=32)

    # Guidance / scheduling
    parser.add_argument("--ablation", type=str, default="full",
                        help="full | no-disc | no-task | fn-only | fp-only")
    parser.add_argument("--t_switch", type=float, default=0.3)
    parser.add_argument("--use_energy_ratio", action="store_true")
    parser.add_argument("--energy_ratio_threshold", type=float, default=0.3)
    parser.add_argument("--energy_cutoff", type=float, default=0.5)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    merge_args(args, cfg)
    run_pipeline(args)


if __name__ == "__main__":
    main()
