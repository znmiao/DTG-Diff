"""Evaluate TSTR and TSRTR protocols for DTG-Diff."""
from __future__ import annotations

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dtg_guidance import (
    WindowConfig,
    prepare_splits,
    prepare_dtg_arrays,
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
)
from dtg_guidance.detector_training import train_detector as train_detector_with_eval
from dtg_guidance.failure_mining import mine_failure_sets
from dtg_guidance.pipeline_full import PIPELINE_DEFAULTS


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = WindowDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_dtg_generator(normal_x, few_x, few_mask, in_channels, timesteps, device,
                        base_epochs, adapt_epochs, base_batch, adapt_batch,
                        base_lr, adapt_lr, num_prototypes, proto_dim, topk,
                        ortho_weight):
    backbone_cfg = DiffusionBackboneConfig(in_channels=in_channels)
    generator = DTGDiff(backbone_cfg, num_prototypes=num_prototypes, proto_dim=proto_dim, topk=topk).to(device)
    diffusion = GaussianDiffusion1D(generator, DiffusionSchedule(timesteps=timesteps), device=device)

    generator.set_base_training()
    base_loader = DataLoader(torch.from_numpy(normal_x).float(), batch_size=base_batch, shuffle=True)
    base_opt = torch.optim.Adam(generator.parameters(), lr=base_lr)
    for _ in range(base_epochs):
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

    generator.set_adaptation_training()
    adapt_opt = torch.optim.Adam(generator.trainable_parameters(), lr=adapt_lr)
    few_tensor = torch.from_numpy(few_x).float()
    if few_mask is None:
        few_mask_tensor = None
    else:
        few_mask_tensor = torch.from_numpy(few_mask).float()
    adapt_loader = DataLoader(
        list(zip(few_tensor, few_mask_tensor)) if few_mask_tensor is not None else few_tensor,
        batch_size=adapt_batch,
        shuffle=True,
    )

    for _ in range(adapt_epochs):
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
            loss = loss + ortho_weight * generator.orthogonality_loss()
            adapt_opt.zero_grad()
            loss.backward()
            adapt_opt.step()

    generator.freeze_all()
    return generator, diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", type=str, default="TSRTR", choices=["TSTR", "TSRTR"])
    parser.add_argument("--timesteps", type=int, default=PIPELINE_DEFAULTS["timesteps"])
    parser.add_argument("--base_epochs", type=int, default=PIPELINE_DEFAULTS["base_epochs"])
    parser.add_argument("--adapt_epochs", type=int, default=PIPELINE_DEFAULTS["adapt_epochs"])
    parser.add_argument("--base_batch", type=int, default=PIPELINE_DEFAULTS["base_batch"])
    parser.add_argument("--adapt_batch", type=int, default=PIPELINE_DEFAULTS["adapt_batch"])
    parser.add_argument("--base_lr", type=float, default=PIPELINE_DEFAULTS["base_lr"])
    parser.add_argument("--adapt_lr", type=float, default=PIPELINE_DEFAULTS["adapt_lr"])
    parser.add_argument("--num_prototypes", type=int, default=PIPELINE_DEFAULTS["num_prototypes"])
    parser.add_argument("--proto_dim", type=int, default=PIPELINE_DEFAULTS["proto_dim"])
    parser.add_argument("--topk", type=int, default=PIPELINE_DEFAULTS["topk"])
    parser.add_argument("--task_alpha", type=float, default=PIPELINE_DEFAULTS["task_alpha"])
    parser.add_argument("--task_epsilon", type=float, default=PIPELINE_DEFAULTS["task_epsilon"])
    parser.add_argument("--task_preconditioner", type=str, default=PIPELINE_DEFAULTS["task_preconditioner"], choices=["l2", "cosine", "none"])
    parser.add_argument("--disc_lambda", type=float, default=PIPELINE_DEFAULTS["disc_lambda"])
    parser.add_argument("--disc_margin", type=float, default=PIPELINE_DEFAULTS["disc_margin"])
    parser.add_argument("--disc_neighbors", type=int, default=PIPELINE_DEFAULTS["disc_neighbors"])
    parser.add_argument("--t_switch", type=float, default=PIPELINE_DEFAULTS["t_switch"])
    parser.add_argument("--ortho_weight", type=float, default=PIPELINE_DEFAULTS["ortho_weight"])
    parser.add_argument("--failure_batch_size", type=int, default=PIPELINE_DEFAULTS["failure_batch_size"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    cfg_ds = get_dataset_config(args.dataset)
    cfg = to_cutaddpaste_config(cfg_ds)

    win_cfg = WindowConfig(window_size=cfg_ds.window_size, time_step=cfg_ds.time_step, few_shot_count=cfg_ds.few_shot_count, seed=args.seed)
    splits = prepare_splits(args.dataset, win_cfg, root=args.root, series_index=args.series_index)
    normal_x, few_x, few_mask = prepare_dtg_arrays(args.dataset, win_cfg, root=args.root, series_index=args.series_index)

    train_dl = make_loader(splits["train_x"], splits["train_y"], cfg.batch_size, shuffle=True)
    val_dl = make_loader(splits["val_x"], splits["val_y"], cfg.batch_size, shuffle=False)
    test_dl = make_loader(splits["test_x"], splits["test_y"], cfg.batch_size, shuffle=False)

    detector, _ = train_detector_with_eval(cfg, train_dl, val_dl, test_dl, device)
    failure_set = mine_failure_sets(detector, val_dl, device)

    generator, diffusion = train_dtg_generator(
        normal_x, few_x, few_mask, cfg_ds.input_channels, args.timesteps, device,
        args.base_epochs, args.adapt_epochs, args.base_batch, args.adapt_batch,
        args.base_lr, args.adapt_lr, args.num_prototypes, args.proto_dim, args.topk,
        args.ortho_weight,
    )

    task_cfg = TaskGuidanceConfig(
        epsilon=args.task_epsilon,
        alpha=args.task_alpha,
        use_caga=True,
        per_sample_grads=True,
        preconditioner=args.task_preconditioner,
    )
    disc_cfg = DiscriminativeGuidanceConfig(margin=args.disc_margin, k_neighbors=args.disc_neighbors, lambda_=args.disc_lambda)
    task_guidance = TaskGradientGuidance(detector, nn.CrossEntropyLoss(), task_cfg, device)
    if failure_set.false_negatives is not None or failure_set.false_positives is not None:
        failure_loader = failure_set.build_failure_loader(batch_size=args.failure_batch_size)
        task_guidance.update_failure_grad(failure_loader)
    encoder = CutAddPasteEncoder(detector)
    disc_guidance = DiscriminativeSpaceGuidance(encoder, disc_cfg, device)
    if failure_set.false_negatives is not None and failure_set.normals is not None:
        disc_guidance.set_anchor_banks(failure_set.false_negatives, failure_set.normals)
    scheduler = DualGuidanceScheduler(t_switch=args.t_switch, alpha=task_cfg.alpha, lambda_=disc_cfg.lambda_)

    def guidance_fn(mu, x_t, x0_pred, t_tensor):
        t_int = int(t_tensor[0].item())
        grad_disc = None
        if failure_set.false_negatives is not None and failure_set.normals is not None:
            grad_disc = disc_guidance.contrastive_gradient(x_t)
        target = torch.ones(x0_pred.size(0), dtype=torch.long, device=x0_pred.device)
        grad_task = task_guidance.influence_gradient(x0_pred, target, task_guidance.failure_grad)
        return scheduler.update_mean(mu, t_int, diffusion.schedule.timesteps, grad_disc, grad_task)

    synth_count = max(1, int((splits["train_y"] == 1).sum()))
    x_few_t = torch.from_numpy(few_x).float().to(device)
    synth = diffusion.sample(
        batch_size=synth_count,
        shape=(cfg_ds.input_channels, cfg_ds.window_size),
        x_few=x_few_t,
        guidance_fn=guidance_fn,
    ).detach().cpu().numpy()

    if args.protocol == "TSTR":
        train_x = np.concatenate([splits["train_x"][splits["train_y"] == 0], synth], axis=0)
        train_y = np.concatenate([
            np.zeros(len(splits["train_x"][splits["train_y"] == 0])),
            np.ones(len(synth)),
        ])
    else:
        train_x = np.concatenate([splits["train_x"], synth], axis=0)
        train_y = np.concatenate([splits["train_y"], np.ones(len(synth))], axis=0)

    train_dl2 = make_loader(train_x, train_y, cfg.batch_size, shuffle=True)
    detector2, metrics = train_detector_with_eval(cfg, train_dl2, val_dl, test_dl, device)
    print(f"Protocol {args.protocol} metrics:", metrics)


if __name__ == "__main__":
    main()
