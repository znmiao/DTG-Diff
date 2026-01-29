"""Minimal example wiring DTG guidance with CutAddPaste detector.

This file is intentionally lightweight and focuses on how to call the
core guidance modules during diffusion sampling.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.CutAddPaste.network.model import base_Model
from dtg_guidance import (
    DiscriminativeGuidanceConfig,
    TaskGuidanceConfig,
    CutAddPasteEncoder,
    TaskGradientGuidance,
    DiscriminativeSpaceGuidance,
    DualGuidanceScheduler,
    DiffusionBackboneConfig,
    DiffusionSchedule,
    DTGDiff,
    GaussianDiffusion1D,
)


def build_detector(configs, device):
    model = base_Model(configs, device).to(device)
    model.eval()
    return model


def build_failure_loader(failure_x, failure_y, batch_size=32):
    dataset = TensorDataset(failure_x, failure_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def guided_mean_update(mu, x_t, x0_pred, t, T, task_guidance, disc_guidance, scheduler):
    """Compute guided mean for one diffusion step.

    Args:
        mu: predicted mean from diffusion model (tensor).
        x_t: current noisy sample (tensor).
        x0_pred: predicted clean sample (tensor).
        t: current step index (int).
        T: total steps (int).
    """
    grad_disc = disc_guidance.contrastive_gradient(x_t)
    target = torch.ones(x0_pred.size(0), dtype=torch.long, device=x0_pred.device)
    grad_task = task_guidance.influence_gradient(x0_pred, target, task_guidance.failure_grad)
    mu_tilde = scheduler.update_mean(mu, t, T, grad_disc, grad_task)
    return mu_tilde


# The following is a schematic usage pattern for integration.
# Replace `dummy_x_t`, `dummy_x0`, and `mu` with real diffusion outputs.
if __name__ == "__main__":
    from conf.cut_add_paste.IOpsCompetition_Configs import Config as Configs

    device = torch.device("cpu")
    configs = Configs()
    detector = build_detector(configs, device)

    # Example anchors (replace with mined anchors from validation set).
    fn_x = torch.randn(16, configs.input_channels, configs.window_size)  # false negatives
    fp_x = torch.randn(16, configs.input_channels, configs.window_size)  # false positives
    normal_x = torch.randn(64, configs.input_channels, configs.window_size)

    failure_x = torch.cat([fn_x, fp_x], dim=0)
    failure_y = torch.cat([
        torch.ones(fn_x.size(0), dtype=torch.long),
        torch.zeros(fp_x.size(0), dtype=torch.long),
    ], dim=0)
    failure_loader = build_failure_loader(failure_x, failure_y)

    task_cfg = TaskGuidanceConfig(epsilon=1e-2, alpha=2.0, use_caga=True, per_sample_grads=True)
    disc_cfg = DiscriminativeGuidanceConfig(margin=0.5, k_neighbors=5, lambda_=1.0)

    task_guidance = TaskGradientGuidance(detector, torch.nn.CrossEntropyLoss(), task_cfg, device)
    task_guidance.update_failure_grad(failure_loader)

    encoder = CutAddPasteEncoder(detector)
    disc_guidance = DiscriminativeSpaceGuidance(encoder, disc_cfg, device)
    disc_guidance.set_anchor_banks(fn_bank=fn_x, normal_bank=normal_x)

    scheduler = DualGuidanceScheduler(t_switch=0.3, alpha=task_cfg.alpha, lambda_=disc_cfg.lambda_)

    # --- Minimal diffusion generator wiring (demo) ---
    backbone_cfg = DiffusionBackboneConfig(in_channels=configs.input_channels)
    generator = DTGDiff(backbone_cfg, num_prototypes=10, proto_dim=128).to(device)
    generator.freeze_all()

    diffusion = GaussianDiffusion1D(generator, DiffusionSchedule(timesteps=100), device=device)

    def guidance_fn(mu, x_t, x0_pred, t_tensor):
        t_int = int(t_tensor[0].item())
        return guided_mean_update(mu, x_t, x0_pred, t_int, diffusion.schedule.timesteps,
                                  task_guidance=task_guidance,
                                  disc_guidance=disc_guidance,
                                  scheduler=scheduler)

    # x_few acts as the conditioning few-shot anomalies for the prototype pool.
    x_few = fn_x.to(device)
    samples = diffusion.sample(batch_size=4,
                               shape=(configs.input_channels, configs.window_size),
                               x_few=x_few,
                               guidance_fn=guidance_fn)
    print(samples.shape)
