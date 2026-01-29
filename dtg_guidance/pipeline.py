"""End-to-end helpers for blind spot detection and rectification."""
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from .failure_mining import mine_failure_sets, FailureSet
from .guidance import GuidanceState, TaskGradientGuidance, DiscriminativeSpaceGuidance


def blind_spot_detection(model, val_loader: DataLoader, device: torch.device, threshold: float = 0.5) -> FailureSet:
    """Run detector on validation data to mine FN/FP/Normal anchors."""
    return mine_failure_sets(model, val_loader, device, threshold=threshold)


def build_guidance_state(
    failure_set: FailureSet,
    task_guidance: TaskGradientGuidance,
    disc_guidance: DiscriminativeSpaceGuidance,
    failure_batch_size: int = 32,
) -> GuidanceState:
    """Prepare guidance state from a failure set."""
    failure_loader = failure_set.build_failure_loader(batch_size=failure_batch_size)
    failure_grad = task_guidance.update_failure_grad(failure_loader)
    disc_guidance.set_anchor_banks(fn_bank=failure_set.false_negatives, normal_bank=failure_set.normals)
    return GuidanceState(
        failure_grad=failure_grad,
        fn_bank=failure_set.false_negatives,
        normal_bank=failure_set.normals,
    )


def rectified_dataloader(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    synth_x: torch.Tensor,
    synth_y: Optional[torch.Tensor] = None,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Combine real and synthetic samples for retraining detectors."""
    if synth_y is None:
        synth_y = torch.ones(synth_x.size(0), dtype=train_y.dtype)
    x = torch.cat([train_x, synth_x], dim=0)
    y = torch.cat([train_y, synth_y], dim=0)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
