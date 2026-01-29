"""Core guidance modules for DTG-Diff-style sampling."""
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn

from .config import DiscriminativeGuidanceConfig, TaskGuidanceConfig, PhaseScheduleConfig


def _flatten_grads(grads, device: Optional[torch.device] = None) -> torch.Tensor:
    """Flatten a list of parameter gradients into a single vector."""
    flat = []
    for g in grads:
        if g is None:
            continue
        flat.append(g.reshape(-1))
    if not flat:
        # Fallback if all grads are None (rare but safe).
        if device is None:
            for g in grads:
                if g is not None:
                    device = g.device
                    break
        if device is None:
            device = torch.device("cpu")
        return torch.zeros(1, device=device)
    return torch.cat(flat, dim=0)


def _caga_project(grads: torch.Tensor, eps: float = 1e-12, use_updated: bool = False) -> torch.Tensor:
    """Conflict-Aware Gradient Aggregation (CAGA).

    Args:
        grads: tensor of shape [N, P] where N is number of failure samples.
        use_updated: whether to use progressively updated gradients for projection.
    """
    if grads.dim() != 2 or grads.size(0) == 1:
        return grads.sum(dim=0) if grads.dim() == 2 else grads

    order = torch.randperm(grads.size(0), device=grads.device)
    updated = grads.clone()

    for i in order:
        gi = updated[i]
        for j in order:
            if i == j:
                continue
            gj = updated[j] if use_updated else grads[j]
            dot = torch.dot(gi, gj)
            if dot < 0:
                gi = gi - (dot / (gj.norm() ** 2 + eps)) * gj
        updated[i] = gi

    return updated.sum(dim=0)


def high_frequency_energy_ratio(x: torch.Tensor, cutoff: float = 0.5, eps: float = 1e-12) -> torch.Tensor:
    """Compute high-frequency energy ratio over the last dimension."""
    if x.dim() != 3:
        raise ValueError("Expected x with shape [B, C, T].")
    freq = torch.fft.rfft(x, dim=-1)
    power = freq.abs().pow(2)
    cutoff_idx = max(1, int(power.size(-1) * cutoff))
    high = power[..., cutoff_idx:].sum(dim=-1)
    total = power.sum(dim=-1) + eps
    ratio = (high / total).mean()
    return ratio


def _subsample_bank(bank: torch.Tensor, max_size: Optional[int]) -> torch.Tensor:
    if bank is None or max_size is None or bank.size(0) <= max_size:
        return bank
    idx = torch.randperm(bank.size(0), device=bank.device)[:max_size]
    return bank[idx]


def _per_sample_loss_fn(loss_fn: nn.Module) -> Optional[nn.Module]:
    """Build a per-sample loss if possible."""
    if hasattr(loss_fn, "reduction") and loss_fn.reduction == "none":
        return loss_fn
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return nn.CrossEntropyLoss(
            weight=loss_fn.weight,
            ignore_index=loss_fn.ignore_index,
            reduction="none",
            label_smoothing=getattr(loss_fn, "label_smoothing", 0.0),
        )
    if isinstance(loss_fn, nn.BCEWithLogitsLoss):
        return nn.BCEWithLogitsLoss(
            weight=loss_fn.weight,
            pos_weight=loss_fn.pos_weight,
            reduction="none",
        )
    if isinstance(loss_fn, nn.MSELoss):
        return nn.MSELoss(reduction="none")
    return None


@dataclass
class GuidanceState:
    """State carried across sampling steps."""
    failure_grad: Optional[torch.Tensor] = None
    fn_bank: Optional[torch.Tensor] = None
    normal_bank: Optional[torch.Tensor] = None


class TaskGradientGuidance:
    """Downstream task-guided influence alignment."""

    def __init__(self, model: nn.Module, loss_fn: nn.Module, config: TaskGuidanceConfig, device: torch.device):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.failure_grad = None
        self.loss_fn_per_sample = _per_sample_loss_fn(loss_fn) if config.per_sample_grads else None

    def compute_failure_gradient(self, failure_loader, max_batches: Optional[int] = None) -> torch.Tensor:
        """Aggregate gradients over failure anchors.

        Returns a vector in parameter space to be aligned with generated samples.
        """
        self.model.eval()
        grads = []
        for i, (x, y) in enumerate(failure_loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            logits = self.model(x)
            if self.loss_fn_per_sample is not None:
                losses = self.loss_fn_per_sample(logits, y)
                for idx in range(losses.size(0)):
                    retain = idx < losses.size(0) - 1
                    g = torch.autograd.grad(
                        losses[idx],
                        self.model.parameters(),
                        retain_graph=retain,
                        create_graph=False,
                    )
                    grads.append(_flatten_grads(g, device=self.device).detach())
            else:
                loss = self.loss_fn(logits, y)
                g = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False, create_graph=False)
                grads.append(_flatten_grads(g, device=self.device).detach())

        if not grads:
            return None

        stacked = torch.stack(grads, dim=0)
        if self.config.use_caga:
            agg = _caga_project(
                stacked,
                eps=self.config.grad_eps,
                use_updated=self.config.caga_use_updated,
            )
        else:
            agg = stacked.sum(dim=0)

        # Negative scaling follows the paper's failure-corrective direction.
        return -self.config.epsilon * agg

    def update_failure_grad(self, failure_loader, max_batches: Optional[int] = None) -> Optional[torch.Tensor]:
        self.failure_grad = self.compute_failure_gradient(failure_loader, max_batches=max_batches)
        return self.failure_grad

    def influence_gradient(self, x: torch.Tensor, target: torch.Tensor,
                           failure_grad: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute influence gradient with respect to the input sample x.

        Args:
            x: generated sample (typically x0 prediction) with requires_grad.
            target: class labels for x.
            failure_grad: aggregated parameter-space gradient.
        """
        if failure_grad is None:
            return torch.zeros_like(x)
        if x.requires_grad is False:
            x = x.requires_grad_(True)

        logits = self.model(x)
        loss = self.loss_fn(logits, target)
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True, retain_graph=True)
        g_vec = _flatten_grads(grads, device=x.device)
        denom = g_vec.pow(2).sum().detach() + self.config.grad_eps
        align = torch.dot(failure_grad.to(g_vec.device), g_vec) / denom
        grad_x = torch.autograd.grad(align, x, retain_graph=True)[0]
        return grad_x


class DiscriminativeSpaceGuidance:
    """Discriminative-space guidance via summed-margin triplet loss."""

    def __init__(self, encoder: nn.Module, config: DiscriminativeGuidanceConfig, device: torch.device):
        self.encoder = encoder
        self.config = config
        self.device = device
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.fn_bank_z = None
        self.normal_bank_z = None

    def _encode_bank(self, bank: torch.Tensor, batch_size: int) -> torch.Tensor:
        if bank is None:
            return None
        self.encoder.eval()
        zs = []
        with torch.no_grad():
            for i in range(0, bank.size(0), batch_size):
                x = bank[i:i + batch_size].to(self.device)
                z = self.encoder(x)
                zs.append(z.detach())
        return torch.cat(zs, dim=0)

    def set_anchor_banks(self, fn_bank: torch.Tensor, normal_bank: torch.Tensor,
                         precomputed: bool = False):
        fn_bank = _subsample_bank(fn_bank, self.config.max_bank_size)
        normal_bank = _subsample_bank(normal_bank, self.config.max_bank_size)

        if precomputed:
            self.fn_bank_z = fn_bank.to(self.device) if fn_bank is not None else None
            self.normal_bank_z = normal_bank.to(self.device) if normal_bank is not None else None
            return

        self.fn_bank_z = self._encode_bank(fn_bank, self.config.bank_batch_size)
        self.normal_bank_z = self._encode_bank(normal_bank, self.config.bank_batch_size)

    def _select_pos_neg(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select nearest positive and K nearest negatives in embedding space."""
        if self.fn_bank_z is None or self.normal_bank_z is None:
            raise ValueError("Anchor banks must be set before guidance.")

        # Nearest FN anchor (positive)
        d_pos = torch.cdist(z, self.fn_bank_z, p=2)
        pos_idx = torch.argmin(d_pos, dim=1)
        z_pos = self.fn_bank_z[pos_idx]

        # K nearest normal neighbors (negative)
        d_neg = torch.cdist(z, self.normal_bank_z, p=2)
        k = min(self.config.k_neighbors, d_neg.size(1))
        neg_idx = torch.topk(d_neg, k=k, largest=False, dim=1).indices
        z_neg = self.normal_bank_z[neg_idx]
        return z_pos, z_neg

    def contrastive_gradient(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad is False:
            x = x.requires_grad_(True)
        z = self.encoder(x)
        z_pos, z_neg = self._select_pos_neg(z)

        d_pos = (z - z_pos).pow(2).sum(dim=1, keepdim=True)
        d_neg = (z.unsqueeze(1) - z_neg).pow(2).sum(dim=2)
        loss = torch.relu(d_pos - d_neg + self.config.margin).sum(dim=1).mean()
        grad_x = torch.autograd.grad(loss, x, retain_graph=True)[0]
        return grad_x


class DualGuidanceScheduler:
    """Phase scheduler for discriminative vs task guidance."""

    def __init__(self, t_switch: float, alpha: float, lambda_: float,
                 phase_cfg: Optional[PhaseScheduleConfig] = None):
        self.t_switch = t_switch
        self.alpha = alpha
        self.lambda_ = lambda_
        self.phase_cfg = phase_cfg

    def update_mean(self, mu: torch.Tensor, t: int, T: int,
                    grad_disc: Optional[torch.Tensor], grad_task: Optional[torch.Tensor],
                    x_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.phase_cfg is not None and self.phase_cfg.use_energy_ratio:
            if x_t is None:
                raise ValueError("x_t is required when using energy-ratio scheduling.")
            ratio = high_frequency_energy_ratio(
                x_t,
                cutoff=self.phase_cfg.energy_cutoff,
                eps=self.phase_cfg.energy_eps,
            )
            if ratio < self.phase_cfg.energy_ratio_threshold:
                return mu - self.lambda_ * grad_disc if grad_disc is not None else mu
            return mu + self.alpha * grad_task if grad_task is not None else mu

        if isinstance(self.t_switch, float):
            switch_step = int(self.t_switch * T)
        else:
            switch_step = int(self.t_switch)

        if t > switch_step:
            if grad_disc is None:
                return mu
            return mu - self.lambda_ * grad_disc
        if grad_task is None:
            return mu
        return mu + self.alpha * grad_task
