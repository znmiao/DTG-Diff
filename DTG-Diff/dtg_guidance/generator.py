"""DTG-Diff generator components."""
from dataclasses import dataclass
from typing import Optional

import math
import torch
from torch import nn


@dataclass
class DiffusionBackboneConfig:
    in_channels: int
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim: int = 128


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class FiLM(nn.Module):
    def __init__(self, emb_dim: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, channels)
        self.to_beta = nn.Linear(emb_dim, channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(emb).unsqueeze(-1)
        beta = self.to_beta(emb).unsqueeze(-1)
        return (1 + gamma) * x + beta


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, film_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film = FiLM(film_dim, out_ch)
        self.act = nn.SiLU()
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, z_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        t = self.time_mlp(t_emb).unsqueeze(-1)
        h = h + t
        h = self.film(h, z_emb)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.residual(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, cfg: DiffusionBackboneConfig, film_dim: int):
        super().__init__()
        self.cfg = cfg
        self.time_embed = TimeEmbedding(cfg.time_emb_dim)

        ch = cfg.base_channels
        self.init_conv = nn.Conv1d(cfg.in_channels, ch, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self._skip_channels = []
        for i, mult in enumerate(cfg.channel_mults):
            out_ch = cfg.base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(ResBlock1D(ch, out_ch, cfg.time_emb_dim, film_dim))
                ch = out_ch
                self._skip_channels.append(ch)
            self.down_blocks.append(blocks)
            if i < len(cfg.channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = ResBlock1D(ch, ch, cfg.time_emb_dim, film_dim)
        self.mid2 = ResBlock1D(ch, ch, cfg.time_emb_dim, film_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in enumerate(reversed(cfg.channel_mults)):
            out_ch = cfg.base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                skip_ch = self._skip_channels.pop()
                blocks.append(ResBlock1D(ch + skip_ch, out_ch, cfg.time_emb_dim, film_dim))
                ch = out_ch
            self.up_blocks.append(blocks)
            if i < len(cfg.channel_mults) - 1:
                self.upsamples.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv1d(ch, cfg.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.init_conv(x)
        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, t_emb, z)
                skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb, z)
        h = self.mid2(h, t_emb, z)

        for idx, blocks in enumerate(self.up_blocks):
            if idx > 0:
                h = self.upsamples[idx - 1](h)
            for block in blocks:
                skip = skips.pop()
                if h.size(-1) != skip.size(-1):
                    target = min(h.size(-1), skip.size(-1))
                    h = h[..., :target]
                    skip = skip[..., :target]
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb, z)

        h = self.out_norm(h)
        h = torch.nn.functional.silu(h)
        return self.out_conv(h)


class AnomalyPrototypePool(nn.Module):
    """Learnable prototype pool with sparse gating."""

    def __init__(self, num_prototypes: int, proto_dim: int, in_channels: int, topk: Optional[int] = None,
                 stat_dim: int = 5):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, proto_dim))
        self.stat_dim = stat_dim
        self.gate = nn.Sequential(
            nn.Linear(in_channels * stat_dim, proto_dim),
            nn.SiLU(),
            nn.LayerNorm(proto_dim),
            nn.Linear(proto_dim, num_prototypes),
        )
        self.topk = topk

    def _stats(self, x_few: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            mask = mask.to(dtype=x_few.dtype, device=x_few.device)
            denom = mask.sum(dim=-1).clamp_min(1.0)
            mean = (x_few * mask).sum(dim=-1) / denom
            centered = (x_few - mean.unsqueeze(-1)) * mask
            std = torch.sqrt((centered.pow(2).sum(dim=-1) / denom).clamp_min(1e-8))
            maximum = x_few.masked_fill(mask == 0, float("-inf")).amax(dim=-1)
            minimum = x_few.masked_fill(mask == 0, float("inf")).amin(dim=-1)
        else:
            mean = x_few.mean(dim=-1)
            std = x_few.std(dim=-1, unbiased=False)
            maximum = x_few.amax(dim=-1)
            minimum = x_few.amin(dim=-1)
        slope = x_few[..., -1] - x_few[..., 0]
        return torch.cat([mean, std, maximum, minimum, slope], dim=1)

    def forward(self, x_few: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self._stats(x_few, mask)
        weights = torch.softmax(self.gate(features), dim=-1)
        if self.topk is not None and self.topk < weights.size(1):
            topk_vals, topk_idx = torch.topk(weights, k=self.topk, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_idx, topk_vals)
            weights = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)
        z_sem = weights @ self.prototypes
        return z_sem

    def orthogonality_loss(self) -> torch.Tensor:
        """Encourage orthogonal prototypes."""
        p = torch.nn.functional.normalize(self.prototypes, dim=1)
        gram = p @ p.t()
        eye = torch.eye(gram.size(0), device=gram.device)
        return (gram - eye).pow(2).mean()


class DTGDiff(nn.Module):
    """DTG-Diff generator: frozen backbone + prototype pool + FiLM adapters."""

    def __init__(self, backbone_cfg: DiffusionBackboneConfig, num_prototypes: int,
                 proto_dim: int, topk: Optional[int] = None):
        super().__init__()
        self.prototype_pool = AnomalyPrototypePool(num_prototypes, proto_dim, backbone_cfg.in_channels, topk=topk)
        self.backbone = UNet1D(backbone_cfg, film_dim=proto_dim)

    def set_base_training(self):
        """Enable full training (normal data pretraining stage)."""
        for p in self.parameters():
            p.requires_grad_(True)

    def set_adaptation_training(self):
        """Freeze backbone except FiLM + prototype pool (few-shot adaptation stage)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for module in self.backbone.modules():
            if isinstance(module, FiLM):
                for p in module.parameters():
                    p.requires_grad_(True)
        for p in self.prototype_pool.parameters():
            p.requires_grad_(True)

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self):
        params = list(self.prototype_pool.parameters())
        for module in self.backbone.modules():
            if isinstance(module, FiLM):
                params.extend(list(module.parameters()))
        return params

    def _align_condition_batch(self, z_sem: torch.Tensor, batch_size: int) -> torch.Tensor:
        if z_sem.size(0) == batch_size:
            return z_sem
        repeats = math.ceil(batch_size / z_sem.size(0))
        return z_sem.repeat(repeats, 1)[:batch_size]

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, x_few: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_few is None:
            z_sem = torch.zeros(x_noisy.size(0), self.prototype_pool.prototypes.size(1), device=x_noisy.device)
        else:
            z_sem = self.prototype_pool(x_few, mask=mask)
            z_sem = self._align_condition_batch(z_sem, x_noisy.size(0))
        return self.backbone(x_noisy, t, z_sem)

    def base_loss(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """L_base = ||eps - eps_theta(x_t)||^2 (Eq.1)."""
        eps_pred = self.forward(x_t, t, x_few=None)
        return torch.nn.functional.mse_loss(eps_pred, noise)

    def adaptation_loss(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                        x_few: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """L_adapt = || M ⊙ (eps - eps_theta(x_t, x_few)) ||^2 (Eq.2)."""
        eps_pred = self.forward(x_t, t, x_few=x_few, mask=mask)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            diff = (noise - eps_pred) * mask
            return (diff.pow(2)).mean()
        return torch.nn.functional.mse_loss(eps_pred, noise)

    def orthogonality_loss(self) -> torch.Tensor:
        return self.prototype_pool.orthogonality_loss()


AnomalyGenerator = DTGDiff
