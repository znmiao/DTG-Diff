"""Gaussian diffusion utilities for 1D time series."""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn


@dataclass
class DiffusionSchedule:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GaussianDiffusion1D:
    def __init__(self, model: nn.Module, schedule: DiffusionSchedule, device: torch.device):
        self.model = model
        self.schedule = schedule
        self.device = device

        betas = torch.linspace(schedule.beta_start, schedule.beta_end, schedule.timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cum = alphas_cum
        self.sqrt_alphas_cum = torch.sqrt(alphas_cum)
        self.sqrt_one_minus_alphas_cum = torch.sqrt(1.0 - alphas_cum)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas = self.sqrt_alphas_cum[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cum[t].view(-1, 1, 1)
        return sqrt_alphas * x0 + sqrt_one_minus * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_alphas = self.sqrt_alphas_cum[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cum[t].view(-1, 1, 1)
        return (x_t - sqrt_one_minus * eps) / (sqrt_alphas + 1e-8)

    def p_mean(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        beta = self.betas[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        alpha_bar = self.alphas_cum[t].view(-1, 1, 1)
        return (1.0 / torch.sqrt(alpha)) * (x_t - beta / torch.sqrt(1 - alpha_bar) * eps)

    def p_sample(self,
                 x_t: torch.Tensor,
                 t: torch.Tensor,
                 x_few: torch.Tensor,
                 guidance_fn: Optional[Callable] = None) -> torch.Tensor:
        eps = self.model(x_t, t, x_few)
        mu = self.p_mean(x_t, t, eps)
        x0_pred = self.predict_x0(x_t, t, eps)

        if guidance_fn is not None:
            mu = guidance_fn(mu, x_t, x0_pred, t)

        if t.min().item() == 0:
            return mu

        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(self.betas[t]).view(-1, 1, 1)
        return mu + sigma * noise

    def sample(self,
               batch_size: int,
               shape: tuple,
               x_few: torch.Tensor,
               guidance_fn: Optional[Callable] = None) -> torch.Tensor:
        x_t = torch.randn((batch_size,) + shape, device=self.device)
        for step in reversed(range(self.schedule.timesteps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, x_few, guidance_fn=guidance_fn)
        return x_t
