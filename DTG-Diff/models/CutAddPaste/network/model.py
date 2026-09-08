from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class base_Model(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        in_channels = int(getattr(cfg, "input_channels", getattr(cfg, "channels", 1)))
        hidden = int(getattr(cfg, "final_out_channels", 32))
        kernel_size = int(getattr(cfg, "kernel_size", 5))
        stride = int(getattr(cfg, "stride", 1))
        dropout = float(getattr(cfg, "dropout", 0.1))

        self.conv_block1 = ConvBlock(in_channels, hidden, kernel_size, stride, dropout)
        self.conv_block2 = TemporalResidualBlock(hidden, kernel_size, dropout)
        self.conv_block3 = TemporalResidualBlock(hidden, kernel_size, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        projection_dim = int(getattr(cfg, "projection_dim", max(16, hidden)))
        self.projection_head = nn.Sequential(
            nn.Linear(hidden, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(projection_dim, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.pool(x).squeeze(-1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.encode(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.project(x))
