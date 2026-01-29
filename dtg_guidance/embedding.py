"""Embedding utilities for discriminative-space guidance."""
from typing import Callable
import torch
from torch import nn


class CutAddPasteEncoder(nn.Module):
    """Extracts the penultimate embedding from the CutAddPaste base model.

    This wrapper reuses the model's convolutional blocks to produce a
    flattened feature embedding that matches the discriminative space.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        required = ["conv_block1", "conv_block2", "conv_block3"]
        for name in required:
            if not hasattr(model, name):
                raise ValueError(f"Model missing required attribute: {name}")
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv_block1(x)
        x = self.model.conv_block2(x)
        x = self.model.conv_block3(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), -1)
        return x


def build_embedding_fn(model: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns an embedding callable for a compatible detector model."""
    return CutAddPasteEncoder(model)
