"""Dataset-specific configs for DTG-Diff and detector training."""
from dataclasses import dataclass


@dataclass
class DTGDatasetConfig:
    # dataset
    name: str
    input_channels: int
    window_size: int
    time_step: int

    # detector model (CutAddPaste base model)
    kernel_size: int = 8
    stride: int = 1
    final_out_channels: int = 32
    project: int = 2
    dropout: float = 0.45
    features_len: int = 6

    # detector training
    num_epoch: int = 100
    beta1: float = 0.9
    beta2: float = 0.99
    lr: float = 3e-4
    weight: float = 5e-3
    drop_last: bool = False
    batch_size: int = 512

    # CutAddPaste augmentation
    trend_rate: float = 0.01
    rate: float = 1.0
    dim: int = 5
    cut_rate: int = 10

    # thresholding
    detect_nu: float = 0.001
    threshold_determine: str = "floating"

    # few-shot setup for generator
    few_shot_count: int = 50


class SimpleNamespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def to_cutaddpaste_config(cfg: DTGDatasetConfig):
    """Return an object compatible with CutAddPaste training code."""
    return SimpleNamespace(**cfg.__dict__)
