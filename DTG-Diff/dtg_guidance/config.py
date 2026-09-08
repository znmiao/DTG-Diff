"""Configuration dataclasses for DTG-Diff guidance."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiscriminativeGuidanceConfig:
    """Hyperparameters for discriminative-space guidance."""
    margin: float = 0.5
    k_neighbors: int = 5
    lambda_: float = 1.0
    bank_batch_size: int = 256
    max_bank_size: Optional[int] = None


@dataclass
class TaskGuidanceConfig:
    """Hyperparameters for downstream task guidance."""
    epsilon: float = 1e-2
    alpha: float = 2.0
    use_caga: bool = True
    caga_use_updated: bool = False
    grad_eps: float = 1e-12
    per_sample_grads: bool = False
    preconditioner: str = "l2"


@dataclass
class PhaseScheduleConfig:
    """Switch point between discriminative and task guidance.

    If `t_switch` is a float in (0, 1], it is interpreted as a fraction of T.
    If it is an int, it is treated as an absolute step index.
    """
    t_switch: Optional[float] = 0.3
    use_energy_ratio: bool = False
    energy_ratio_threshold: float = 0.3
    energy_cutoff: float = 0.5
    energy_eps: float = 1e-12
