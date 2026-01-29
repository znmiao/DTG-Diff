"""DTG-Diff guidance primitives for CutAddPaste-based workflows."""

from .config import DiscriminativeGuidanceConfig, TaskGuidanceConfig, PhaseScheduleConfig
from .embedding import CutAddPasteEncoder, build_embedding_fn
from .failure_mining import FailureSet, mine_failure_sets
from .guidance import (
    GuidanceState,
    TaskGradientGuidance,
    DiscriminativeSpaceGuidance,
    DualGuidanceScheduler,
)
from .generator import DiffusionBackboneConfig, DTGDiff, AnomalyGenerator
from .diffusion import DiffusionSchedule, GaussianDiffusion1D
from .pipeline import blind_spot_detection, build_guidance_state, rectified_dataloader
from .dataset_processing import (
    WindowConfig,
    load_dataset,
    prepare_dtg_arrays,
    prepare_and_save,
    prepare_splits,
    compute_stats,
    plot_stats,
)
from .configs import DTGDatasetConfig, get_config as get_dataset_config, to_cutaddpaste_config
from .experiment_logging import ExperimentLogger
from .checkpoints import CheckpointManager
from .evaluation import evaluate_detector
from .visualization import plot_sample_compare, plot_embedding
from .metrics_utils import aggregate_metrics

__all__ = [
    "DiscriminativeGuidanceConfig",
    "TaskGuidanceConfig",
    "PhaseScheduleConfig",
    "CutAddPasteEncoder",
    "build_embedding_fn",
    "FailureSet",
    "mine_failure_sets",
    "GuidanceState",
    "TaskGradientGuidance",
    "DiscriminativeSpaceGuidance",
    "DualGuidanceScheduler",
    "DiffusionBackboneConfig",
    "DTGDiff",
    "AnomalyGenerator",
    "DiffusionSchedule",
    "GaussianDiffusion1D",
    "blind_spot_detection",
    "build_guidance_state",
    "rectified_dataloader",
    "WindowConfig",
    "load_dataset",
    "prepare_dtg_arrays",
    "prepare_and_save",
    "prepare_splits",
    "compute_stats",
    "plot_stats",
    "get_dataset_config",
    "DTGDatasetConfig",
    "to_cutaddpaste_config",
    "ExperimentLogger",
    "CheckpointManager",
    "evaluate_detector",
    "plot_sample_compare",
    "plot_embedding",
    "aggregate_metrics",
]
