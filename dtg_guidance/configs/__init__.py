"""Dataset-specific configs for DTG-Diff workflows."""
from .base import DTGDatasetConfig, to_cutaddpaste_config
from .swat_config import get_config as swat_config
from .wadi_config import get_config as wadi_config
from .psm_config import get_config as psm_config


def get_config(name: str) -> DTGDatasetConfig:
    key = name.upper()
    if key == "SWAT":
        return swat_config()
    if key == "WADI":
        return wadi_config()
    if key == "PSM":
        return psm_config()
    raise KeyError(f"No DTG config for dataset: {name}")


__all__ = [
    "DTGDatasetConfig",
    "to_cutaddpaste_config",
    "get_config",
    "swat_config",
    "wadi_config",
    "psm_config",
]
