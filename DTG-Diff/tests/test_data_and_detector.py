import numpy as np
import torch

from dataloader.data_preprocessing import synthetic
from dtg_guidance.dataset_processing import WindowConfig, prepare_splits
from dtg_guidance.configs import get_config, to_cutaddpaste_config
from models.CutAddPaste.network.model import base_Model


def test_synthetic_dataset_produces_windowed_anomalies():
    cfg = WindowConfig(window_size=32, time_step=16, few_shot_count=4, seed=7)

    splits = prepare_splits("synthetic", cfg)

    assert splits["train_x"].ndim == 3
    assert splits["train_x"].shape[1:] == (3, 32)
    assert splits["test_y"].sum() > 0
    assert splits["train_mask"].shape[-1] == 32


def test_local_cutaddpaste_detector_matches_expected_interface():
    train_data, _, train_labels, _ = synthetic(seed=3)
    assert train_data.shape[1] == 3
    assert train_labels.shape[0] == train_data.shape[0]

    cfg = to_cutaddpaste_config(get_config("synthetic"))
    model = base_Model(cfg, torch.device("cpu"))
    x = torch.from_numpy(train_data[:8].T[None, :, :32]).float()

    logits = model(x)

    assert logits.shape == (1, 2)
    for name in ["conv_block1", "conv_block2", "conv_block3"]:
        assert hasattr(model, name)
