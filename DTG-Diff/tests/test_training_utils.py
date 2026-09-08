import torch
from torch import nn

from dtg_guidance.checkpoints import CheckpointManager


def test_checkpoint_manager_can_load_best_weights(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    model = nn.Linear(2, 1)

    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.25)
    best_metric, path = manager.save_best("detector_best", model, 0.7, None)

    with torch.no_grad():
        model.weight.fill_(9.0)
        model.bias.fill_(9.0)
    manager.load_model(path, model)

    assert best_metric == 0.7
    assert torch.allclose(model.weight, torch.ones_like(model.weight))
    assert torch.allclose(model.bias, torch.full_like(model.bias, 0.25))
