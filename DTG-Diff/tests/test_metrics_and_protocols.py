import numpy as np
import torch

import dtg_guidance.protocols as protocols
from models.anomaly_predict import ad_predict


def test_point_adjusted_and_revised_point_adjusted_can_differ():
    targets = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.2, 0.8, 0.1, 0.2, 0.9, 0.1])

    _, rpa, pa, _, _ = ad_predict(targets, scores, threshold_mode="fixed", nu=0.5, threshold=0.5)

    assert pa.f1() > rpa.f1()


def test_protocol_generator_uses_passed_adaptation_lr(monkeypatch):
    used_lrs = []

    class Optim:
        def __init__(self, params, lr):
            used_lrs.append(lr)

        def zero_grad(self):
            pass

        def step(self):
            pass

    monkeypatch.setattr(torch.optim, "Adam", Optim)

    class Diffusion:
        schedule = type("Schedule", (), {"timesteps": 1})()

        def __init__(self, model, schedule, device):
            pass

        def q_sample(self, x, t, noise):
            return x

    class Generator(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(1.0))

        def set_base_training(self):
            pass

        def set_adaptation_training(self):
            pass

        def trainable_parameters(self):
            return [self.p]

        def base_loss(self, x_t, t, noise):
            return self.p * 0 + x_t.mean() * 0 + 1

        def adaptation_loss(self, x_t, t, noise, x_few, mask=None):
            return self.p * 0 + x_t.mean() * 0 + 1

        def orthogonality_loss(self):
            return self.p * 0

        def freeze_all(self):
            pass

    monkeypatch.setattr(protocols, "DTGDiff", Generator)
    monkeypatch.setattr(protocols, "GaussianDiffusion1D", Diffusion)

    x = np.zeros((1, 2, 8), dtype=np.float32)
    protocols.train_dtg_generator(
        x,
        x,
        None,
        in_channels=2,
        timesteps=1,
        device=torch.device("cpu"),
        base_epochs=1,
        adapt_epochs=1,
        base_batch=1,
        adapt_batch=1,
        base_lr=0.004,
        adapt_lr=0.009,
        num_prototypes=2,
        proto_dim=4,
        topk=1,
        ortho_weight=0.2,
    )

    assert used_lrs == [0.004, 0.009]
