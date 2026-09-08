import torch
from torch import nn

from dtg_guidance.diffusion import DiffusionSchedule, GaussianDiffusion1D


class ZeroNoiseModel(nn.Module):
    def forward(self, x_t, t, x_few=None):
        return torch.zeros_like(x_t)


def test_posterior_variance_is_precomputed_and_zero_at_first_step():
    diffusion = GaussianDiffusion1D(
        ZeroNoiseModel(),
        DiffusionSchedule(timesteps=8, beta_start=0.01, beta_end=0.08),
        torch.device("cpu"),
    )

    assert hasattr(diffusion, "posterior_variance")
    assert diffusion.posterior_variance.shape == (8,)
    assert torch.isclose(diffusion.posterior_variance[0], torch.tensor(0.0))
    assert torch.all(diffusion.posterior_variance[1:] > 0)


def test_p_sample_uses_model_output_and_preserves_shape():
    torch.manual_seed(0)
    diffusion = GaussianDiffusion1D(
        ZeroNoiseModel(),
        DiffusionSchedule(timesteps=4, beta_start=0.01, beta_end=0.04),
        torch.device("cpu"),
    )
    x = torch.randn(2, 3, 16)
    t = torch.full((2,), 2, dtype=torch.long)

    out = diffusion.p_sample(x, t, x_few=torch.randn(1, 3, 16))

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
