import torch

from dtg_guidance.generator import AnomalyPrototypePool, DiffusionBackboneConfig, DTGDiff


def test_prototype_pool_uses_masked_temporal_content():
    pool = AnomalyPrototypePool(num_prototypes=4, proto_dim=8, in_channels=2, topk=2)
    x = torch.zeros(2, 2, 12)
    x[:, :, 4:8] = 3.0
    mask = torch.zeros(2, 12)
    mask[:, 4:8] = 1.0

    z_masked = pool(x, mask=mask)
    z_unmasked = pool(x)

    assert z_masked.shape == (2, 8)
    assert not torch.allclose(z_masked, z_unmasked)


def test_dtgdiff_conditions_generation_batch_without_global_mean_broadcast():
    cfg = DiffusionBackboneConfig(in_channels=2, base_channels=8, channel_mults=(1,), num_res_blocks=1, time_emb_dim=16)
    model = DTGDiff(cfg, num_prototypes=4, proto_dim=8, topk=2)
    x_noisy = torch.randn(5, 2, 16)
    x_few = torch.randn(2, 2, 16)
    t = torch.zeros(5, dtype=torch.long)

    out = model(x_noisy, t, x_few=x_few)

    assert out.shape == x_noisy.shape
