from .base import DTGDatasetConfig


def get_config():
    # PSM dataset defaults (25 channels in common benchmarks).
    # Adjust window/time_step if you use a different preprocessing.
    return DTGDatasetConfig(
        name="PSM",
        input_channels=25,
        window_size=32,
        time_step=16,
        kernel_size=8,
        stride=1,
        final_out_channels=32,
        project=2,
        dropout=0.45,
        features_len=6,
        num_epoch=100,
        lr=3e-4,
        weight=5e-3,
        batch_size=512,
        trend_rate=0.01,
        rate=1.0,
        dim=5,
        cut_rate=10,
        detect_nu=0.001,
        threshold_determine="floating",
        few_shot_count=50,
    )
