from .base import DTGDatasetConfig


def get_config():
    return DTGDatasetConfig(
        name="synthetic",
        input_channels=3,
        window_size=32,
        time_step=16,
        kernel_size=5,
        stride=1,
        final_out_channels=16,
        project=2,
        dropout=0.1,
        features_len=6,
        num_epoch=2,
        lr=1e-3,
        weight=1e-4,
        batch_size=64,
        detect_nu=0.05,
        threshold_determine="floating",
        few_shot_count=8,
    )
