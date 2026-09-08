from .base import DTGDatasetConfig


def get_config():
    return DTGDatasetConfig(
        name="WADI",
        input_channels=127,
        window_size=32,
        time_step=16,
        kernel_size=4,
        stride=1,
        final_out_channels=32,
        project=2,
        dropout=0.45,
        features_len=6,
        num_epoch=50,
        lr=3e-4,
        weight=5e-3,
        batch_size=512,
        trend_rate=0.1,
        rate=1.0,
        dim=15,
        cut_rate=16,
        detect_nu=0.001,
        threshold_determine="floating",
        few_shot_count=50,
    )
