import argparse

from dtg_guidance.pipeline_full import build_run_args


def test_build_run_args_contains_all_pipeline_fields():
    args = build_run_args(dataset="synthetic", overrides={"seed": 5})

    required = {
        "task_epsilon",
        "task_alpha",
        "disable_caga",
        "disable_per_sample_grads",
        "disc_lambda",
        "disc_margin",
        "disc_neighbors",
        "max_bank_size",
        "ortho_weight",
        "failure_batch_size",
        "guidance_mode",
    }

    assert isinstance(args, argparse.Namespace)
    assert required.issubset(vars(args))
    assert args.dataset == "synthetic"
    assert args.seed == 5
