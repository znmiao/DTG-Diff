"""Run ablations: no-disc, no-task, fn-only, fp-only."""
from __future__ import annotations

import argparse
import json
import os

from dtg_guidance.pipeline_full import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/ablation.json")
    args = parser.parse_args()

    variants = ["full", "no-disc", "no-task", "fn-only", "fp-only"]
    results = {}
    for variant in variants:
        run_args = argparse.Namespace(
            dataset=args.dataset,
            root=args.root,
            series_index=args.series_index,
            device=args.device,
            seed=args.seed,
            stats_plot=False,
            save_visuals=False,
            timesteps=100,
            base_epochs=5,
            adapt_epochs=3,
            synth_ratio=1.0,
            num_prototypes=10,
            proto_dim=128,
            topk=3,
            base_lr=1e-4,
            adapt_lr=5e-5,
            base_batch=64,
            adapt_batch=32,
            run_dir=None,
            protocol="TSRTR",
            log_dir=f"outputs/logs/{variant}",
            ckpt_dir=f"outputs/checkpoints/{variant}",
            ablation=variant,
            t_switch=0.3,
            use_energy_ratio=False,
            energy_ratio_threshold=0.3,
            energy_cutoff=0.5,
        )
        metrics = run_pipeline(run_args)
        results[variant] = metrics

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablation results to {args.out}")


if __name__ == "__main__":
    main()
