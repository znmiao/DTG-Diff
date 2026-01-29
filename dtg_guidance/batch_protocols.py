"""Batch runner for TSTR/TSRTR protocols (optionally multi-seed)."""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from dtg_guidance.pipeline_full import run_pipeline
from dtg_guidance.metrics_utils import aggregate_metrics


def parse_seeds(seed_str: str) -> List[int]:
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(list(range(int(a), int(b) + 1)))
        elif part:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated datasets")
    parser.add_argument("--seeds", type=str, default="0", help="Comma or range, e.g., 0,1,2 or 0-4")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/protocol_batch.json")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = parse_seeds(args.seeds)
    protocols = ["TSTR", "TSRTR"]

    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for protocol in protocols:
            run_summaries = []
            for seed in seeds:
                run_args = argparse.Namespace(
                    dataset=dataset,
                    root=args.root,
                    series_index=args.series_index,
                    device=args.device,
                    seed=seed,
                    run_dir=None,
                    protocol=protocol,
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
                    ablation="full",
                    t_switch=0.3,
                    use_energy_ratio=False,
                    energy_ratio_threshold=0.3,
                    energy_cutoff=0.5,
                )
                summary = run_pipeline(run_args)
                summary["seed"] = seed
                run_summaries.append(summary)
            results[dataset][protocol] = {
                "runs": run_summaries,
                "aggregate": aggregate_metrics(run_summaries),
            }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved batch protocol results to {args.out}")


if __name__ == "__main__":
    main()
