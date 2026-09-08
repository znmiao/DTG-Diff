"""Run pipeline with multiple seeds and report mean/std."""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from dtg_guidance.pipeline_full import build_run_args, run_pipeline
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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--protocol", type=str, default="TSRTR", choices=["TSTR", "TSRTR"])
    parser.add_argument("--out", type=str, default="outputs/multi_seed.json")
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    results = []
    for seed in seeds:
        run_args = build_run_args(args.dataset, {
            "root": args.root,
            "series_index": args.series_index,
            "device": args.device,
            "seed": seed,
            "run_dir": None if args.run_dir is None else os.path.join(args.run_dir, f"seed_{seed}"),
            "protocol": args.protocol,
        })
        summary = run_pipeline(run_args)
        summary["seed"] = seed
        results.append(summary)

    agg = aggregate_metrics(results)
    payload = {"dataset": args.dataset, "protocol": args.protocol, "seeds": seeds, "runs": results, "aggregate": agg}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved multi-seed results to {args.out}")


if __name__ == "__main__":
    main()
