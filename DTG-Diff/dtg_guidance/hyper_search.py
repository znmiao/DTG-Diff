"""Hyperparameter search for CAGA and tN scheduling."""
from __future__ import annotations

import argparse
import json
import os

from dtg_guidance.pipeline_full import build_run_args, run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--series_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/hyper_search.json")
    args = parser.parse_args()

    t_switch_vals = [0.1, 0.3, 0.5]
    results = []

    for t_switch in t_switch_vals:
        run_args = build_run_args(args.dataset, {
            "root": args.root,
            "series_index": args.series_index,
            "device": args.device,
            "seed": args.seed,
            "t_switch": t_switch,
        })
        metrics = run_pipeline(run_args)
        results.append({"mode": "fixed", "t_switch": t_switch, "metrics": metrics})

    run_args = build_run_args(args.dataset, {
        "root": args.root,
        "series_index": args.series_index,
        "device": args.device,
        "seed": args.seed,
        "use_energy_ratio": True,
    })
    metrics = run_pipeline(run_args)
    results.append({"mode": "energy", "t_switch": 0.3, "metrics": metrics})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved hyper search results to {args.out}")


if __name__ == "__main__":
    main()
