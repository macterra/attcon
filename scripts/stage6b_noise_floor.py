"""Run the calibrated Stage 6B uncertainty/allocation-error probe audit.

The audit compares controller-state probes with capacity-matched previous-observation
probes, then repeats those comparisons under independently permuted train and test
labels. Stage 6B support requires every gated signal to clear both the directional
capacity guard and its empirical accuracy/positive-recall noise floors.

Usage:
    .venv/bin/python scripts/stage6b_noise_floor.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attcon.eval import load_models_from_checkpoint, uncertainty_report_metrics
from attcon.train import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the calibrated Stage 6B audit.")
    parser.add_argument("--config", default="configs/tune_prob_035.yaml")
    parser.add_argument("--checkpoint", default="outputs/tune_prob_035/experiment.pt")
    parser.add_argument("--out", default="audits/stage6b_noise_floor_tune_prob_035.json")
    parser.add_argument("--permutations", type=int, default=12)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--seed-offset", type=int, default=9738)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    cfg = load_config(args.config)
    probe_cfg = cfg["evaluation"]["uncertainty_report_probes"]
    probe_cfg["noise_floor"] = {
        "enabled": True,
        "permutations": args.permutations,
        "percentile": args.percentile,
    }
    _, task_cfg, models = load_models_from_checkpoint(args.checkpoint, device, cfg)
    model = models["recurrent"]
    model.eval()
    metrics = uncertainty_report_metrics(
        model,
        cfg,
        task_cfg,
        device,
        int(cfg["seed"]) + args.seed_offset,
    )
    payload = {
        "config_base": args.config,
        "checkpoint": args.checkpoint,
        "permutations": args.permutations,
        "percentile": args.percentile,
        "method": (
            "controller state versus capacity-matched previous observation; independent "
            "train/test label permutations; identical matched-probe initialization"
        ),
        "result": metrics,
        "supported": metrics.get("supported", False),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
