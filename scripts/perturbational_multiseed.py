"""Multi-seed / multi-checkpoint robustness for the perturbational-complexity family.

The Stage 8 gate's non-reportability leg (`perturbational_complexity_metrics`) had only
single-seed "bounded" support. This audit upgrades that toward robust by sweeping the verdict
across many perturbation seeds (the stochastic noise + batch draw) and across every trained
checkpoint available, under one standardized perturbation config so the comparison is
apples-to-apples.

Two robustness questions, kept separate because they answer different things:

1. Perturbation-seed stability: on a fixed checkpoint, does `supported` hold across independent
   noise/batch draws, or is the single-seed verdict a lucky draw?
2. Cross-checkpoint replication: does it hold on independently trained controllers? The v3
   content-memory seeds are cross-training-seed and mildly cross-recipe replication -- NOT the
   structurally-different architecture the Stage 8 gate item (d) ultimately wants. State that
   boundary explicitly when reporting.

Usage:
    .venv/bin/python scripts/perturbational_multiseed.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from attcon.eval import (
    load_config,
    load_models_from_checkpoint,
    perturbational_complexity_metrics,
)

# (label, config, checkpoint). tune_prob_035 is the primary access/report checkpoint; the v3 seeds
# are the freshly-trained content-memory controllers (cross-training-seed replication).
CHECKPOINTS = [
    ("tune_prob_035", "configs/tune_prob_035.yaml", "outputs/tune_prob_035/experiment.pt"),
    ("v3_seed107", "configs/stage7_content_memory_v3_seed107.yaml", "outputs/stage7_content_memory_v3_seed107/experiment.pt"),
    ("v3_seed207", "configs/stage7_content_memory_v3_seed207.yaml", "outputs/stage7_content_memory_v3_seed207/experiment.pt"),
    ("v3_seed307", "configs/stage7_content_memory_v3_seed307.yaml", "outputs/stage7_content_memory_v3_seed307/experiment.pt"),
]

# Standardized perturbation config so every checkpoint is probed identically.
STD_PERTURBATIONAL = {
    "enabled": True,
    "probe_scenes": 16,
    "step": 2,
    "magnitudes": [0.5, 1.0, 2.0],
    "min_recovery_ratio": 0.1,
    "min_attention_propagation": 0.05,
}

RATIO_KEYS = (
    "recurrent_mean_recovery_ratio",
    "recurrent_mean_attention_propagation",
    "feedforward_mean_attention_propagation",
    "freeze_mean_recovery_ratio",
)
FLAG_KEYS = ("rich_but_recoverable", "integration_exceeds_feedforward", "recovery_exceeds_freeze", "supported")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perturbational-complexity multi-seed robustness.")
    parser.add_argument("--out", default="audits/perturbational_multiseed.json")
    parser.add_argument("--seeds", type=int, default=25, help="number of perturbation seeds per checkpoint")
    parser.add_argument("--seed-base", type=int, default=4000)
    parser.add_argument("--seed-stride", type=int, default=7)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="LABEL:CONFIG:CHECKPOINT",
        help="append an extra checkpoint (e.g. a different-architecture controller for cross-architecture replication); repeatable",
    )
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 6),
        "std": round(statistics.pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def audit_checkpoint(label: str, config: str, checkpoint: str, seeds: list[int]) -> dict[str, Any]:
    device = torch.device("cpu")
    cfg = load_config(config)
    _, task_cfg, models = load_models_from_checkpoint(checkpoint, device, cfg)
    cfg.setdefault("evaluation", {})["perturbational"] = dict(STD_PERTURBATIONAL)
    model = models["recurrent"]
    model.eval()

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        r = perturbational_complexity_metrics(model, cfg, task_cfg, device, seed=seed)
        per_seed.append({
            "seed": seed,
            **{k: bool(r[k]) for k in FLAG_KEYS},
            **{k: round(float(r[k]), 6) for k in RATIO_KEYS},
        })

    n = len(per_seed)
    flag_fractions = {k: round(sum(1 for s in per_seed if s[k]) / n, 4) for k in FLAG_KEYS}
    ratio_stats = {k: _mean_std([s[k] for s in per_seed]) for k in RATIO_KEYS}
    return {
        "label": label,
        "config": config,
        "checkpoint": checkpoint,
        "num_seeds": n,
        "supported_fraction": flag_fractions["supported"],
        "flag_fractions": flag_fractions,
        "ratio_stats": ratio_stats,
        "per_seed": per_seed,
    }


def main() -> None:
    args = parse_args()
    seeds = [args.seed_base + i * args.seed_stride for i in range(args.seeds)]

    checkpoints = list(CHECKPOINTS)
    for spec in args.extra:
        label, config, checkpoint = spec.split(":", 2)
        checkpoints.append((label, config, checkpoint))

    results = [audit_checkpoint(label, config, checkpoint, seeds) for label, config, checkpoint in checkpoints]

    # Robustness verdicts, kept honest about what they establish.
    per_seed_robust = all(r["supported_fraction"] == 1.0 for r in results)
    replicates = all(r["supported_fraction"] >= 0.9 for r in results)
    result = {
        "note": (
            "Perturbational-complexity robustness across perturbation seeds and checkpoints, under a "
            "standardized perturbation config. `supported` = rich-but-recoverable AND "
            "integration>feedforward AND recovery>freeze. `robust_across_perturbation_seeds` requires "
            "every checkpoint supported on 100% of seeds; `replicates_across_checkpoints` requires "
            ">=90% per checkpoint. CAVEAT: the v3 seeds are cross-training-seed / mild cross-recipe "
            "replication of the SAME recurrent controller architecture -- this does NOT satisfy the "
            "Stage 8 cross-architecture requirement (item d), which needs a structurally different "
            "controller."
        ),
        "standardized_perturbational": STD_PERTURBATIONAL,
        "num_perturbation_seeds": len(seeds),
        "robust_across_perturbation_seeds": per_seed_robust,
        "replicates_across_checkpoints": replicates,
        "checkpoints": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    for r in results:
        print(f"  {r['label']:14} supported {r['supported_fraction']*100:.0f}% of {r['num_seeds']} seeds "
              f"| rec_attn_prop {r['ratio_stats']['recurrent_mean_attention_propagation']['mean']:.3f}"
              f" vs ff {r['ratio_stats']['feedforward_mean_attention_propagation']['mean']:.3f}"
              f" | rec_recovery {r['ratio_stats']['recurrent_mean_recovery_ratio']['mean']:.3f}"
              f" vs freeze {r['ratio_stats']['freeze_mean_recovery_ratio']['mean']:.3f}")
    print(f"robust_across_perturbation_seeds={per_seed_robust}  replicates_across_checkpoints={replicates}")


if __name__ == "__main__":
    main()
