from __future__ import annotations

"""Train the structurally different temporal-relay replication pilot."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from attcon.temporal_relay import TemporalRelayConfig, generate_temporal_relay_examples
from attcon.temporal_relay_experiment import (
    TemporalRelayModel,
    RelationalTemporalRelayModel,
    evaluate_temporal_relay_model,
    parameter_count,
    payload_direction_metrics,
    tensorize_temporal_relay_examples,
    train_temporal_relay_model,
)


THRESHOLDS = {
    "shared_task_joint_accuracy": 0.85,
    "split_task_joint_accuracy": 0.85,
    "binding_advantage_over_order_destroyed": 0.50,
    "shared_binding_payload_follow": 0.75,
    "shared_access_payload_follow": 0.75,
    "shared_joint_payload_follow": 0.70,
    "binding_other_fields_stability": 0.85,
    "joint_advantage_over_permuted": 0.55,
    "coordination_advantage_over_split": 0.55,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("generic", "relational"), default="generic"
    )
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=1103)
    parser.add_argument("--model-seed", type=int, default=1123)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--out", default="audits/stage8_temporal_relay_pilot.json")
    args = parser.parse_args()
    config = TemporalRelayConfig()
    examples = generate_temporal_relay_examples(
        args.episodes, config=config, seed=args.data_seed
    )
    train = tensorize_temporal_relay_examples(
        [example for example in examples if example.split == "train"], config
    )
    heldout = tensorize_temporal_relay_examples(
        [example for example in examples if example.split == "heldout_event_bundle"], config
    )
    runs = {}
    model_class = (
        RelationalTemporalRelayModel
        if args.architecture == "relational"
        else TemporalRelayModel
    )
    for mode in ("shared", "split", "pooled"):
        torch.manual_seed(args.model_seed)
        model = model_class(config, 64, mode=mode)
        losses = train_temporal_relay_model(
            model, train, seed=args.model_seed, epochs=args.epochs
        )
        runs[mode] = {
            "model": model,
            "parameter_count": parameter_count(model),
            "initial_epoch_loss": losses[0],
            "final_epoch_loss": losses[-1],
            "heldout": evaluate_temporal_relay_model(model, heldout),
            "true_direction": payload_direction_metrics(model, train, heldout),
            "permuted_direction": payload_direction_metrics(
                model, train, heldout, permute_fit_labels=True
            ),
        }
    shared, split, pooled = runs["shared"], runs["split"], runs["pooled"]
    joint_key = "accessible_joint_donor_follow_rate"
    observed = {
        "shared_task_joint_accuracy": shared["heldout"]["binding_and_access_joint_accuracy"],
        "split_task_joint_accuracy": split["heldout"]["binding_and_access_joint_accuracy"],
        "binding_advantage_over_order_destroyed": shared["heldout"]["binding_joint_accuracy"]
        - pooled["heldout"]["binding_joint_accuracy"],
        "shared_binding_payload_follow": shared["true_direction"]["binding_payload_donor_follow_rate"],
        "shared_access_payload_follow": shared["true_direction"]["accessible_access_donor_follow_rate"],
        "shared_joint_payload_follow": shared["true_direction"][joint_key],
        "binding_other_fields_stability": shared["true_direction"]["binding_other_fields_stability"],
        "joint_advantage_over_permuted": shared["true_direction"][joint_key]
        - shared["permuted_direction"][joint_key],
        "coordination_advantage_over_split": shared["true_direction"][joint_key]
        - split["true_direction"][joint_key],
    }
    gates = {name: observed[name] >= threshold for name, threshold in THRESHOLDS.items()}
    gates["parameters_exactly_matched"] = len(
        {run["parameter_count"] for run in runs.values()}
    ) == 1
    result = {
        "audit": (
            "stage8_temporal_relay_relational_pilot"
            if args.architecture == "relational"
            else "stage8_temporal_relay_pilot"
        ),
        "status": "different_benchmark_engineering_support"
        if all(gates.values())
        else "different_benchmark_engineering_unsupported",
        "scope": (
            "single-seed engineered shared-state replication on a structurally different "
            "ordered event-stream task; Stage 8 support is not claimed"
        ),
        "config": {
            **asdict(config),
            "architecture": args.architecture,
            "episode_count": args.episodes,
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "epochs": args.epochs,
            "train_count": len(train),
            "heldout_count": len(heldout),
        },
        "models": {
            mode: {key: value for key, value in run.items() if key != "model"}
            for mode, run in runs.items()
        },
        "observed": observed,
        "predeclared_thresholds": THRESHOLDS,
        "gates": gates,
        "all_pilot_gates_pass": all(gates.values()),
        "different_benchmark_replication_established": False,
        "support_boundary": (
            "A passing single seed validates transfer of the engineered assay, not robust "
            "cross-benchmark convergence; fresh seeds and non-forced sharing remain required."
        ),
        "next_experiment": "repeat the frozen temporal-relay gates across fresh seeds",
    }
    output = ROOT / args.out
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
