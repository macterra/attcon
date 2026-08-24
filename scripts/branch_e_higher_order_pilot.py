from __future__ import annotations

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

from attcon.higher_order import HigherOrderConfig, generate_higher_order_examples
from attcon.higher_order_experiment import (
    HigherOrderBehaviorModel,
    behavior_metrics,
    fixed_capacity_lift,
    paired_wrong_access_intervention,
    tensorize_higher_order_examples,
    train_higher_order_behavior_model,
    train_status_probe,
)


THRESHOLDS = {
    "minimum_behavior_accuracy": 0.90,
    "latent_status_accuracy": 0.85,
    "latent_advantage_over_first_order": 0.45,
    "latent_advantage_over_observation": 0.35,
    "confidence_increase_rate": 0.85,
    "reinspection_turns_off_rate": 0.85,
    "correction_turns_off_rate": 0.85,
    "newly_accessible_report_content_accuracy": 0.85,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=16380)
    parser.add_argument("--data-seed", type=int, default=13)
    parser.add_argument("--model-seed", type=int, default=83)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_e_higher_order_pilot.json")
    args = parser.parse_args()

    config = HigherOrderConfig()
    examples = generate_higher_order_examples(
        args.count, config=config, seed=args.data_seed
    )
    train_examples = [example for example in examples if example.split == "train"]
    heldout_examples = [
        example for example in examples if example.split == "heldout_content_status"
    ]
    train = tensorize_higher_order_examples(train_examples, config)
    heldout = tensorize_higher_order_examples(heldout_examples, config)
    torch.manual_seed(args.model_seed)
    model = HigherOrderBehaviorModel(config, args.hidden_size)
    losses = train_higher_order_behavior_model(
        model,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.model_seed,
        device=args.device,
    )
    train_behavior = behavior_metrics(model, train, device=args.device)
    heldout_behavior = behavior_metrics(model, heldout, device=args.device)
    model.eval()
    with torch.no_grad():
        train_latent = model(train.model_input.to(args.device)).hidden.cpu()
        heldout_latent = model(heldout.model_input.to(args.device)).hidden.cpu()
    probe_inputs = {
        "latent": (train_latent, heldout_latent),
        "first_order": (
            fixed_capacity_lift(
                train.first_order_features, args.hidden_size, seed=101
            ),
            fixed_capacity_lift(
                heldout.first_order_features, args.hidden_size, seed=101
            ),
        ),
        "observation_only": (
            fixed_capacity_lift(
                train.observation_features, args.hidden_size, seed=103
            ),
            fixed_capacity_lift(
                heldout.observation_features, args.hidden_size, seed=103
            ),
        ),
    }
    probes = {
        name: train_status_probe(
            train_features,
            train.status_target,
            heldout_features,
            heldout.status_target,
            steps=args.probe_steps,
            seed=97,
        )
        for name, (train_features, heldout_features) in probe_inputs.items()
    }
    interventions = paired_wrong_access_intervention(
        model, examples, config, device=args.device
    )
    observed = {
        "minimum_behavior_accuracy": min(heldout_behavior.values()),
        "latent_status_accuracy": probes["latent"]["heldout_accuracy"],
        "latent_advantage_over_first_order": probes["latent"]["heldout_accuracy"]
        - probes["first_order"]["heldout_accuracy"],
        "latent_advantage_over_observation": probes["latent"]["heldout_accuracy"]
        - probes["observation_only"]["heldout_accuracy"],
        "confidence_increase_rate": interventions["confidence_increase_rate"],
        "reinspection_turns_off_rate": interventions[
            "reinspection_turns_off_rate"
        ],
        "correction_turns_off_rate": interventions["correction_turns_off_rate"],
        "newly_accessible_report_content_accuracy": interventions[
            "newly_accessible_report_content_accuracy"
        ],
    }
    gates = {
        name: observed[name] >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    result = {
        "audit": "branch_e_higher_order_pilot",
        "status": "engineering_support" if all(gates.values()) else "unsupported_pilot",
        "config": {
            **asdict(config),
            "count": args.count,
            "train_count": len(train),
            "heldout_count": len(heldout),
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "probe_steps": args.probe_steps,
            "device": args.device,
        },
        "supervision": {
            "exact_six_way_status_labels_used_for_behavior_model": False,
            "behavior_targets": [
                "report answer",
                "confidence band",
                "reinspection decision",
                "correction decision",
            ],
            "status_labels_used_only_by_post_hoc_frozen_latent_probes": True,
            "engineering_support_only": True,
        },
        "training": {
            "initial_epoch_loss": losses[0],
            "final_epoch_loss": losses[-1],
            "train_behavior": train_behavior,
            "heldout_behavior": heldout_behavior,
        },
        "capacity_matched_status_probes": probes,
        "paired_wrong_access_intervention": interventions,
        "observed_gate_metrics": observed,
        "predeclared_thresholds": THRESHOLDS,
        "gates": gates,
        "all_engineering_gates_pass": all(gates.values()),
        "branch_e_supported": False,
        "support_boundary": (
            "The exact six-way labels are withheld from representation learning, but confidence, "
            "reinspection, and correction targets directly engineer access-sensitive behavior. "
            "A pass is engineering support only; spontaneous higher-order representation must "
            "emerge under objectives that do not reward these higher-order targets."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
