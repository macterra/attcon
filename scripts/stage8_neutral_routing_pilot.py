from __future__ import annotations

"""Test whether same-content causal routing emerges without a forced shared state."""

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

from attcon.integrated_content import (
    IntegratedContentConfig,
    generate_integrated_content_examples,
)
from attcon.integrated_content_experiment import (
    NeutralRoutingContentModel,
    evaluate_integrated_content_model,
    parameter_count,
    tensorize_integrated_content_examples,
    train_integrated_content_model,
    value_direction_intervention_metrics,
)


THRESHOLDS = {
    "learned_task_joint_accuracy": 0.90,
    "blocked_task_joint_accuracy": 0.90,
    "final_routing_weight": 0.20,
    "routing_weight_increase": 0.10,
    "learned_joint_directional_follow": 0.60,
    "joint_advantage_over_permuted_direction": 0.40,
    "joint_coordination_advantage_over_blocked": 0.40,
    "binding_other_fields_stability": 0.90,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=997)
    parser.add_argument("--model-seed", type=int, default=1009)
    parser.add_argument("--null-seed", type=int, default=1013)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--initial-routing-weight", type=float, default=0.05)
    parser.add_argument("--private-access-dropout", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", default="audits/stage8_neutral_routing_pilot.json"
    )
    args = parser.parse_args()

    config = IntegratedContentConfig()
    examples = generate_integrated_content_examples(
        args.episodes, config=config, seed=args.data_seed
    )
    train = tensorize_integrated_content_examples(
        [example for example in examples if example.split == "train"], config
    )
    heldout = tensorize_integrated_content_examples(
        [
            example for example in examples
            if example.split == "heldout_content_bundle"
        ],
        config,
    )

    runs = {}
    for routing in ("learned", "blocked"):
        torch.manual_seed(args.model_seed)
        model = NeutralRoutingContentModel(
            config,
            args.hidden_size,
            routing=routing,
            initial_routing_weight=args.initial_routing_weight,
            private_access_dropout=args.private_access_dropout,
        )
        initial_weight = float(torch.sigmoid(model.routing_logit).item())
        losses = train_integrated_content_model(
            model,
            train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.model_seed,
            device=args.device,
        )
        runs[routing] = {
            "model": model,
            "parameter_count": parameter_count(model),
            "initial_routing_weight": initial_weight,
            "final_routing_weight": float(model.routing_weight().item()),
            "learned_logit_sigmoid": float(torch.sigmoid(model.routing_logit).item()),
            "initial_epoch_loss": losses[0],
            "final_epoch_loss": losses[-1],
            "heldout": evaluate_integrated_content_model(
                model, heldout, device=args.device
            ),
            "true_direction": value_direction_intervention_metrics(
                model, train, heldout, alpha=args.alpha, device=args.device
            ),
            "permuted_label_direction": value_direction_intervention_metrics(
                model,
                train,
                heldout,
                alpha=args.alpha,
                permute_fit_labels=True,
                seed=args.null_seed,
                device=args.device,
            ),
        }

    learned = runs["learned"]
    blocked = runs["blocked"]
    joint_key = "accessible_binding_access_joint_donor_follow_rate"
    observed = {
        "learned_task_joint_accuracy": learned["heldout"][
            "binding_and_access_joint_accuracy"
        ],
        "blocked_task_joint_accuracy": blocked["heldout"][
            "binding_and_access_joint_accuracy"
        ],
        "final_routing_weight": learned["final_routing_weight"],
        "routing_weight_increase": learned["final_routing_weight"]
        - learned["initial_routing_weight"],
        "learned_joint_directional_follow": learned["true_direction"][joint_key],
        "joint_advantage_over_permuted_direction": learned["true_direction"][
            joint_key
        ]
        - learned["permuted_label_direction"][joint_key],
        "joint_coordination_advantage_over_blocked": learned["true_direction"][
            joint_key
        ]
        - blocked["true_direction"][joint_key],
        "binding_other_fields_stability": learned["true_direction"][
            "binding_other_fields_stability"
        ],
    }
    gates = {
        name: observed[name] >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    gates["parameters_exactly_matched"] = (
        learned["parameter_count"] == blocked["parameter_count"]
    )
    serialized = {
        name: {key: value for key, value in run.items() if key != "model"}
        for name, run in runs.items()
    }
    result = {
        "audit": (
            "stage8_task_induced_routing_pilot"
            if args.private_access_dropout
            else "stage8_neutral_routing_pilot"
        ),
        "status": (
            "task_induced_routing_supported"
            if args.private_access_dropout and all(gates.values())
            else "task_induced_routing_unsupported"
            if args.private_access_dropout
            else "neutral_routing_supported"
            if all(gates.values())
            else "neutral_routing_unsupported"
        ),
        "scope": (
            "single-seed dual-lane pilot; the cross-lane gate starts near closed and receives "
            + (
                "indirect task pressure from stochastic private-lane loss but no direct gate "
                "or overlap supervision"
                if args.private_access_dropout
                else "no direct overlap supervision or private-lane robustness pressure"
            )
        ),
        "config": {
            **asdict(config),
            "episode_count": args.episodes,
            "train_count": len(train),
            "heldout_count": len(heldout),
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "null_seed": args.null_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "initial_routing_weight": args.initial_routing_weight,
            "private_access_dropout": args.private_access_dropout,
            "alpha": args.alpha,
            "device": args.device,
        },
        "models": serialized,
        "observed": observed,
        "predeclared_thresholds": THRESHOLDS,
        "gates": gates,
        "all_neutral_routing_gates_pass": all(gates.values()),
        "all_routing_gates_pass": all(gates.values()),
        "stage8_same_content_gate_satisfied": False,
        "next_experiment": (
            "If routing emerges, repeat across seeds and architectures; if it does not, vary "
            "task pressure without directly supervising the gate or overlap."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
