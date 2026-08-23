from __future__ import annotations

"""Cross-validated direction-limited causal overlap audit for integrated content."""

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
    IntegratedContentModel,
    evaluate_integrated_content_model,
    parameter_count,
    tensorize_integrated_content_examples,
    train_integrated_content_model,
    value_direction_intervention_metrics,
)


THRESHOLDS = {
    "shared_task_joint_accuracy": 0.90,
    "split_task_joint_accuracy": 0.90,
    "shared_binding_value_donor_follow": 0.80,
    "shared_access_value_donor_follow": 0.80,
    "shared_joint_value_donor_follow": 0.75,
    "binding_other_fields_stability": 0.90,
    "unavailable_unknown_retention": 0.90,
    "joint_advantage_over_permuted_direction": 0.60,
    "joint_coordination_advantage_over_split": 0.60,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=887)
    parser.add_argument("--model-seed", type=int, default=907)
    parser.add_argument("--null-seed", type=int, default=911)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", default="audits/stage8_integrated_content_directional.json"
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

    trained = {}
    for mode in ("shared", "split"):
        torch.manual_seed(args.model_seed)
        model = IntegratedContentModel(config, args.hidden_size, mode=mode)
        losses = train_integrated_content_model(
            model,
            train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.model_seed,
            device=args.device,
        )
        trained[mode] = {
            "model": model,
            "parameter_count": parameter_count(model),
            "initial_epoch_loss": losses[0],
            "final_epoch_loss": losses[-1],
            "heldout": evaluate_integrated_content_model(
                model, heldout, device=args.device
            ),
            "true_direction": value_direction_intervention_metrics(
                model,
                train,
                heldout,
                alpha=args.alpha,
                seed=args.null_seed,
                device=args.device,
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

    shared = trained["shared"]
    split = trained["split"]
    true = shared["true_direction"]
    permuted = shared["permuted_label_direction"]
    split_true = split["true_direction"]
    joint_key = "accessible_binding_access_joint_donor_follow_rate"
    observed = {
        "shared_task_joint_accuracy": shared["heldout"][
            "binding_and_access_joint_accuracy"
        ],
        "split_task_joint_accuracy": split["heldout"][
            "binding_and_access_joint_accuracy"
        ],
        "shared_binding_value_donor_follow": true[
            "binding_value_donor_follow_rate"
        ],
        "shared_access_value_donor_follow": true[
            "accessible_access_donor_follow_rate"
        ],
        "shared_joint_value_donor_follow": true[joint_key],
        "binding_other_fields_stability": true[
            "binding_other_fields_stability"
        ],
        "unavailable_unknown_retention": true[
            "unavailable_unknown_retention_rate"
        ],
        "joint_advantage_over_permuted_direction": true[joint_key]
        - permuted[joint_key],
        "joint_coordination_advantage_over_split": true[joint_key]
        - split_true[joint_key],
    }
    gates = {
        name: observed[name] >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    gates["parameters_exactly_matched"] = (
        shared["parameter_count"] == split["parameter_count"]
    )
    serialized = {
        mode: {key: value for key, value in result.items() if key != "model"}
        for mode, result in trained.items()
    }
    result = {
        "audit": "stage8_integrated_content_directional",
        "status": "directional_engineering_support"
        if all(gates.values())
        else "directional_engineering_unsupported",
        "scope": (
            "value-content directions are fitted only on training representations and "
            "intervened on held-out content bundles; the model's shared bottleneck remains "
            "architecturally imposed"
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
            "alpha": args.alpha,
            "device": args.device,
        },
        "models": serialized,
        "observed": observed,
        "predeclared_thresholds": THRESHOLDS,
        "gates": gates,
        "all_directional_gates_pass": all(gates.values()),
        "cross_validated_directional_overlap_observed": all(gates.values()),
        "stage8_same_content_gate_satisfied": False,
        "support_boundary": (
            "The intervention is disjoint-split and direction-limited, but it audits a "
            "supervised value direction inside an explicitly shared architecture. Independent "
            "theory-family emergence and a different benchmark are still absent."
        ),
        "next_experiment": (
            "repeat the directional audit across seeds, then remove the forced sharing and "
            "ask whether joint training produces the overlap relative to a matched split null"
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
