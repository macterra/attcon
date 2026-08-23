from __future__ import annotations

"""Train and causally audit the Stage 8 paired same-content pilot."""

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
    binding_state_swap_metrics,
    evaluate_integrated_content_model,
    parameter_count,
    tensorize_integrated_content_examples,
    train_integrated_content_model,
)


THRESHOLDS = {
    "shared_binding_joint_accuracy": 0.90,
    "shared_access_accuracy": 0.90,
    "shared_binding_and_access_joint_accuracy": 0.85,
    "split_binding_and_access_joint_accuracy": 0.85,
    "binding_advantage_over_pooled": 0.50,
    "shared_binding_donor_follow_rate": 0.90,
    "shared_access_donor_follow_rate": 0.85,
    "shared_joint_donor_follow_rate": 0.85,
    "coordination_advantage_over_split": 0.65,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=811)
    parser.add_argument("--model-seed", type=int, default=827)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", default="audits/stage8_integrated_content_pilot.json"
    )
    args = parser.parse_args()

    config = IntegratedContentConfig()
    examples = generate_integrated_content_examples(
        args.episodes, config=config, seed=args.data_seed
    )
    train_examples = [example for example in examples if example.split == "train"]
    heldout_examples = [
        example for example in examples
        if example.split == "heldout_content_bundle"
    ]
    train = tensorize_integrated_content_examples(train_examples, config)
    heldout = tensorize_integrated_content_examples(heldout_examples, config)

    models = {}
    for mode in ("shared", "split", "pooled"):
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
        models[mode] = {
            "model": model,
            "initial_epoch_loss": losses[0],
            "final_epoch_loss": losses[-1],
            "heldout": evaluate_integrated_content_model(
                model, heldout, device=args.device
            ),
            "binding_state_swap": binding_state_swap_metrics(
                model, heldout, device=args.device
            ),
        }

    counts = {mode: parameter_count(item["model"]) for mode, item in models.items()}
    shared_eval = models["shared"]["heldout"]
    split_eval = models["split"]["heldout"]
    pooled_eval = models["pooled"]["heldout"]
    shared_swap = models["shared"]["binding_state_swap"]
    split_swap = models["split"]["binding_state_swap"]
    observed = {
        "shared_binding_joint_accuracy": shared_eval["binding_joint_accuracy"],
        "shared_access_accuracy": shared_eval["access_accuracy"],
        "shared_binding_and_access_joint_accuracy": shared_eval[
            "binding_and_access_joint_accuracy"
        ],
        "split_binding_and_access_joint_accuracy": split_eval[
            "binding_and_access_joint_accuracy"
        ],
        "binding_advantage_over_pooled": shared_eval["binding_joint_accuracy"]
        - pooled_eval["binding_joint_accuracy"],
        "shared_binding_donor_follow_rate": shared_swap[
            "binding_donor_follow_rate"
        ],
        "shared_access_donor_follow_rate": shared_swap[
            "accessible_access_donor_follow_rate"
        ],
        "shared_joint_donor_follow_rate": shared_swap[
            "accessible_joint_donor_follow_rate"
        ],
        "coordination_advantage_over_split": shared_swap[
            "accessible_joint_donor_follow_rate"
        ]
        - split_swap["accessible_joint_donor_follow_rate"],
    }
    gates = {
        name: observed[name] >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    gates["parameters_exactly_matched"] = len(set(counts.values())) == 1

    serialized_models = {}
    for mode, item in models.items():
        serialized_models[mode] = {
            key: value for key, value in item.items() if key != "model"
        }
        serialized_models[mode]["parameter_count"] = counts[mode]
    result = {
        "audit": "stage8_integrated_content_pilot",
        "status": "engineering_pilot_supported"
        if all(gates.values())
        else "engineering_pilot_unsupported",
        "scope": (
            "single-seed engineered shared bottleneck on one synthetic benchmark; this "
            "validates the same-content causal-overlap measurement but does not establish "
            "spontaneous multi-theory convergence or Stage 8 support"
        ),
        "config": {
            **asdict(config),
            "episode_count": args.episodes,
            "train_count": len(train),
            "heldout_count": len(heldout),
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "device": args.device,
        },
        "models": serialized_models,
        "observed": observed,
        "predeclared_thresholds": THRESHOLDS,
        "gates": gates,
        "all_engineering_pilot_gates_pass": all(gates.values()),
        "same_content_causal_overlap_observed": all(gates.values()),
        "stage8_same_content_gate_satisfied": False,
        "stage8_support_blockers": [
            "shared content bottleneck is imposed by architecture",
            "single data/model seed",
            "no cross-validated learned content direction",
            "no structurally different benchmark replication",
        ],
        "next_experiment": (
            "repeat over fresh seeds, learn the intervention direction on disjoint training "
            "data, and test it on held-out content bundles before changing task structure"
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
