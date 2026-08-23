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

from attcon.access_experiment import (
    RelationalRecurrentAccessModel,
    RecurrentAccessModel,
    access_intervention_metrics,
    evaluate_access_model,
    parameter_count,
    tensorize_access_examples,
    train_access_model,
)
from attcon.counterfactual_access import (
    CounterfactualAccessConfig,
    TARGET_STATUSES,
    generate_counterfactual_access_examples,
)


THRESHOLDS = {
    "internal_overall_accuracy": 0.90,
    "internal_previously_attended_accuracy": 0.85,
    "internal_counterfactual_accuracy": 0.85,
    "memory_and_tension_advantage_over_no_cache": 0.50,
    "unavailable_accuracy": 0.90,
    "merely_visible_accuracy": 0.90,
    "cache_erasure_accuracy_drop": 0.50,
    "observation_change_cache_retention": 0.85,
}


def _baseline_metrics(tensors, config: CounterfactualAccessConfig) -> dict:
    result = {}
    for name, predictions in (
        ("scene_only", tensors.scene_answers),
        ("current_glimpse", tensors.glimpse_answers),
    ):
        correct = predictions.eq(tensors.targets)
        by_status = {}
        for index, status in enumerate(TARGET_STATUSES):
            mask = tensors.statuses.eq(index)
            by_status[status] = correct[mask].float().mean().item()
        result[name] = {
            "accuracy": correct.float().mean().item(),
            "by_status_accuracy": by_status,
        }
    result["symbolic_upper_bound"] = {"accuracy": 1.0}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("unstructured_gru", "relational_gru"),
        default="unstructured_gru",
    )
    parser.add_argument("--count", type=int, default=16384)
    parser.add_argument("--data-seed", type=int, default=11)
    parser.add_argument("--model-seed", type=int, default=71)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--fusion-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_d_access_pilot.json")
    args = parser.parse_args()

    config = CounterfactualAccessConfig()
    examples = generate_counterfactual_access_examples(
        args.count, config=config, seed=args.data_seed
    )
    train_examples = [example for example in examples if example.split == "train"]
    heldout_examples = [
        example for example in examples if example.split == "heldout_query_value"
    ]
    train = tensorize_access_examples(train_examples, config)
    heldout = tensorize_access_examples(heldout_examples, config)
    train_keys = {example.switched_query_key for example in train_examples}
    train_values = {example.expected_answer for example in train_examples}
    train_pairs = {
        (example.switched_query_key, example.expected_answer)
        for example in train_examples
    }
    model_class = (
        RelationalRecurrentAccessModel
        if args.architecture == "relational_gru"
        else RecurrentAccessModel
    )
    torch.manual_seed(args.model_seed)
    internal = model_class(config, args.hidden_size, args.fusion_size)
    torch.manual_seed(args.model_seed)
    no_cache = model_class(config, args.hidden_size, args.fusion_size)
    internal_losses = train_access_model(
        internal,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.model_seed,
        device=args.device,
    )
    no_cache_losses = train_access_model(
        no_cache,
        train,
        erase_cache=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.model_seed,
        device=args.device,
    )
    internal_eval = evaluate_access_model(internal, heldout, device=args.device)
    no_cache_eval = evaluate_access_model(
        no_cache, heldout, erase_cache=True, device=args.device
    )
    internal_train_eval = evaluate_access_model(internal, train, device=args.device)
    no_cache_train_eval = evaluate_access_model(
        no_cache, train, erase_cache=True, device=args.device
    )
    internal_eval.pop("predictions")
    no_cache_eval.pop("predictions")
    internal_train_eval.pop("predictions")
    no_cache_train_eval.pop("predictions")
    interventions = access_intervention_metrics(
        internal, heldout, config, device=args.device
    )
    observed = {
        "internal_overall_accuracy": internal_eval["accuracy"],
        "internal_previously_attended_accuracy": internal_eval["by_status"][
            "previously_attended"
        ]["accuracy"],
        "internal_counterfactual_accuracy": internal_eval["by_status"][
            "counterfactually_accessible"
        ]["accuracy"],
        "memory_and_tension_advantage_over_no_cache": internal_eval[
            "memory_and_tension_accuracy"
        ]
        - no_cache_eval["memory_and_tension_accuracy"],
        "unavailable_accuracy": internal_eval["by_status"]["unavailable"][
            "accuracy"
        ],
        "merely_visible_accuracy": internal_eval["by_status"]["merely_visible"][
            "accuracy"
        ],
        "cache_erasure_accuracy_drop": interventions[
            "memory_target_cache_erasure_accuracy_drop"
        ],
        "observation_change_cache_retention": interventions[
            "counterfactual_cache_answer_retention_after_observation_change"
        ],
    }
    gates = {
        name: observed[name] >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    result = {
        "audit": "branch_d_access_pilot",
        "status": "supported_pilot" if all(gates.values()) else "unsupported_pilot",
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
            "fusion_size": args.fusion_size,
            "device": args.device,
            "architecture": args.architecture,
        },
        "split_coverage": {
            "all_heldout_keys_seen_individually_in_train": all(
                example.switched_query_key in train_keys for example in heldout_examples
            ),
            "all_heldout_answers_seen_individually_in_train": all(
                example.expected_answer in train_values for example in heldout_examples
            ),
            "heldout_query_answer_pairs_seen_in_train": sum(
                (example.switched_query_key, example.expected_answer)
                in train_pairs
                for example in heldout_examples
            ),
        },
        "internal_access": {
            "architecture": (
                "query-key-addressed recurrent value states plus current scene"
                if args.architecture == "relational_gru"
                else "GRU-compressed access events plus current scene and switched query"
            ),
            "reads_explicit_cache_at_report_time": False,
            "parameter_count": parameter_count(internal),
            "initial_epoch_loss": internal_losses[0],
            "final_epoch_loss": internal_losses[-1],
            "train": internal_train_eval,
            "heldout": internal_eval,
        },
        "matched_no_cache": {
            "architecture": "identical GRU with all access events erased",
            "parameter_count": parameter_count(no_cache),
            "parameters_exactly_matched": parameter_count(internal)
            == parameter_count(no_cache),
            "initial_epoch_loss": no_cache_losses[0],
            "final_epoch_loss": no_cache_losses[-1],
            "train": no_cache_train_eval,
            "heldout": no_cache_eval,
        },
        "deterministic_controls": _baseline_metrics(heldout, config),
        "interventions": interventions,
        "observed_gate_metrics": observed,
        "predeclared_thresholds": THRESHOLDS,
        "gates": gates,
        "all_pilot_gates_pass": all(gates.values()),
        "branch_d_supported": all(gates.values()),
        "support_level": "bounded" if all(gates.values()) else "unsupported",
        "support_boundary": (
            "A passing run is bounded evidence for an engineered recurrent access cache. "
            "Branch D still requires multi-seed and alternate-architecture replication, and "
            "must not be described as spontaneous access in the original attention controller."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
