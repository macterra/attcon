from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from attcon.access_experiment import (
    SetTransformerAccessModel,
    access_intervention_metrics,
    evaluate_access_model,
    parameter_count,
    tensorize_access_examples,
    train_access_model,
)
from attcon.counterfactual_access import (
    CounterfactualAccessConfig,
    generate_counterfactual_access_examples,
)
from scripts.branch_d_access_pilot import THRESHOLDS


VARIANTS = {
    "original": {
        "config": CounterfactualAccessConfig(),
        "data_seed": 811,
        "model_seed": 1201,
        "surface_schema": {
            "query": "object_key",
            "answer": "digit_value",
            "memory_event": "inspection_or_task_access",
        },
    },
    "surface_v2": {
        "config": CounterfactualAccessConfig(
            num_items=10,
            key_vocab_size=20,
            value_vocab_size=7,
            heldout_modulus=4,
        ),
        "data_seed": 919,
        "model_seed": 1301,
        "surface_schema": {
            "query": "shape_key",
            "answer": "texture_code",
            "memory_event": "touch_or_priority_access",
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--fusion-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_d_access_cross_model.json")
    args = parser.parse_args()

    runs = []
    for variant_name, variant in VARIANTS.items():
        print(f"running set-transformer variant={variant_name}", flush=True)
        config = variant["config"]
        examples = generate_counterfactual_access_examples(
            args.count, config=config, seed=variant["data_seed"]
        )
        train = tensorize_access_examples(
            [example for example in examples if example.split == "train"], config
        )
        heldout = tensorize_access_examples(
            [
                example
                for example in examples
                if example.split == "heldout_query_value"
            ],
            config,
        )
        torch.manual_seed(variant["model_seed"])
        integrated = SetTransformerAccessModel(
            config,
            args.hidden_size,
            args.fusion_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
        torch.manual_seed(variant["model_seed"])
        no_cache = SetTransformerAccessModel(
            config,
            args.hidden_size,
            args.fusion_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
        integrated_loss = train_access_model(
            integrated,
            train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=variant["model_seed"],
            device=args.device,
        )
        no_cache_loss = train_access_model(
            no_cache,
            train,
            erase_cache=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=variant["model_seed"],
            device=args.device,
        )
        integrated_eval = evaluate_access_model(
            integrated, heldout, device=args.device
        )
        no_cache_eval = evaluate_access_model(
            no_cache, heldout, erase_cache=True, device=args.device
        )
        integrated_eval.pop("predictions")
        no_cache_eval.pop("predictions")
        interventions = access_intervention_metrics(
            integrated, heldout, config, device=args.device
        )
        observed = {
            "internal_overall_accuracy": integrated_eval["accuracy"],
            "internal_previously_attended_accuracy": integrated_eval["by_status"][
                "previously_attended"
            ]["accuracy"],
            "internal_counterfactual_accuracy": integrated_eval["by_status"][
                "counterfactually_accessible"
            ]["accuracy"],
            "memory_and_tension_advantage_over_no_cache": integrated_eval[
                "memory_and_tension_accuracy"
            ]
            - no_cache_eval["memory_and_tension_accuracy"],
            "unavailable_accuracy": integrated_eval["by_status"]["unavailable"][
                "accuracy"
            ],
            "merely_visible_accuracy": integrated_eval["by_status"]["merely_visible"][
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
        run = {
            "variant": variant_name,
            "surface_schema": variant["surface_schema"],
            "benchmark_config": asdict(config),
            "data_seed": variant["data_seed"],
            "model_seed": variant["model_seed"],
            "train_count": len(train),
            "heldout_count": len(heldout),
            "integrated_parameter_count": parameter_count(integrated),
            "no_cache_parameter_count": parameter_count(no_cache),
            "parameters_exactly_matched": parameter_count(integrated)
            == parameter_count(no_cache),
            "integrated_final_loss": integrated_loss[-1],
            "no_cache_final_loss": no_cache_loss[-1],
            "integrated": integrated_eval,
            "no_cache": no_cache_eval,
            "interventions": interventions,
            "observed_gate_metrics": observed,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
        runs.append(run)
        print(
            f"  gates={run['all_gates_pass']} "
            f"accuracy={integrated_eval['accuracy']:.4f} "
            f"memory_advantage={observed['memory_and_tension_advantage_over_no_cache']:.4f}",
            flush=True,
        )

    schemas_differ = runs[0]["surface_schema"] != runs[1]["surface_schema"]
    configs_differ = runs[0]["benchmark_config"] != runs[1]["benchmark_config"]
    result = {
        "audit": "branch_d_access_cross_model",
        "architecture": (
            "permutation-equivariant event-set transformer with relational key addressing"
        ),
        "control": "identical transformer with all access events erased",
        "training_config": {
            "count_per_run": args.count,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "fusion_size": args.fusion_size,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "device": args.device,
        },
        "predeclared_thresholds": THRESHOLDS,
        "runs": runs,
        "summary": {
            "all_variants_pass": all(run["all_gates_pass"] for run in runs),
            "all_parameters_exactly_matched": all(
                run["parameters_exactly_matched"] for run in runs
            ),
            "surface_schemas_differ": schemas_differ,
            "benchmark_cardinalities_differ": configs_differ,
            "minimum_internal_accuracy": min(
                run["integrated"]["accuracy"] for run in runs
            ),
            "minimum_memory_and_tension_advantage": min(
                run["observed_gate_metrics"][
                    "memory_and_tension_advantage_over_no_cache"
                ]
                for run in runs
            ),
            "minimum_cache_erasure_drop": min(
                run["interventions"]["memory_target_cache_erasure_accuracy_drop"]
                for run in runs
            ),
            "minimum_observation_change_cache_retention": min(
                run["interventions"][
                    "counterfactual_cache_answer_retention_after_observation_change"
                ]
                for run in runs
            ),
        },
    }
    result["cross_model_surface_supported"] = all(result["summary"].values())
    result["support_boundary"] = (
        "Both architectures share explicit relational key addressing, and both benchmarks "
        "retain the synthetic query-access structure; this is strong bounded, not Stage 8-robust."
    )
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
