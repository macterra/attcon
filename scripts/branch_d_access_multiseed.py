from __future__ import annotations

import argparse
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
    RelationalRecurrentAccessModel,
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


SEED_PAIRS = ((103, 409), (211, 521), (307, 631))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--fusion-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_d_access_multiseed.json")
    args = parser.parse_args()

    config = CounterfactualAccessConfig()
    runs = []
    for data_seed, model_seed in SEED_PAIRS:
        print(
            f"running data_seed={data_seed} model_seed={model_seed}", flush=True
        )
        examples = generate_counterfactual_access_examples(
            args.count, config=config, seed=data_seed
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
        torch.manual_seed(model_seed)
        internal = RelationalRecurrentAccessModel(
            config, args.hidden_size, args.fusion_size
        )
        torch.manual_seed(model_seed)
        no_cache = RelationalRecurrentAccessModel(
            config, args.hidden_size, args.fusion_size
        )
        internal_loss = train_access_model(
            internal,
            train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=model_seed,
            device=args.device,
        )
        no_cache_loss = train_access_model(
            no_cache,
            train,
            erase_cache=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=model_seed,
            device=args.device,
        )
        internal_eval = evaluate_access_model(internal, heldout, device=args.device)
        no_cache_eval = evaluate_access_model(
            no_cache, heldout, erase_cache=True, device=args.device
        )
        internal_eval.pop("predictions")
        no_cache_eval.pop("predictions")
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
        run = {
            "data_seed": data_seed,
            "model_seed": model_seed,
            "train_count": len(train),
            "heldout_count": len(heldout),
            "parameter_count": parameter_count(internal),
            "parameters_exactly_matched": parameter_count(internal)
            == parameter_count(no_cache),
            "internal_final_loss": internal_loss[-1],
            "no_cache_final_loss": no_cache_loss[-1],
            "internal": internal_eval,
            "no_cache": no_cache_eval,
            "interventions": interventions,
            "observed_gate_metrics": observed,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
        runs.append(run)
        print(
            f"  gates={run['all_gates_pass']} "
            f"accuracy={internal_eval['accuracy']:.4f} "
            f"memory_advantage={observed['memory_and_tension_advantage_over_no_cache']:.4f}",
            flush=True,
        )

    result = {
        "audit": "branch_d_access_multiseed",
        "architecture": "query-key-addressed recurrent value states",
        "config": {
            "count_per_run": args.count,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "fusion_size": args.fusion_size,
            "device": args.device,
            "seed_pairs": [
                {"data_seed": data, "model_seed": model}
                for data, model in SEED_PAIRS
            ],
        },
        "predeclared_thresholds": THRESHOLDS,
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "all_gates_pass_rate": sum(run["all_gates_pass"] for run in runs)
            / len(runs),
            "per_gate_pass_rate": {
                name: sum(run["gates"][name] for run in runs) / len(runs)
                for name in THRESHOLDS
            },
            "minimum_internal_accuracy": min(
                run["internal"]["accuracy"] for run in runs
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
        "multi_seed_supported": all(run["all_gates_pass"] for run in runs),
        "robust_support_blocker": (
            "Replicate with an alternate relational architecture and a different surface-task "
            "variant before calling Branch D robust."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
