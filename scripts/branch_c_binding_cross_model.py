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

from attcon.binding import generate_binding_examples
from attcon.binding_experiment import (
    SetTransformerBindingModel,
    binding_intervention_metrics,
    evaluate_binding_model,
    parameter_count,
    tensorize_binding_examples,
    train_binding_model,
)
from scripts.branch_c_binding_pilot import THRESHOLDS, VARIANTS


VARIANT_SEEDS = {
    "original": {"data_seed": 811, "model_seed": 1201},
    "surface_v2": {"data_seed": 919, "model_seed": 1301},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_c_binding_cross_model.json")
    args = parser.parse_args()

    runs = []
    for variant_name, seeds in VARIANT_SEEDS.items():
        print(f"running set-transformer variant={variant_name}", flush=True)
        config = VARIANTS[variant_name]["config"]
        examples = generate_binding_examples(
            args.count, config=config, seed=seeds["data_seed"]
        )
        train = tensorize_binding_examples(
            [example for example in examples if example.split == "train"], config
        )
        heldout = tensorize_binding_examples(
            [example for example in examples if example.split == "heldout_conjunction"],
            config,
        )
        torch.manual_seed(seeds["model_seed"])
        integrated = SetTransformerBindingModel(
            config,
            args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
        torch.manual_seed(seeds["model_seed"])
        independent = SetTransformerBindingModel(
            config,
            args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            pool_objects=True,
        )
        integrated_loss = train_binding_model(
            integrated,
            train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=seeds["model_seed"],
            device=args.device,
        )
        independent_loss = train_binding_model(
            independent,
            train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=seeds["model_seed"],
            device=args.device,
        )
        integrated_eval = evaluate_binding_model(integrated, heldout, device=args.device)
        independent_eval = evaluate_binding_model(independent, heldout, device=args.device)
        interventions = binding_intervention_metrics(
            integrated, heldout, config, device=args.device
        )
        observed = {
            "integrated_heldout_joint_accuracy": integrated_eval["joint_accuracy"],
            "heldout_joint_advantage": integrated_eval["joint_accuracy"]
            - independent_eval["joint_accuracy"],
            "lure_rejection_advantage": integrated_eval[
                "false_binding_lure_rejection"
            ]
            - independent_eval["false_binding_lure_rejection"],
            **interventions,
        }
        gates = {
            name: observed[name] >= threshold
            for name, threshold in THRESHOLDS.items()
        }
        run = {
            "variant": variant_name,
            **seeds,
            "train_count": len(train),
            "heldout_count": len(heldout),
            "integrated_final_loss": integrated_loss[-1],
            "independent_final_loss": independent_loss[-1],
            "integrated_parameter_count": parameter_count(integrated),
            "independent_parameter_count": parameter_count(independent),
            "parameters_exactly_matched": parameter_count(integrated)
            == parameter_count(independent),
            "integrated": integrated_eval,
            "independent": independent_eval,
            "interventions": interventions,
            "observed_gate_metrics": observed,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
        runs.append(run)
        print(
            f"  gates={run['all_gates_pass']} "
            f"joint={integrated_eval['joint_accuracy']:.4f} "
            f"lure={integrated_eval['false_binding_lure_rejection']:.4f}",
            flush=True,
        )

    result = {
        "audit": "branch_c_binding_cross_model",
        "model_family": (
            "permutation-equivariant cue-token transformer; no explicit scalar object selector"
        ),
        "control": (
            "identical transformer and parameter count with object attributes mean-pooled "
            "before tokenization"
        ),
        "config": {
            "count_per_run": args.count,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
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
            "minimum_integrated_joint_accuracy": min(
                run["integrated"]["joint_accuracy"] for run in runs
            ),
            "minimum_integrated_lure_rejection": min(
                run["integrated"]["false_binding_lure_rejection"]
                for run in runs
            ),
            "minimum_target_type_follow_rate": min(
                run["interventions"]["target_type_follow_rate"] for run in runs
            ),
            "minimum_target_other_field_joint_stability": min(
                run["interventions"]["target_other_field_joint_stability"]
                for run in runs
            ),
            "minimum_non_target_all_field_invariance": min(
                run["interventions"]["non_target_all_field_invariance"]
                for run in runs
            ),
        },
    }
    result["cross_model_supported"] = (
        result["summary"]["all_variants_pass"]
        and result["summary"]["all_parameters_exactly_matched"]
    )
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
