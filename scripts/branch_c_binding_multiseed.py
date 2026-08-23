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
    IndependentFeatureBaseline,
    SharedSelectionBindingModel,
    binding_intervention_metrics,
    evaluate_binding_model,
    parameter_count,
    tensorize_binding_examples,
    train_binding_model,
)
from scripts.branch_c_binding_pilot import THRESHOLDS, VARIANTS


SEED_PAIRS = ((101, 401), (211, 503), (307, 601))


def _run_seed(
    variant_name: str,
    data_seed: int,
    model_seed: int,
    *,
    count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    device: str,
) -> dict:
    config = VARIANTS[variant_name]["config"]
    examples = generate_binding_examples(count, config=config, seed=data_seed)
    train = tensorize_binding_examples(
        [example for example in examples if example.split == "train"], config
    )
    heldout = tensorize_binding_examples(
        [example for example in examples if example.split == "heldout_conjunction"],
        config,
    )
    torch.manual_seed(model_seed)
    integrated = SharedSelectionBindingModel(config, hidden_size)
    torch.manual_seed(model_seed)
    independent = IndependentFeatureBaseline(config, hidden_size)
    integrated_loss = train_binding_model(
        integrated,
        train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=model_seed,
        device=device,
    )
    independent_loss = train_binding_model(
        independent,
        train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=model_seed,
        device=device,
    )
    integrated_eval = evaluate_binding_model(integrated, heldout, device=device)
    independent_eval = evaluate_binding_model(independent, heldout, device=device)
    interventions = binding_intervention_metrics(
        integrated, heldout, config, device=device
    )
    observed = {
        "integrated_heldout_joint_accuracy": integrated_eval["joint_accuracy"],
        "heldout_joint_advantage": integrated_eval["joint_accuracy"]
        - independent_eval["joint_accuracy"],
        "lure_rejection_advantage": integrated_eval["false_binding_lure_rejection"]
        - independent_eval["false_binding_lure_rejection"],
        **interventions,
    }
    gates = {
        name: observed[name] >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    return {
        "variant": variant_name,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "train_count": len(train),
        "heldout_count": len(heldout),
        "integrated_parameter_count": parameter_count(integrated),
        "independent_parameter_count": parameter_count(independent),
        "integrated_final_loss": integrated_loss[-1],
        "independent_final_loss": independent_loss[-1],
        "integrated": integrated_eval,
        "independent": independent_eval,
        "interventions": interventions,
        "observed_gate_metrics": observed,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_c_binding_multiseed.json")
    args = parser.parse_args()

    runs = []
    for variant_name in VARIANTS:
        for data_seed, model_seed in SEED_PAIRS:
            print(
                f"running variant={variant_name} data_seed={data_seed} "
                f"model_seed={model_seed}",
                flush=True,
            )
            run = _run_seed(
                variant_name,
                data_seed,
                model_seed,
                count=args.count,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                hidden_size=args.hidden_size,
                device=args.device,
            )
            runs.append(run)
            print(
                f"  gates={run['all_gates_pass']} "
                f"joint={run['integrated']['joint_accuracy']:.4f} "
                f"lure={run['integrated']['false_binding_lure_rejection']:.4f}",
                flush=True,
            )

    gate_names = tuple(THRESHOLDS)
    result = {
        "audit": "branch_c_binding_multiseed",
        "config": {
            "count_per_run": args.count,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
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
                for name in gate_names
            },
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
        "multi_seed_supported": all(run["all_gates_pass"] for run in runs),
        "robust_support_blocker": (
            "The shared-selection bottleneck is common to every run; replicate with a "
            "structurally different binding model before calling Branch C robust."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
