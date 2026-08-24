from __future__ import annotations

"""Repeat the disjoint-split directional overlap audit over fresh seeds."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "scripts" / "stage8_integrated_content_directional.py"
SEED_TRIPLES = ((937, 941, 947), (953, 967, 971), (977, 983, 991))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default="audits/stage8_integrated_content_directional_multiseed.json",
    )
    args = parser.parse_args()

    runs = []
    with tempfile.TemporaryDirectory(prefix="attcon_stage8_directional_") as temp_dir:
        for data_seed, model_seed, null_seed in SEED_TRIPLES:
            output = Path(temp_dir) / f"run_{data_seed}_{model_seed}.json"
            print(
                f"running data={data_seed} model={model_seed} null={null_seed}",
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PILOT),
                    "--episodes",
                    str(args.episodes),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--hidden-size",
                    str(args.hidden_size),
                    "--alpha",
                    str(args.alpha),
                    "--device",
                    args.device,
                    "--data-seed",
                    str(data_seed),
                    "--model-seed",
                    str(model_seed),
                    "--null-seed",
                    str(null_seed),
                    "--out",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            run = json.loads(output.read_text())
            runs.append(run)
            print(
                f"  gates={run['all_directional_gates_pass']} "
                f"joint={run['observed']['shared_joint_value_donor_follow']:.4f} "
                f"null_adv={run['observed']['joint_advantage_over_permuted_direction']:.4f}",
                flush=True,
            )

    metric_names = tuple(runs[0]["observed"])
    result = {
        "audit": "stage8_integrated_content_directional_multiseed",
        "scope": (
            "seed robustness of disjoint-split, direction-limited causal overlap inside "
            "an architecturally imposed shared-state benchmark"
        ),
        "config": {
            "episodes_per_run": args.episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "alpha": args.alpha,
            "device": args.device,
            "seed_triples": [
                {
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "null_seed": null_seed,
                }
                for data_seed, model_seed, null_seed in SEED_TRIPLES
            ],
        },
        "predeclared_thresholds": runs[0]["predeclared_thresholds"],
        "runs": [
            {
                "data_seed": run["config"]["data_seed"],
                "model_seed": run["config"]["model_seed"],
                "null_seed": run["config"]["null_seed"],
                "heldout_count": run["config"]["heldout_count"],
                "observed": run["observed"],
                "gates": run["gates"],
                "all_directional_gates_pass": run["all_directional_gates_pass"],
            }
            for run in runs
        ],
        "summary": {
            "run_count": len(runs),
            "all_gates_pass_rate": sum(
                run["all_directional_gates_pass"] for run in runs
            )
            / len(runs),
            "per_gate_pass_rate": {
                name: sum(run["gates"][name] for run in runs) / len(runs)
                for name in runs[0]["gates"]
            },
            "minimum_metrics": {
                name: min(run["observed"][name] for run in runs)
                for name in metric_names
            },
        },
        "multi_seed_directional_engineering_support": all(
            run["all_directional_gates_pass"] for run in runs
        ),
        "stage8_same_content_gate_satisfied": False,
        "support_boundary": (
            "The result is cross-validated and seed-robust, but supervised directions are "
            "tested within an explicitly shared architecture rather than independently "
            "emergent theory-family mechanisms."
        ),
        "next_experiment": (
            "remove forced state sharing and test whether a jointly trained neutral architecture "
            "develops causal overlap beyond the exactly matched split-state null"
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
