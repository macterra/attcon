from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "scripts" / "branch_f_broadcast_pilot.py"
SEED_PAIRS = ((101, 401), (211, 503), (307, 601))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8190)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_f_broadcast_multiseed.json")
    args = parser.parse_args()

    runs = []
    with tempfile.TemporaryDirectory(prefix="attcon_branch_f_") as temp_dir:
        for data_seed, model_seed in SEED_PAIRS:
            output = Path(temp_dir) / f"run_{data_seed}_{model_seed}.json"
            print(
                f"running data_seed={data_seed} model_seed={model_seed}", flush=True
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PILOT),
                    "--count",
                    str(args.count),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--hidden-size",
                    str(args.hidden_size),
                    "--device",
                    args.device,
                    "--data-seed",
                    str(data_seed),
                    "--model-seed",
                    str(model_seed),
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
                f"  gates={run['all_engineering_gates_pass']} "
                f"coordination={run['observed_gate_metrics']['coordinated_ablation_advantage']:.4f}",
                flush=True,
            )

    metric_names = tuple(runs[0]["observed_gate_metrics"])
    result = {
        "audit": "branch_f_broadcast_multiseed",
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
        "predeclared_thresholds": runs[0]["predeclared_thresholds"],
        "runs": [
            {
                "data_seed": run["config"]["data_seed"],
                "model_seed": run["config"]["model_seed"],
                "heldout_count": run["config"]["heldout_count"],
                "shared_parameter_count": run["shared_model"]["parameter_count"],
                "private_parameter_count": run["private_shortcut_upper_bound"][
                    "parameter_count"
                ],
                "parameters_exactly_matched": run["private_shortcut_upper_bound"][
                    "parameters_exactly_matched"
                ],
                "shared_heldout": run["shared_model"]["heldout"],
                "private_heldout": run["private_shortcut_upper_bound"]["heldout"],
                "interventions": run["interventions"],
                "observed_gate_metrics": run["observed_gate_metrics"],
                "gates": run["gates"],
                "all_engineering_gates_pass": run["all_engineering_gates_pass"],
            }
            for run in runs
        ],
        "summary": {
            "run_count": len(runs),
            "all_gates_pass_rate": sum(
                run["all_engineering_gates_pass"] for run in runs
            )
            / len(runs),
            "per_gate_pass_rate": {
                name: sum(run["gates"][name] for run in runs) / len(runs)
                for name in runs[0]["gates"]
            },
            "minimum_metrics": {
                name: min(run["observed_gate_metrics"][name] for run in runs)
                for name in metric_names
                if name != "private_single_route_accuracy_drop_ceiling"
            },
            "maximum_private_single_route_accuracy_drop": max(
                run["observed_gate_metrics"][
                    "private_single_route_accuracy_drop_ceiling"
                ]
                for run in runs
            ),
        },
        "multi_seed_engineering_support": all(
            run["all_engineering_gates_pass"] for run in runs
        ),
        "branch_f_supported": False,
        "support_boundary": (
            "Seed robustness does not remove the imposed-bottleneck/direct-supervision caveat; "
            "this remains engineering support only."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
