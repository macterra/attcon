from __future__ import annotations

"""Repeat the Stage 8 integrated-content causal pilot over fresh seed pairs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "scripts" / "stage8_integrated_content_pilot.py"
SEED_PAIRS = ((839, 853), (857, 859), (863, 877))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", default="audits/stage8_integrated_content_multiseed.json"
    )
    args = parser.parse_args()

    runs = []
    with tempfile.TemporaryDirectory(prefix="attcon_stage8_integrated_") as temp_dir:
        for data_seed, model_seed in SEED_PAIRS:
            output = Path(temp_dir) / f"run_{data_seed}_{model_seed}.json"
            print(
                f"running data_seed={data_seed} model_seed={model_seed}", flush=True
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
                f"  gates={run['all_engineering_pilot_gates_pass']} "
                f"coordination={run['observed']['coordination_advantage_over_split']:.4f}",
                flush=True,
            )

    metric_names = tuple(runs[0]["observed"])
    result = {
        "audit": "stage8_integrated_content_multiseed",
        "scope": (
            "multi-seed robustness of the engineered same-content assay; the shared "
            "bottleneck remains imposed and this is still one synthetic benchmark"
        ),
        "config": {
            "episodes_per_run": args.episodes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "device": args.device,
            "seed_pairs": [
                {"data_seed": data_seed, "model_seed": model_seed}
                for data_seed, model_seed in SEED_PAIRS
            ],
        },
        "predeclared_thresholds": runs[0]["predeclared_thresholds"],
        "runs": [
            {
                "data_seed": run["config"]["data_seed"],
                "model_seed": run["config"]["model_seed"],
                "heldout_count": run["config"]["heldout_count"],
                "parameter_counts": {
                    mode: run["models"][mode]["parameter_count"]
                    for mode in ("shared", "split", "pooled")
                },
                "observed": run["observed"],
                "gates": run["gates"],
                "all_engineering_pilot_gates_pass": run[
                    "all_engineering_pilot_gates_pass"
                ],
            }
            for run in runs
        ],
        "summary": {
            "run_count": len(runs),
            "all_gates_pass_rate": sum(
                run["all_engineering_pilot_gates_pass"] for run in runs
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
        "multi_seed_engineering_support": all(
            run["all_engineering_pilot_gates_pass"] for run in runs
        ),
        "stage8_same_content_gate_satisfied": False,
        "support_boundary": (
            "Seed robustness alone cannot establish spontaneous overlap: the shared state is "
            "architecturally imposed, the intervention replaces the entire state rather than "
            "a direction learned on disjoint data, and no different benchmark is included."
        ),
        "next_experiment": (
            "fit content directions using training representations only, intervene along those "
            "directions on held-out bundles, and compare cross-branch effects with permuted-label "
            "and split-state nulls"
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
