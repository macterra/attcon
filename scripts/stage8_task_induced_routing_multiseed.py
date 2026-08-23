from __future__ import annotations

"""Replicate zero-pressure and task-induced routing endpoints over fresh seeds."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "scripts" / "stage8_neutral_routing_pilot.py"
SEED_TRIPLES = ((1019, 1021, 1031), (1033, 1039, 1049), (1051, 1061, 1063))


def _run(
    args: argparse.Namespace,
    output: Path,
    data_seed: int,
    model_seed: int,
    null_seed: int,
    *,
    dropout: float,
    epochs: int,
) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(PILOT),
            "--episodes",
            str(args.episodes),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--hidden-size",
            str(args.hidden_size),
            "--initial-routing-weight",
            str(args.initial_routing_weight),
            "--private-access-dropout",
            str(dropout),
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
    return json.loads(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--initial-routing-weight", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out", default="audits/stage8_task_induced_routing_multiseed.json"
    )
    args = parser.parse_args()

    runs = []
    with tempfile.TemporaryDirectory(prefix="attcon_stage8_routing_") as temp_dir:
        for data_seed, model_seed, null_seed in SEED_TRIPLES:
            print(
                f"running data={data_seed} model={model_seed} null={null_seed}",
                flush=True,
            )
            neutral = _run(
                args,
                Path(temp_dir) / f"neutral_{data_seed}_{model_seed}.json",
                data_seed,
                model_seed,
                null_seed,
                dropout=0.0,
                epochs=30,
            )
            induced = _run(
                args,
                Path(temp_dir) / f"induced_{data_seed}_{model_seed}.json",
                data_seed,
                model_seed,
                null_seed,
                dropout=0.95,
                epochs=60,
            )
            runs.append(
                {
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "null_seed": null_seed,
                    "zero_pressure": {
                        "observed": neutral["observed"],
                        "gates": neutral["gates"],
                        "all_routing_gates_pass": neutral[
                            "all_routing_gates_pass"
                        ],
                    },
                    "task_induced": {
                        "observed": induced["observed"],
                        "gates": induced["gates"],
                        "all_routing_gates_pass": induced[
                            "all_routing_gates_pass"
                        ],
                    },
                }
            )
            print(
                "  neutral_joint="
                f"{neutral['observed']['learned_joint_directional_follow']:.4f} "
                "induced_joint="
                f"{induced['observed']['learned_joint_directional_follow']:.4f} "
                f"induced_gates={induced['all_routing_gates_pass']}",
                flush=True,
            )

    metric_names = tuple(runs[0]["task_induced"]["observed"])
    result = {
        "audit": "stage8_task_induced_routing_multiseed",
        "scope": (
            "three fresh-seed replications of the zero-pressure negative control and the "
            "0.95-dropout/60-epoch task-induced routing endpoint"
        ),
        "config": {
            "episodes_per_condition": args.episodes,
            "zero_pressure_epochs": 30,
            "task_induced_epochs": 60,
            "task_induced_private_access_dropout": 0.95,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "initial_routing_weight": args.initial_routing_weight,
            "alpha": args.alpha,
            "device": args.device,
            "seed_triples": [
                {
                    "data_seed": data,
                    "model_seed": model,
                    "null_seed": null,
                }
                for data, model, null in SEED_TRIPLES
            ],
        },
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "zero_pressure_support_rate": sum(
                run["zero_pressure"]["all_routing_gates_pass"] for run in runs
            )
            / len(runs),
            "task_induced_support_rate": sum(
                run["task_induced"]["all_routing_gates_pass"] for run in runs
            )
            / len(runs),
            "maximum_zero_pressure_joint_directional_follow": max(
                run["zero_pressure"]["observed"][
                    "learned_joint_directional_follow"
                ]
                for run in runs
            ),
            "minimum_task_induced_metrics": {
                name: min(
                    run["task_induced"]["observed"][name] for run in runs
                )
                for name in metric_names
            },
            "per_gate_task_induced_pass_rate": {
                name: sum(run["task_induced"]["gates"][name] for run in runs)
                / len(runs)
                for name in runs[0]["task_induced"]["gates"]
            },
        },
        "multi_seed_task_induced_routing_support": all(
            run["task_induced"]["all_routing_gates_pass"] for run in runs
        )
        and not any(
            run["zero_pressure"]["all_routing_gates_pass"] for run in runs
        ),
        "stage8_same_content_gate_satisfied": False,
        "support_boundary": (
            "Even if seed-robust, this is one synthetic benchmark with severe private-lane "
            "dropout; it is task-induced engineering evidence, not spontaneous cross-family "
            "convergence."
        ),
        "next_experiment": (
            "replace explicit private-lane dropout with distributional occlusion or resource "
            "competition, then replicate on a structurally different task"
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
