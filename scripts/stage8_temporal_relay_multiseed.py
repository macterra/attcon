from __future__ import annotations

"""Repeat the relational temporal-relay pilot over fresh seed pairs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "scripts" / "stage8_temporal_relay_pilot.py"
SEED_PAIRS = ((1151, 1153), (1163, 1171), (1181, 1187))


def main() -> None:
    runs = []
    with tempfile.TemporaryDirectory(prefix="attcon_temporal_relay_") as temp_dir:
        for data_seed, model_seed in SEED_PAIRS:
            output = Path(temp_dir) / f"run_{data_seed}_{model_seed}.json"
            print(
                f"running data_seed={data_seed} model_seed={model_seed}", flush=True
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PILOT),
                    "--architecture",
                    "relational",
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
                f"  gates={run['all_pilot_gates_pass']} "
                f"task={run['observed']['shared_task_joint_accuracy']:.4f} "
                f"coordination={run['observed']['coordination_advantage_over_split']:.4f}",
                flush=True,
            )
    metric_names = tuple(runs[0]["observed"])
    result = {
        "audit": "stage8_temporal_relay_multiseed",
        "scope": (
            "fresh-seed replication of the frozen relational temporal-relay gates on the "
            "structurally different ordered event-stream benchmark"
        ),
        "config": {
            "episodes_per_run": 2048,
            "epochs": 35,
            "architecture": "relational_gru",
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
                "observed": run["observed"],
                "gates": run["gates"],
                "all_pilot_gates_pass": run["all_pilot_gates_pass"],
            }
            for run in runs
        ],
        "summary": {
            "run_count": len(runs),
            "support_rate": sum(run["all_pilot_gates_pass"] for run in runs)
            / len(runs),
            "minimum_metrics": {
                name: min(run["observed"][name] for run in runs)
                for name in metric_names
            },
            "per_gate_pass_rate": {
                name: sum(run["gates"][name] for run in runs) / len(runs)
                for name in runs[0]["gates"]
            },
        },
        "multi_seed_different_benchmark_engineering_support": all(
            run["all_pilot_gates_pass"] for run in runs
        ),
        "different_benchmark_replication_established": False,
        "support_boundary": (
            "Seed-robust transfer would establish the engineered assay on a different benchmark, "
            "but not independent emergence because relational matching and shared state remain "
            "explicit architectural features."
        ),
        "next_experiment": (
            "replicate with a structurally different sequence architecture and remove forced "
            "state sharing before upgrading the Stage 8 benchmark gate"
        ),
    }
    output = ROOT / "audits" / "stage8_temporal_relay_multiseed.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
