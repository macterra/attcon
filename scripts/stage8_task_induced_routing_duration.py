from __future__ import annotations

"""Frozen-threshold 90-epoch duration check for task-induced routing."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "scripts" / "stage8_neutral_routing_pilot.py"
PRIOR = ROOT / "audits" / "stage8_task_induced_routing_multiseed.json"
SEED_TRIPLES = ((1019, 1021, 1031), (1033, 1039, 1049), (1051, 1061, 1063))


def main() -> None:
    runs = []
    with tempfile.TemporaryDirectory(prefix="attcon_stage8_routing_duration_") as temp_dir:
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
                    "--private-access-dropout",
                    "0.95",
                    "--epochs",
                    "90",
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
            runs.append(
                {
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "null_seed": null_seed,
                    "observed": run["observed"],
                    "gates": run["gates"],
                    "all_routing_gates_pass": run["all_routing_gates_pass"],
                }
            )
            print(
                f"  gates={run['all_routing_gates_pass']} "
                f"route={run['observed']['final_routing_weight']:.4f} "
                f"blocked_task={run['observed']['blocked_task_joint_accuracy']:.4f} "
                f"joint={run['observed']['learned_joint_directional_follow']:.4f}",
                flush=True,
            )

    prior = json.loads(PRIOR.read_text())
    metric_names = tuple(runs[0]["observed"])
    result = {
        "audit": "stage8_task_induced_routing_duration",
        "scope": (
            "post-replication duration check: the 0.95 private-lane dropout condition is "
            "extended from 60 to 90 epochs with every model and gate otherwise unchanged"
        ),
        "config": {
            "episodes_per_run": 2048,
            "epochs": 90,
            "private_access_dropout": 0.95,
            "seed_triples": [
                {
                    "data_seed": data,
                    "model_seed": model,
                    "null_seed": null,
                }
                for data, model, null in SEED_TRIPLES
            ],
        },
        "predeclared_thresholds": json.loads(
            (ROOT / "audits" / "stage8_task_induced_routing_dropout_095_epochs_60.json").read_text()
        )["predeclared_thresholds"],
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "prior_60_epoch_support_rate": prior["summary"][
                "task_induced_support_rate"
            ],
            "support_rate": sum(run["all_routing_gates_pass"] for run in runs)
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
        "duration_robust_support": all(
            run["all_routing_gates_pass"] for run in runs
        ),
        "stage8_same_content_gate_satisfied": False,
        "support_boundary": (
            "This is a post-replication duration check on the same synthetic benchmark and "
            "severe engineered dropout condition; longer optimization is part of the result."
        ),
        "next_experiment": (
            "replace private-lane dropout with distributional occlusion or resource competition "
            "and repeat on a genuinely different benchmark"
        ),
    }
    output = ROOT / "audits" / "stage8_task_induced_routing_duration.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
