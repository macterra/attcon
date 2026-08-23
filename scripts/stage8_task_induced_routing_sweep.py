from __future__ import annotations

"""Aggregate the frozen-threshold task-induced routing pressure sweep."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "audits"
CONDITIONS = (
    (0.0, 30, "stage8_neutral_routing_pilot.json"),
    (0.5, 30, "stage8_task_induced_routing_pilot.json"),
    (0.75, 30, "stage8_task_induced_routing_dropout_075.json"),
    (0.9, 30, "stage8_task_induced_routing_dropout_090.json"),
    (0.95, 30, "stage8_task_induced_routing_dropout_095.json"),
    (0.95, 60, "stage8_task_induced_routing_dropout_095_epochs_60.json"),
)


def build_sweep() -> dict:
    runs = []
    for dropout, epochs, filename in CONDITIONS:
        result = json.loads((AUDITS / filename).read_text())
        runs.append(
            {
                "private_access_dropout": dropout,
                "epochs": epochs,
                "artifact": f"audits/{filename}",
                "status": result["status"],
                "observed": result["observed"],
                "gates": result["gates"],
                "all_routing_gates_pass": all(result["gates"].values()),
            }
        )
    supported = [run for run in runs if run["all_routing_gates_pass"]]
    return {
        "audit": "stage8_task_induced_routing_sweep",
        "scope": (
            "post-pilot pressure mapping under unchanged routing thresholds; private-lane "
            "dropout supplies an indirect robustness demand, never a gate or overlap label"
        ),
        "predeclared_thresholds": json.loads(
            (AUDITS / CONDITIONS[0][2]).read_text()
        )["predeclared_thresholds"],
        "runs": runs,
        "summary": {
            "condition_count": len(runs),
            "supported_condition_count": len(supported),
            "no_pressure_supported": runs[0]["all_routing_gates_pass"],
            "no_pressure_routing_weight_change": runs[0]["observed"][
                "routing_weight_increase"
            ],
            "no_pressure_joint_directional_follow": runs[0]["observed"][
                "learned_joint_directional_follow"
            ],
            "first_supported_condition": (
                {
                    "private_access_dropout": supported[0][
                        "private_access_dropout"
                    ],
                    "epochs": supported[0]["epochs"],
                }
                if supported
                else None
            ),
            "supported_endpoint": supported[-1]["observed"] if supported else None,
        },
        "task_induced_routing_pilot_supported": bool(supported),
        "stage8_same_content_gate_satisfied": False,
        "support_boundary": (
            "Routing does not emerge under ordinary joint supervision. It appears only under "
            "severe stochastic loss of the private access lane and is currently single-seed on "
            "one engineered benchmark."
        ),
        "next_experiment": (
            "replicate the 0.95/60-epoch endpoint and the zero-pressure negative control across "
            "fresh seeds, then test a less benchmark-specific robustness demand"
        ),
    }


def main() -> None:
    result = build_sweep()
    output = AUDITS / "stage8_task_induced_routing_sweep.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
