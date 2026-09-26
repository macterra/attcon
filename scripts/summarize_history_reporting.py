from __future__ import annotations

"""Summarize comparable paired-history runs without relaxing any gates."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("audits/paired_history_reporting_coverage_multiseed.json"))
    parser.add_argument("--audit-name", default="paired_history_reporting_coverage_multiseed")
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.artifacts]
    if len(runs) < 3 or len({run["settings"]["seed"] for run in runs}) != len(runs):
        parser.error("at least three independent seed runs are required")
    reference = runs[0]
    for run in runs:
        for field in ("audit", "config", "thresholds", "split_group_counts"):
            if run[field] != reference[field]:
                parser.error(f"incomparable {field}")
        for setting in ("epochs", "probe_steps", "null_fits", "norm_matched_controls", "train_groups"):
            if run["settings"][setting] != reference["settings"][setting]:
                parser.error(f"incomparable {setting}")
        for setting in ("architecture", "fit_groups"):
            if run["settings"].get(setting) != reference["settings"].get(setting):
                parser.error(f"incomparable {setting}")
        if not all(run["dataset_checks"].values()):
            parser.error("invalid dataset checks")
        if "source_sha256" in reference and run.get("source_sha256") != reference["source_sha256"]:
            parser.error("incomparable source fingerprints")
        if set(run["observed"]) != set(reference["observed"]):
            parser.error("incomparable observed metrics")
        expected = {key: run["observed"][key] >= threshold for key, threshold in run["thresholds"].items()}
        if expected != run["gates"]:
            parser.error("artifact gate values do not match its thresholds")
    gates = {key: all(run["gates"][key] for run in runs) for key in reference["gates"]}
    result = {
        "audit": args.audit_name,
        "sources": [str(path) for path in args.artifacts],
        "seeds": [run["settings"]["seed"] for run in runs],
        "status": "replicated_bounded_memory_report_coupling" if all(gates.values()) else "reporting_gates_not_met",
        "all_seed_gates": gates,
        "gate_pass_counts": {key: sum(run["gates"][key] for run in runs) for key in gates},
        "observed_ranges": {
            key: {"min": min(run["observed"][key] for run in runs), "max": max(run["observed"][key] for run in runs)}
            for key in reference["observed"]
        },
        "baseline_balanced_accuracy_ranges": {
            baseline: {
                "min": min(run["evaluations"]["test"]["reports"][baseline]["balanced_accuracy"] for run in runs),
                "max": max(run["evaluations"]["test"]["reports"][baseline]["balanced_accuracy"] for run in runs),
            } for baseline in reference["evaluations"]["test"]["reports"]
        },
        "thresholds": reference["thresholds"],
        "claim_boundary": "Replication within one synthetic task and one GRU architecture. Reporters are supervised measurements of a frozen task-trained state. Gate failures remain failures; no Stage 8 upgrade or consciousness claim.",
    }
    if all("uncalibrated_state" in run["evaluations"]["test"]["reports"] for run in runs):
        result["within_seed_calibration_changes"] = {
            str(run["settings"]["seed"]): {
                metric: run["evaluations"]["test"]["reports"]["state"][metric]
                - run["evaluations"]["test"]["reports"]["uncalibrated_state"][metric]
                for metric in ("seen_value_accuracy", "unseen_unknown_accuracy", "paired_accuracy", "balanced_accuracy")
            } for run in runs
        }
        result["selected_readouts"] = {
            str(run["settings"]["seed"]): {
                name: selection["selected"] for name, selection in run["selections"].items()
            } for run in runs
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
