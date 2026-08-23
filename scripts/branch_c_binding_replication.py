from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="audits/branch_c_binding_pilot.json")
    parser.add_argument(
        "--replication", default="audits/branch_c_binding_surface_variant.json"
    )
    parser.add_argument("--out", default="audits/branch_c_binding_replication.json")
    args = parser.parse_args()

    original = _load(args.original)
    replication = _load(args.replication)
    config_fields = (
        "grid_size",
        "num_objects",
        "num_visible_types",
        "digit_vocab_size",
        "num_cues",
        "heldout_modulus",
    )
    original_config = {name: original["config"][name] for name in config_fields}
    replication_config = {
        name: replication["config"][name] for name in config_fields
    }
    checks = {
        "original_all_gates_pass": original["all_pilot_gates_pass"],
        "replication_all_gates_pass": replication["all_pilot_gates_pass"],
        "thresholds_frozen_across_variants": (
            original["predeclared_thresholds"]
            == replication["predeclared_thresholds"]
        ),
        "surface_schemas_differ": (
            original["surface_schema"] != replication["surface_schema"]
        ),
        "benchmark_cardinalities_differ": original_config != replication_config,
        "independent_baselines_capacity_advantaged": (
            original["models"]["baseline_to_integrated_parameter_ratio"] >= 1.0
            and replication["models"]["baseline_to_integrated_parameter_ratio"] >= 1.0
        ),
    }
    result = {
        "audit": "branch_c_binding_replication",
        "original_artifact": args.original,
        "replication_artifact": args.replication,
        "original_variant": original["variant"],
        "replication_variant": replication["variant"],
        "original_surface_schema": original["surface_schema"],
        "replication_surface_schema": replication["surface_schema"],
        "original_config": original_config,
        "replication_config": replication_config,
        "original_metrics": {
            "integrated_joint_accuracy": original["models"]["integrated"]["heldout"][
                "joint_accuracy"
            ],
            "integrated_lure_rejection": original["models"]["integrated"]["heldout"][
                "false_binding_lure_rejection"
            ],
            "independent_joint_accuracy": original["models"][
                "independent_feature_baseline"
            ]["heldout"]["joint_accuracy"],
            "independent_lure_rejection": original["models"][
                "independent_feature_baseline"
            ]["heldout"]["false_binding_lure_rejection"],
        },
        "replication_metrics": {
            "integrated_joint_accuracy": replication["models"]["integrated"][
                "heldout"
            ]["joint_accuracy"],
            "integrated_lure_rejection": replication["models"]["integrated"][
                "heldout"
            ]["false_binding_lure_rejection"],
            "independent_joint_accuracy": replication["models"][
                "independent_feature_baseline"
            ]["heldout"]["joint_accuracy"],
            "independent_lure_rejection": replication["models"][
                "independent_feature_baseline"
            ]["heldout"]["false_binding_lure_rejection"],
        },
        "checks": checks,
        "branch_c_supported": all(checks.values()),
        "support_scope": (
            "bounded synthetic-benchmark support; robustness still requires multiple model "
            "seeds and replication outside the shared-selection benchmark family"
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
