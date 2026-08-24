from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attcon.counterfactual_access import (
    CounterfactualAccessConfig,
    TARGET_STATUSES,
    generate_counterfactual_access_examples,
    validate_counterfactual_access_example,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="audits/branch_d_access_dataset.json")
    args = parser.parse_args()

    config = CounterfactualAccessConfig()
    examples = generate_counterfactual_access_examples(
        args.count, config=config, seed=args.seed
    )
    failures = [
        {"example_id": example.example_id, "failures": validation}
        for example in examples
        if (validation := validate_counterfactual_access_example(example))
    ]
    status_counts = Counter(example.target_status for example in examples)
    split_counts = Counter(example.split for example in examples)
    by_status = {}
    for status in TARGET_STATUSES:
        cases = [example for example in examples if example.target_status == status]
        by_status[status] = {
            "count": len(cases),
            "scene_only_accuracy": sum(
                case.scene_only_answer == case.expected_answer for case in cases
            )
            / len(cases),
            "current_glimpse_accuracy": sum(
                case.current_glimpse_answer == case.expected_answer for case in cases
            )
            / len(cases),
            "attention_held_fixed_rate": sum(
                case.current_attention_before == case.current_attention_after
                for case in cases
            )
            / len(cases),
        }
    result = {
        "audit": "branch_d_access_dataset",
        "status": "scaffold_valid" if not failures else "scaffold_invalid",
        "config": {**asdict(config), "count": args.count, "seed": args.seed},
        "target_status_counts": dict(status_counts),
        "split_counts": dict(split_counts),
        "by_status": by_status,
        "counterfactual_tension_rate": sum(
            example.scene_only_answer != example.expected_answer
            for example in examples
            if example.target_status == "counterfactually_accessible"
        )
        / status_counts["counterfactually_accessible"],
        "query_switch_rate": sum(
            example.initial_query_key != example.switched_query_key
            for example in examples
        )
        / len(examples),
        "attention_held_fixed_rate": sum(
            example.current_attention_before == example.current_attention_after
            for example in examples
        )
        / len(examples),
        "invalid_count": len(failures),
        "invalid_examples": failures[:20],
        "example_preview": [example.to_dict() for example in examples[:4]],
        "branch_d_supported": False,
        "next_required_experiment": (
            "Train an internal-access route and compare it with scene-only, current-glimpse, "
            "symbolic-dump, and matched no-cache controls on the held-out query-value split."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
