from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attcon.higher_order import (
    HIGHER_ORDER_STATUSES,
    HigherOrderConfig,
    generate_higher_order_examples,
    validate_counterbalance_group,
    validate_higher_order_example,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4092)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="audits/branch_e_higher_order_dataset.json")
    args = parser.parse_args()

    config = HigherOrderConfig()
    examples = generate_higher_order_examples(args.count, config=config, seed=args.seed)
    groups = defaultdict(list)
    for example in examples:
        groups[example.counterbalance_group].append(example)
    example_failures = [
        {"example_id": example.example_id, "failures": failures}
        for example in examples
        if (failures := validate_higher_order_example(example))
    ]
    group_failures = [
        {"group": group, "failures": failures}
        for group, cases in groups.items()
        if (failures := validate_counterbalance_group(cases))
    ]
    status_counts = Counter(example.status for example in examples)
    split_counts = Counter(example.split for example in examples)

    # Best possible status classification using only first-order content is the majority
    # status within each identical-content signature. Observation-only additionally sees
    # whether the matching content is currently visible, but not the access gate.
    first_order_oracle_correct = 0
    observation_oracle_correct = 0
    by_content = defaultdict(Counter)
    by_observation = defaultdict(Counter)
    for example in examples:
        by_content[(example.content_key, example.content_value)][example.status] += 1
        observation_signature = (
            example.content_key,
            example.content_value,
            example.current_observation_value,
        )
        by_observation[observation_signature][example.status] += 1
    first_order_oracle_correct = sum(max(counts.values()) for counts in by_content.values())
    observation_oracle_correct = sum(
        max(counts.values()) for counts in by_observation.values()
    )
    result = {
        "audit": "branch_e_higher_order_dataset",
        "status": (
            "scaffold_valid"
            if not example_failures and not group_failures
            else "scaffold_invalid"
        ),
        "config": {**asdict(config), "count": args.count, "seed": args.seed},
        "status_counts": dict(status_counts),
        "split_counts": dict(split_counts),
        "counterbalance_group_count": len(groups),
        "complete_counterbalance_rate": sum(
            not validate_counterbalance_group(cases) for cases in groups.values()
        )
        / len(groups),
        "content_only_best_status_accuracy": first_order_oracle_correct / len(examples),
        "observation_only_best_status_accuracy": observation_oracle_correct
        / len(examples),
        "fresh_current_wrong_access_observation_match_rate": sum(
            next(
                case
                for case in cases
                if case.status == "fresh_current"
            ).current_observation_value
            == next(
                case
                for case in cases
                if case.status == "wrong_access_lure"
            ).current_observation_value
            for cases in groups.values()
        )
        / len(groups),
        "inferred_content_lure_count": status_counts["inferred_content"],
        "stale_access_lure_count": status_counts["stale_access_lure"],
        "wrong_access_lure_count": status_counts["wrong_access_lure"],
        "invalid_example_count": len(example_failures),
        "invalid_group_count": len(group_failures),
        "invalid_examples": example_failures[:20],
        "invalid_groups": group_failures[:20],
        "group_preview": [case.to_dict() for case in next(iter(groups.values()))],
        "branch_e_supported": False,
        "next_required_experiment": (
            "Learn higher-order access/source/confidence state without supervising the exact "
            "evaluation labels, then compare against first-order and observation-only probes "
            "and intervene while holding content fixed."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
