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

from attcon.broadcast import (
    CONSUMERS,
    BroadcastConfig,
    generate_broadcast_examples,
    validate_broadcast_example,
    validate_broadcast_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4095)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="audits/branch_f_broadcast_dataset.json")
    args = parser.parse_args()

    config = BroadcastConfig()
    examples = generate_broadcast_examples(args.count, config=config, seed=args.seed)
    sweeps = defaultdict(list)
    for example in examples:
        sweeps[example.sweep_group].append(example)
    example_failures = [
        {"example_id": example.example_id, "failures": failures}
        for example in examples
        if (failures := validate_broadcast_example(example))
    ]
    sweep_failures = [
        {"sweep_group": group, "failures": failures}
        for group, cases in sweeps.items()
        if (failures := validate_broadcast_sweep(cases))
    ]
    split_counts = Counter(example.split for example in examples)
    evidence_counts = Counter(example.evidence_quality for example in examples)
    ignited = [example for example in examples if example.ignited]
    result = {
        "audit": "branch_f_broadcast_dataset",
        "status": (
            "scaffold_valid"
            if not example_failures and not sweep_failures
            else "scaffold_invalid"
        ),
        "config": {**asdict(config), "count": args.count, "seed": args.seed},
        "consumers": list(CONSUMERS),
        "split_counts": dict(split_counts),
        "evidence_level_counts": dict(evidence_counts),
        "sweep_group_count": len(sweeps),
        "complete_threshold_crossing_sweep_rate": sum(
            not validate_broadcast_sweep(cases) for cases in sweeps.values()
        )
        / len(sweeps),
        "ignition_rate": len(ignited) / len(examples),
        "local_action_availability_rate": sum(
            example.consumer_available[0] for example in examples
        )
        / len(examples),
        "ignited_broad_consumer_alignment_rate": sum(
            len(set(example.consumer_onset_step[1:])) == 1
            and all(example.consumer_available[1:])
            for example in ignited
        )
        / len(ignited),
        "nonignited_broad_consumer_unavailability_rate": sum(
            not any(example.consumer_available[1:])
            for example in examples
            if not example.ignited
        )
        / (len(examples) - len(ignited)),
        "invalid_example_count": len(example_failures),
        "invalid_sweep_count": len(sweep_failures),
        "invalid_examples": example_failures[:20],
        "invalid_sweeps": sweep_failures[:20],
        "sweep_preview": [case.to_dict() for case in next(iter(sweeps.values()))],
        "branch_f_supported": False,
        "next_required_experiment": (
            "Train a shared broadcast bottleneck and capacity-matched private-head comparator, "
            "then intervene on the shared state and test coordinated content-specific effects."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
