from __future__ import annotations

"""Generate and audit the structurally different temporal-relay scaffold."""

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attcon.temporal_relay import (
    RELAY_STATUSES,
    TemporalRelayConfig,
    generate_temporal_relay_examples,
    validate_temporal_relay_example,
    validate_temporal_relay_group,
)


def build_temporal_relay_scaffold(episode_count: int = 2048, seed: int = 1103) -> dict:
    config = TemporalRelayConfig()
    examples = generate_temporal_relay_examples(
        episode_count, config=config, seed=seed
    )
    groups = defaultdict(list)
    example_failures = Counter()
    for example in examples:
        groups[example.pair_group_id].append(example)
        example_failures.update(validate_temporal_relay_example(example))
    group_failures = Counter()
    for group in groups.values():
        group_failures.update(validate_temporal_relay_group(group))
    statuses = Counter(example.target_status for example in examples)
    splits = Counter(example.split for example in examples)
    gates = {
        "all_examples_valid": not example_failures,
        "all_pair_groups_valid": not group_failures,
        "statuses_balanced": set(statuses) == set(RELAY_STATUSES)
        and len(set(statuses.values())) == 1,
        "train_and_heldout_present": set(splits)
        == {"train", "heldout_event_bundle"},
        "last_write_semantics": all(
            example.target
            == [
                event for event in example.events
                if event.entity == example.query_entity
            ][-1]
            for example in examples
        ),
        "same_event_identity_across_demands": all(
            example.target.event_id == example.target_event_id
            for example in examples
        ),
        "attention_fixed_elsewhere": all(
            example.current_attention_before == example.current_attention_after
            != example.target_event_index
            for example in examples
        ),
    }
    return {
        "audit": "stage8_temporal_relay_scaffold",
        "status": "scaffold_ready" if all(gates.values()) else "scaffold_invalid",
        "scope": (
            "dataset invariants only; ordered last-write resolution is structurally different "
            "from the spatial object-set benchmark, but no trained replication is claimed"
        ),
        "config": {**asdict(config), "episode_count": episode_count, "seed": seed},
        "counts": {
            "examples": len(examples),
            "pair_groups": len(groups),
            "statuses": dict(sorted(statuses.items())),
            "splits": dict(sorted(splits.items())),
        },
        "structural_difference_contract": {
            "source_structure": "ordered event stream with repeated entity updates",
            "selection_rule": "resolve the last write for a queried entity",
            "binding_target": "time-operation-payload chronology bundle",
            "access_transition": "delayed live/archive/conflict/missing query",
            "excluded_shortcut": "unordered spatial object selection",
        },
        "validation_failures": {
            "examples": dict(example_failures),
            "pair_groups": dict(group_failures),
        },
        "gates": gates,
        "all_scaffold_gates_pass": all(gates.values()),
        "different_benchmark_replication_established": False,
        "next_experiment": (
            "train sequence and matched order-destroying controls, then repeat the disjoint "
            "directional binding/access intervention on held-out event bundles"
        ),
    }


def main() -> None:
    result = build_temporal_relay_scaffold()
    output = ROOT / "audits" / "stage8_temporal_relay_scaffold.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
