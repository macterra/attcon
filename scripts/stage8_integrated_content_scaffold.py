from __future__ import annotations

"""Generate and audit the paired same-content scaffold for the Stage 8 experiment."""

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

from attcon.integrated_content import (
    IntegratedContentConfig,
    TARGET_STATUSES,
    generate_integrated_content_examples,
    validate_integrated_content_example,
    validate_paired_content_group,
)


def build_scaffold_audit(episode_count: int, seed: int) -> dict:
    config = IntegratedContentConfig()
    examples = generate_integrated_content_examples(
        episode_count, config=config, seed=seed
    )
    groups = defaultdict(list)
    example_failures = Counter()
    for example in examples:
        groups[example.pair_group_id].append(example)
        example_failures.update(validate_integrated_content_example(example))
    group_failures = Counter()
    for group in groups.values():
        group_failures.update(validate_paired_content_group(group))

    split_counts = Counter(example.split for example in examples)
    status_counts = Counter(example.target_status for example in examples)
    gates = {
        "all_examples_valid": not example_failures,
        "all_pair_groups_valid": not group_failures,
        "all_statuses_balanced": len(set(status_counts.values())) == 1
        and set(status_counts) == set(TARGET_STATUSES),
        "train_and_heldout_present": set(split_counts)
        == {"train", "heldout_content_bundle"},
        "same_identity_feeds_binding_and_access": all(
            example.binding_cue_content_id
            == example.switched_query_content_id
            == example.target.content_id
            for example in examples
        ),
        "target_attention_held_off_content": all(
            example.current_attention_after == example.current_attention_before
            and example.current_attention_after != example.target_index
            for example in examples
        ),
        "perturbation_point_reserved": all(
            example.shared_state_perturbation_step == 1 for example in examples
        ),
    }
    return {
        "audit": "stage8_integrated_content_scaffold",
        "status": "scaffold_ready" if all(gates.values()) else "scaffold_invalid",
        "scope": (
            "dataset and identity invariants only; no trained shared-state model, causal "
            "overlap result, or Stage 8 support is claimed"
        ),
        "config": {**asdict(config), "episode_count": episode_count, "seed": seed},
        "counts": {
            "examples": len(examples),
            "pair_groups": len(groups),
            "splits": dict(sorted(split_counts.items())),
            "target_statuses": dict(sorted(status_counts.items())),
        },
        "identity_contract": {
            "binding_target": "initial object selected by binding_cue_content_id",
            "access_target": "same object selected later by switched_query_content_id",
            "paired_control": "same initial target bundle under all four access statuses",
            "future_perturbation_target": "shared target state after selection and before both readouts",
        },
        "validation_failures": {
            "examples": dict(example_failures),
            "pair_groups": dict(group_failures),
        },
        "gates": gates,
        "all_scaffold_gates_pass": all(gates.values()),
        "same_content_causal_overlap_established": False,
        "next_experiment": (
            "Train one recurrent shared-state model plus identity-destroying and split-state "
            "controls; test held-out binding/access accuracy and bidirectional, content-specific "
            "effects of shared-state swaps and perturbations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument(
        "--out", default="audits/stage8_integrated_content_scaffold.json"
    )
    args = parser.parse_args()
    result = build_scaffold_audit(args.episodes, args.seed)
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
