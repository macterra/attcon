"""Generate and audit the Branch C recombinable-attribute benchmark scaffold."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from attcon.binding import BindingConfig, generate_binding_examples, validate_binding_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Branch C binding dataset scaffold.")
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="audits/branch_c_binding_dataset.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BindingConfig()
    examples = generate_binding_examples(args.examples, config=config, seed=args.seed)
    failures = [
        {"example_id": example.example_id, "failures": found}
        for example in examples
        if (found := validate_binding_example(example))
    ]
    split_counts = Counter(example.split for example in examples)
    target_conjunctions = {example.target.conjunction() for example in examples}
    feature_names = ("location", "visible_type", "digit", "cue_tag", "inspected")
    train_target_values = {
        field: {
            getattr(example.target, field)
            for example in examples
            if example.split == "train"
        }
        for field in feature_names
    }
    heldout_target_values = {
        field: {
            getattr(example.target, field)
            for example in examples
            if example.split == "heldout_conjunction"
        }
        for field in feature_names
    }
    shared_feature_vocabulary = all(
        heldout_target_values[field] <= train_target_values[field]
        for field in feature_names
    )
    payload = {
        "status": "complete" if not failures else "invalid",
        "scope": "benchmark_scaffold_only_no_binding_claim",
        "seed": args.seed,
        "config": {
            "grid_size": config.grid_size,
            "num_objects": config.num_objects,
            "num_visible_types": config.num_visible_types,
            "digit_vocab_size": config.digit_vocab_size,
            "num_cues": config.num_cues,
            "heldout_modulus": config.heldout_modulus,
        },
        "num_examples": len(examples),
        "split_counts": dict(split_counts),
        "unique_target_conjunctions": len(target_conjunctions),
        "heldout_individual_feature_vocabulary_present_in_train": shared_feature_vocabulary,
        "target_feature_cardinality": {
            field: {
                "train": len(train_target_values[field]),
                "heldout": len(heldout_target_values[field]),
            }
            for field in feature_names
        },
        "false_binding_lures_valid": not failures,
        "invariant_failures": failures,
        "sample_train": next(
            example.to_dict() for example in examples if example.split == "train"
        ),
        "sample_heldout": next(
            example.to_dict() for example in examples if example.split == "heldout_conjunction"
        ),
        "next_required_experiment": (
            "Train an integrated binding state and compare its held-out conjunction/lure accuracy "
            "with matched-capacity independent feature probes before changing Branch C support."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
