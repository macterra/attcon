from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from attcon.binding import BindingConfig, generate_binding_examples
from attcon.binding_experiment import (
    IndependentFeatureBaseline,
    SharedSelectionBindingModel,
    binding_intervention_metrics,
    evaluate_binding_model,
    parameter_count,
    tensorize_binding_examples,
    train_binding_model,
    wilson_lower_bound,
)


THRESHOLDS = {
    "integrated_heldout_joint_accuracy": 0.75,
    "heldout_joint_advantage": 0.25,
    "lure_rejection_advantage": 0.15,
    "target_type_follow_rate": 0.75,
    "target_other_field_joint_stability": 0.90,
    "non_target_all_field_invariance": 0.90,
}

VARIANTS = {
    "original": {
        "config": BindingConfig(),
        "surface_schema": {
            "location": "grid_cell",
            "visible_type": "visual_category",
            "digit": "glyph_digit",
            "cue_tag": "cue_category",
            "inspected": "inspection_status",
        },
    },
    "surface_v2": {
        "config": BindingConfig(
            grid_size=4,
            num_objects=10,
            num_visible_types=6,
            digit_vocab_size=7,
            num_cues=5,
            heldout_modulus=4,
        ),
        "surface_schema": {
            "location": "tile_slot",
            "visible_type": "geometric_shape",
            "digit": "texture_pattern",
            "cue_tag": "color_key",
            "inspected": "illumination_flag",
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="original")
    parser.add_argument("--count", type=int, default=16384)
    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--model-seed", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_c_binding_pilot.json")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    config = variant["config"]
    examples = generate_binding_examples(args.count, config=config, seed=args.data_seed)
    train_examples = [example for example in examples if example.split == "train"]
    heldout_examples = [
        example for example in examples if example.split == "heldout_conjunction"
    ]
    train = tensorize_binding_examples(train_examples, config)
    heldout = tensorize_binding_examples(heldout_examples, config)

    torch.manual_seed(args.model_seed)
    integrated = SharedSelectionBindingModel(config, args.hidden_size)
    torch.manual_seed(args.model_seed)
    independent = IndependentFeatureBaseline(config, args.hidden_size)
    integrated_losses = train_binding_model(
        integrated,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.model_seed,
        device=args.device,
    )
    independent_losses = train_binding_model(
        independent,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.model_seed,
        device=args.device,
    )
    integrated_eval = evaluate_binding_model(integrated, heldout, device=args.device)
    independent_eval = evaluate_binding_model(independent, heldout, device=args.device)
    interventions = binding_intervention_metrics(
        integrated, heldout, config, device=args.device
    )
    comparisons = {
        "heldout_joint_advantage": integrated_eval["joint_accuracy"]
        - independent_eval["joint_accuracy"],
        "lure_rejection_advantage": integrated_eval["false_binding_lure_rejection"]
        - independent_eval["false_binding_lure_rejection"],
    }
    observed = {
        "integrated_heldout_joint_accuracy": integrated_eval["joint_accuracy"],
        **comparisons,
        **interventions,
    }
    gates = {
        name: observed[name] >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    result = {
        "audit": "branch_c_binding_pilot",
        "variant": args.variant,
        "surface_schema": variant["surface_schema"],
        "status": "supported_pilot" if all(gates.values()) else "unsupported_pilot",
        "scope": (
            "single synthetic surface-attribute benchmark; a second variant is required "
            "for Branch C support"
        ),
        "config": {
            **asdict(config),
            "count": args.count,
            "train_count": len(train),
            "heldout_count": len(heldout),
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "device": args.device,
        },
        "models": {
            "integrated": {
                "architecture": "one shared soft object selection feeding all four heads",
                "parameter_count": parameter_count(integrated),
                "initial_epoch_loss": integrated_losses[0],
                "final_epoch_loss": integrated_losses[-1],
                "heldout": integrated_eval,
            },
            "independent_feature_baseline": {
                "architecture": "object-identity-destroying mean pool feeding four heads",
                "parameter_count": parameter_count(independent),
                "initial_epoch_loss": independent_losses[0],
                "final_epoch_loss": independent_losses[-1],
                "heldout": independent_eval,
            },
            "baseline_to_integrated_parameter_ratio": parameter_count(independent)
            / parameter_count(integrated),
        },
        "comparisons": comparisons,
        "integrated_interventions": interventions,
        "uncertainty": {
            "integrated_joint_wilson_95_lower": wilson_lower_bound(
                integrated_eval["joint_accuracy"], len(heldout)
            ),
            "integrated_lure_wilson_95_lower": wilson_lower_bound(
                integrated_eval["false_binding_lure_rejection"], len(heldout)
            ),
        },
        "predeclared_thresholds": THRESHOLDS,
        "gates": gates,
        "all_pilot_gates_pass": all(gates.values()),
        "branch_c_supported": False,
        "branch_c_support_blocker": (
            "Even a passing pilot requires replication on a benchmark variant with "
            "different surface attributes."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
