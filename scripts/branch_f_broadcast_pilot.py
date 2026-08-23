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

from attcon.broadcast import BroadcastConfig, generate_broadcast_examples
from attcon.broadcast_experiment import (
    BroadcastConsumerModel,
    broadcast_intervention_metrics,
    broadcast_metrics,
    parameter_count,
    tensorize_broadcast_examples,
    train_broadcast_model,
)


THRESHOLDS = {
    "shared_broad_joint_accuracy": 0.85,
    "shared_onset_accuracy": 0.90,
    "shared_onset_alignment": 0.95,
    "private_broad_joint_accuracy": 0.85,
    "shared_ablation_accuracy_drop": 0.75,
    "private_single_route_accuracy_drop_ceiling": 0.25,
    "coordinated_ablation_advantage": 0.60,
    "content_swap_follow_rate": 0.80,
    "local_action_invariance": 0.95,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=16380)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--model-seed", type=int, default=107)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="audits/branch_f_broadcast_pilot.json")
    args = parser.parse_args()

    config = BroadcastConfig()
    examples = generate_broadcast_examples(
        args.count, config=config, seed=args.data_seed
    )
    train = tensorize_broadcast_examples(
        [example for example in examples if example.split == "train"], config
    )
    heldout = tensorize_broadcast_examples(
        [example for example in examples if example.split == "heldout_content_strength"],
        config,
    )
    torch.manual_seed(args.model_seed)
    shared = BroadcastConsumerModel(config, args.hidden_size, shared=True)
    torch.manual_seed(args.model_seed)
    private = BroadcastConsumerModel(config, args.hidden_size, shared=False)
    shared_losses = train_broadcast_model(
        shared,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.model_seed,
        device=args.device,
    )
    private_losses = train_broadcast_model(
        private,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.model_seed,
        device=args.device,
    )
    shared_eval = broadcast_metrics(shared, heldout, device=args.device)
    private_eval = broadcast_metrics(private, heldout, device=args.device)
    interventions = broadcast_intervention_metrics(
        shared, private, heldout, device=args.device
    )
    observed = {
        "shared_broad_joint_accuracy": shared_eval["broad_joint_accuracy"],
        "shared_onset_accuracy": shared_eval["onset_accuracy"],
        "shared_onset_alignment": shared_eval["onset_alignment_rate"],
        "private_broad_joint_accuracy": private_eval["broad_joint_accuracy"],
        "shared_ablation_accuracy_drop": interventions[
            "shared_broad_accuracy_drop_after_zero"
        ],
        "private_single_route_accuracy_drop_ceiling": interventions[
            "private_mean_broad_accuracy_drop_after_one_route_zero"
        ],
        "coordinated_ablation_advantage": interventions[
            "coordinated_ablation_drop_advantage"
        ],
        "content_swap_follow_rate": interventions[
            "shared_content_swap_broad_follow_rate"
        ],
        "local_action_invariance": interventions[
            "local_action_invariance_under_shared_swap"
        ],
    }
    gates = {
        name: (
            observed[name] <= threshold
            if name == "private_single_route_accuracy_drop_ceiling"
            else observed[name] >= threshold
        )
        for name, threshold in THRESHOLDS.items()
    }
    result = {
        "audit": "branch_f_broadcast_pilot",
        "status": "engineering_support" if all(gates.values()) else "unsupported_pilot",
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
        "shared_model": {
            "parameter_count": parameter_count(shared),
            "initial_epoch_loss": shared_losses[0],
            "final_epoch_loss": shared_losses[-1],
            "heldout": shared_eval,
        },
        "private_shortcut_upper_bound": {
            "parameter_count": parameter_count(private),
            "parameters_exactly_matched": parameter_count(shared)
            == parameter_count(private),
            "receives_private_target_shortcuts": True,
            "initial_epoch_loss": private_losses[0],
            "final_epoch_loss": private_losses[-1],
            "heldout": private_eval,
        },
        "interventions": interventions,
        "observed_gate_metrics": observed,
        "predeclared_thresholds": THRESHOLDS,
        "gates": gates,
        "all_engineering_gates_pass": all(gates.values()),
        "branch_f_supported": False,
        "support_boundary": (
            "The benchmark directly supervises all consumer outputs and the shared architecture "
            "imposes a broadcast bottleneck. A pass is engineering support only; spontaneous "
            "broadcast requires emergence without shared-state or ignition supervision."
        ),
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
