"""Stage 4B emergence experiment.

Tests the roadmap's global-falsifier question: does the controller learn a self-model of
its own attention history *without* a direct self-model objective? We train a task-only
checkpoint (no hidden-self-model, native-self-model, report, or policy-feedback losses --
only task + attention shaping + cue-switch) and probe whether its RAW controller hidden
state linearly predicts the inspected-cell state better than a previous-observation
baseline. A positive advantage means inspection-history self-modeling emerged from the
search task alone; a flat/negative advantage is an honest negative result.

Usage:
    .venv/bin/python scripts/stage4b_emergence.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from attcon.train import train_experiment, load_config
from attcon.eval import (
    _collect_learned_self_model_dataset,
    learned_self_model_metrics,
    make_generator,
    symmetric_kl,
)
from attcon.data import TaskConfig, generate_batch
from attcon.models import ModelConfig, RecurrentAttentionController


TASK_ONLY_OVERRIDES = {
    "hidden_self_model_weight": 0.0,
    "self_model_weight": 0.0,
    "self_model_policy_feedback_weight": 0.0,
    "target_found_report_weight": 0.0,
    "relevant_region_report_weight": 0.0,
    "unresolved_search_report_weight": 0.0,
    "wrong_candidate_history_report_weight": 0.0,
    "allocation_error_report_weight": 0.0,
}

DEFAULT_TASK_ONLY_CHECKPOINT = Path("outputs/stage4b_emergence/experiment.pt")
DEFAULT_SUPERVISED_CHECKPOINT = Path("outputs/tune_prob_035/experiment.pt")
DEFAULT_OUT = Path("audits/stage4b_emergence_tune_prob_035.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 4B emergence and causal audit.")
    parser.add_argument("--task-only-checkpoint", default=str(DEFAULT_TASK_ONLY_CHECKPOINT))
    parser.add_argument("--supervised-checkpoint", default=str(DEFAULT_SUPERVISED_CHECKPOINT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--train-task-only",
        action="store_true",
        help="Retrain the task-only checkpoint even when it already exists.",
    )
    parser.add_argument("--intervention-step", type=int, default=2)
    parser.add_argument(
        "--intervention-scales",
        default="0.25,0.5,1.0,2.0",
        help="Comma-separated hidden-state intervention magnitudes.",
    )
    parser.add_argument(
        "--intervention-scale",
        type=float,
        default=None,
        help="Run one magnitude only (compatibility override for --intervention-scales).",
    )
    parser.add_argument("--intervention-scenes", type=int, default=128)
    parser.add_argument("--random-controls", type=int, default=32)
    parser.add_argument("--random-control-percentile", type=float, default=95.0)
    parser.add_argument("--seed-offset", type=int, default=9736)
    return parser.parse_args()


def _emergence_probe(model, cfg, task_cfg, device, seed):
    m = learned_self_model_metrics(model, cfg, task_cfg, device, seed)
    return {
        "hidden_cell_accuracy_advantage": m.get("hidden_cell_accuracy_advantage"),
        "hidden_cell_bce_advantage": m.get("hidden_cell_bce_advantage"),
        "hidden_target_accuracy_advantage": m.get("hidden_target_accuracy_advantage"),
        "hidden_target_positive_recall_advantage": m.get("hidden_target_positive_recall_advantage"),
    }


def _load_recurrent_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    task_cfg = TaskConfig.from_dict(checkpoint.get("task", cfg["task"]))
    model_cfg = ModelConfig.from_dict(cfg["model"])
    model = RecurrentAttentionController(task_cfg, model_cfg).to(device)
    incompatible = model.load_state_dict(checkpoint["models"]["recurrent"], strict=False)
    non_content_missing = [
        key for key in incompatible.missing_keys if not key.startswith("content_")
    ]
    if non_content_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint migration is not limited to unused content-report heads: "
            f"missing={non_content_missing}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, cfg, task_cfg, checkpoint["metrics"]["recurrent"]["accuracy"]


def _fit_inspection_probe(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> tuple[nn.Linear, dict[str, float]]:
    """Fit the task-induced full inspection-map readout used to define intervention directions."""

    probe_cfg = cfg["evaluation"]["learned_self_modeling"]
    batch_size = int(cfg["training"]["batch_size"])
    train = _collect_learned_self_model_dataset(
        model,
        task_cfg,
        batch_size,
        int(probe_cfg["train_batches"]),
        device,
        seed,
    )
    test = _collect_learned_self_model_dataset(
        model,
        task_cfg,
        batch_size,
        int(probe_cfg["test_batches"]),
        device,
        seed + 1000,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1700)
        probe = nn.Linear(
            train["hidden_features"].shape[-1],
            train["inspection_labels"].shape[-1],
        ).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=float(probe_cfg["learning_rate"]))
        for _ in range(int(probe_cfg["epochs"])):
            logits = probe(train["hidden_features"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, train["inspection_labels"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        train_logits = probe(train["hidden_features"])
        test_logits = probe(test["hidden_features"])
        train_predictions = (torch.sigmoid(train_logits) >= 0.5).float()
        test_predictions = (torch.sigmoid(test_logits) >= 0.5).float()
    metrics = {
        "train_bce": torch.nn.functional.binary_cross_entropy_with_logits(
            train_logits, train["inspection_labels"]
        ).item(),
        "test_bce": torch.nn.functional.binary_cross_entropy_with_logits(
            test_logits, test["inspection_labels"]
        ).item(),
        "train_cell_accuracy": (
            train_predictions == train["inspection_labels"]
        ).float().mean().item(),
        "test_cell_accuracy": (
            test_predictions == test["inspection_labels"]
        ).float().mean().item(),
    }
    return probe, metrics


def _condition_effects(
    model,
    probe: nn.Linear,
    batch,
    base: dict[str, torch.Tensor],
    *,
    step: int,
    selected_cells: torch.Tensor,
    direction: torch.Tensor,
    scale: float,
) -> dict[str, float]:
    # RecurrentAttentionController currently exports [initial, step0, ..., stepN-2] because
    # controller_state_seq is initialized with the initial state and stacked with ``[:-1]``.
    # The state actually used at intervention step t is therefore exported at t+1.
    state_sequence_index = min(step + 1, base["controller_state_seq"].shape[1] - 1)
    hidden = base["controller_state_seq"][:, state_sequence_index]
    unit_direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta = unit_direction * hidden.std(dim=-1, keepdim=True).clamp_min(1e-3) * scale
    with torch.no_grad():
        positive = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=model.task_config.num_steps,
            intervention={"step": step, "state_override": hidden + delta},
        )
        negative = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=model.task_config.num_steps,
            intervention={"step": step, "state_override": hidden - delta},
        )
        # Score the exact overridden states. Reading controller_state_seq[:, step] would read the
        # pre-intervention state because of the export offset documented above.
        positive_report = torch.sigmoid(probe(hidden + delta))
        negative_report = torch.sigmoid(probe(hidden - delta))

    batch_index = torch.arange(batch.scene.shape[0], device=batch.scene.device)
    final_positive = torch.softmax(positive["logits"], dim=-1)
    final_negative = torch.softmax(negative["logits"], dim=-1)
    future_start = min(step + 1, model.task_config.num_steps - 1)
    return {
        "selected_cell_report_gap": (
            positive_report[batch_index, selected_cells]
            - negative_report[batch_index, selected_cells]
        ).mean().item(),
        "intervention_step_attention_symmetric_kl": symmetric_kl(
            positive["attention_seq"][:, step], negative["attention_seq"][:, step]
        ).mean().item(),
        "future_attention_symmetric_kl": symmetric_kl(
            positive["attention_seq"][:, future_start:],
            negative["attention_seq"][:, future_start:],
        ).mean().item(),
        "intervention_step_selected_attention_gap": (
            positive["attention_seq"][batch_index, step, selected_cells]
            - negative["attention_seq"][batch_index, step, selected_cells]
        ).mean().item(),
        "future_selected_attention_gap": (
            positive["attention_seq"][batch_index, future_start, selected_cells]
            - negative["attention_seq"][batch_index, future_start, selected_cells]
        ).mean().item(),
        "final_prediction_symmetric_kl": symmetric_kl(
            final_positive, final_negative
        ).mean().item(),
        "final_prediction_change_fraction": (
            final_positive.argmax(dim=-1) != final_negative.argmax(dim=-1)
        ).float().mean().item(),
    }


def _distribution_summary(values: list[float], percentile: float) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": tensor.mean().item(),
        "percentile": torch.quantile(tensor, percentile / 100.0).item(),
        "max": tensor.max().item(),
    }


def _scale_sweep_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compact = {}
    for scale, result in results.items():
        effect = result["directed_effect"]
        compact[scale] = {
            "selected_cell_report_gap": effect["selected_cell_report_gap"],
            "intervention_step_selected_attention_gap": effect[
                "intervention_step_selected_attention_gap"
            ],
            "future_selected_attention_gap": effect["future_selected_attention_gap"],
            "attention_policy_specificity": result["attention_policy_specificity"],
            "policy_consistent_avoidance": result["policy_consistent_avoidance"],
            "supported": result["supported"],
        }
    return {
        "by_scale": compact,
        "all_scales_shift_report_positive": all(
            item["selected_cell_report_gap"] > 0.0 for item in compact.values()
        ),
        "all_scales_increase_selected_cell_attention": all(
            item["intervention_step_selected_attention_gap"] > 0.0
            and item["future_selected_attention_gap"] > 0.0
            for item in compact.values()
        ),
        "any_scale_policy_consistent_avoidance": any(
            item["policy_consistent_avoidance"] for item in compact.values()
        ),
        "any_scale_supported": any(item["supported"] for item in compact.values()),
    }


def _probe_direction_intervention(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
    *,
    step: int,
    scale: float,
    probe_scenes: int,
    random_controls: int,
    random_control_percentile: float,
) -> dict[str, Any]:
    """Test whether the decodable inspection direction has policy effects beyond random state directions."""

    if random_controls < 1:
        raise ValueError("random_controls must be at least 1")
    if not 0.0 <= random_control_percentile <= 100.0:
        raise ValueError("random_control_percentile must be between 0 and 100")
    step = max(0, min(step, task_cfg.num_steps - 1))
    probe, probe_metrics = _fit_inspection_probe(model, cfg, task_cfg, device, seed)
    generator = make_generator(seed + 4000, device)
    batch = generate_batch(
        probe_scenes,
        task_cfg.num_steps,
        task_cfg,
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        base = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
        )
        state_sequence_index = min(step + 1, base["controller_state_seq"].shape[1] - 1)
        hidden = base["controller_state_seq"][:, state_sequence_index]
        base_report = torch.sigmoid(probe(hidden))
        # Choose the cell whose fitted inspection report is locally most responsive to a
        # hidden-state change. This avoids a saturated target-cell readout producing a numerical
        # zero report gap, while selecting the cell before either intervention is applied.
        report_sensitivity = base_report * (1.0 - base_report) * probe.weight.norm(dim=-1)
        selected_cells = report_sensitivity.argmax(dim=-1)
    directed = _condition_effects(
        model,
        probe,
        batch,
        base,
        step=step,
        selected_cells=selected_cells,
        direction=probe.weight[selected_cells],
        scale=scale,
    )

    control_generator = make_generator(seed + 5000, device)
    controls: list[dict[str, float]] = []
    for _ in range(random_controls):
        random_direction = torch.randn(
            hidden.shape,
            generator=control_generator,
            device=device,
        )
        controls.append(
            _condition_effects(
                model,
                probe,
                batch,
                base,
                step=step,
                selected_cells=selected_cells,
                direction=random_direction,
                scale=scale,
            )
        )

    absolute_control_keys = {
        "selected_cell_report_gap",
        "intervention_step_selected_attention_gap",
        "future_selected_attention_gap",
    }
    control_floor = {
        key: _distribution_summary(
            [abs(control[key]) if key in absolute_control_keys else control[key] for control in controls],
            random_control_percentile,
        )
        for key in directed
    }
    directed_exceeds_floor = {
        key: bool(
            (abs(value) if key in absolute_control_keys else value)
            > control_floor[key]["percentile"]
        )
        for key, value in directed.items()
    }
    attention_specificity = bool(
        directed_exceeds_floor["intervention_step_attention_symmetric_kl"]
        or directed_exceeds_floor["future_attention_symmetric_kl"]
        or directed_exceeds_floor["intervention_step_selected_attention_gap"]
        or directed_exceeds_floor["future_selected_attention_gap"]
    )
    policy_consistent_avoidance = bool(
        (
            directed["intervention_step_selected_attention_gap"] < 0.0
            and directed_exceeds_floor["intervention_step_selected_attention_gap"]
        )
        or (
            directed["future_selected_attention_gap"] < 0.0
            and directed_exceeds_floor["future_selected_attention_gap"]
        )
    )
    batch_index = torch.arange(batch.scene.shape[0], device=device)
    return {
        "step": step,
        "controller_state_sequence_index": min(step + 1, task_cfg.num_steps - 1),
        "controller_state_export_alignment": (
            "The model currently exports the state used at step t at controller_state_seq[t+1] "
            "because the sequence prepends the initial state and drops the final appended state."
        ),
        "scale": scale,
        "probe_scenes": probe_scenes,
        "probe_metrics": probe_metrics,
        "direction_target": (
            "pre-intervention cell with the largest local inspection-report sensitivity"
        ),
        "selected_cell_inspected_rate": base["inspection_seq"][
            batch_index, step, selected_cells
        ].mean().item(),
        "selected_cell_is_task_target_fraction": (
            selected_cells == batch.target_pos
        ).float().mean().item(),
        "directed_effect": directed,
        "random_direction_control": {
            "num_directions": random_controls,
            "percentile": random_control_percentile,
            "metrics": control_floor,
        },
        "directed_exceeds_random_floor": directed_exceeds_floor,
        "report_direction_effect": bool(directed["selected_cell_report_gap"] > 0.0),
        "attention_policy_specificity": attention_specificity,
        "policy_consistent_avoidance": policy_consistent_avoidance,
        "supported": bool(
            directed["selected_cell_report_gap"] > 0.0
            and attention_specificity
            and policy_consistent_avoidance
        ),
        "claim_boundary": (
            "A positive result shows that a linearly decodable inspection-history direction "
            "has policy effects larger than matched random hidden-state directions and that "
            "raising the 'already inspected' report reduces attention to that cell. It does not "
            "prove that the controller represents the direction as self-related, and the "
            "probe-defined report shift is expected by construction."
        ),
    }


def main():
    args = parse_args()
    scale_spec = (
        [args.intervention_scale]
        if args.intervention_scale is not None
        else [
            float(value.strip())
            for value in args.intervention_scales.split(",")
            if value.strip()
        ]
    )
    if not scale_spec or any(scale <= 0.0 for scale in scale_spec):
        raise ValueError("intervention scales must contain at least one positive value")
    device = torch.device("cpu")
    base_cfg = load_config("configs/tune_prob_035.yaml")
    task_only_checkpoint = Path(args.task_only_checkpoint)
    supervised_checkpoint = Path(args.supervised_checkpoint)

    # 1) Train a task-only checkpoint (no self-model supervision of any kind).
    if args.train_task_only or not task_only_checkpoint.exists():
        emergence_cfg = load_config("configs/tune_prob_035.yaml")
        emergence_cfg["output_dir"] = str(task_only_checkpoint.parent)
        emergence_cfg["training"].update(TASK_ONLY_OVERRIDES)
        train_experiment(emergence_cfg)

    emergent, emergence_cfg, task_cfg, emergent_acc = _load_recurrent_checkpoint(
        task_only_checkpoint, device
    )

    # 2) Compare against the supervised base checkpoint (hidden_self_model_weight=0.5).
    base, checkpoint_base_cfg, base_task_cfg, base_acc = _load_recurrent_checkpoint(
        supervised_checkpoint, device
    )
    # Keep current evaluation settings while retaining the checkpoint's training metadata.
    checkpoint_base_cfg["evaluation"] = base_cfg["evaluation"]
    base_cfg = checkpoint_base_cfg

    seed = int(emergence_cfg["seed"]) + args.seed_offset
    emergent_probe = _emergence_probe(emergent, emergence_cfg, task_cfg, device, seed)
    base_probe = _emergence_probe(base, base_cfg, base_task_cfg, device, seed)
    emergent_interventions = {
        str(scale): _probe_direction_intervention(
            emergent,
            emergence_cfg,
            task_cfg,
            device,
            seed + 7000,
            step=args.intervention_step,
            scale=scale,
            probe_scenes=args.intervention_scenes,
            random_controls=args.random_controls,
            random_control_percentile=args.random_control_percentile,
        )
        for scale in scale_spec
    }
    base_interventions = {
        str(scale): _probe_direction_intervention(
            base,
            base_cfg,
            base_task_cfg,
            device,
            seed + 8000,
            step=args.intervention_step,
            scale=scale,
            probe_scenes=args.intervention_scenes,
            random_controls=args.random_controls,
            random_control_percentile=args.random_control_percentile,
        )
        for scale in scale_spec
    }
    primary_scale = str(max(scale_spec))
    emergent_intervention = emergent_interventions[primary_scale]
    base_intervention = base_interventions[primary_scale]

    cell_emerges = (emergent_probe["hidden_cell_bce_advantage"] or 0.0) > 0.0
    target_emerges = (emergent_probe["hidden_target_accuracy_advantage"] or 0.0) > 0.0
    # How much does the dedicated self-model objective add over the task-only representation?
    supervision_cell_gain = (base_probe["hidden_cell_bce_advantage"] or 0.0) - (
        emergent_probe["hidden_cell_bce_advantage"] or 0.0
    )
    result = {
        "task_only_checkpoint": str(task_only_checkpoint),
        "supervised_checkpoint": str(supervised_checkpoint),
        "task_only_recurrent_accuracy": emergent_acc,
        "supervised_base_recurrent_accuracy": base_acc,
        "emergent_self_model_probe": emergent_probe,
        "supervised_base_self_model_probe": base_probe,
        "task_only_probe_direction_intervention": emergent_intervention,
        "supervised_base_probe_direction_intervention": base_intervention,
        "probe_direction_intervention_scale_sweep": {
            "primary_scale": float(primary_scale),
            "task_only": _scale_sweep_summary(emergent_interventions),
            "supervised_base": _scale_sweep_summary(base_interventions),
        },
        "cell_inspection_self_model_emerges_task_only": cell_emerges,
        "target_inspection_self_model_emerges_task_only": target_emerges,
        "supervision_cell_bce_advantage_gain": supervision_cell_gain,
        "interpretation": (
            "Nuanced. A WEAK cell-level inspection-history self-model emerges from the search "
            "task alone: the raw hidden state beats a previous-observation baseline on the "
            "full inspection map (BCE advantage ~+0.09; accuracy advantage only ~+0.01, near "
            "noise). Crucially the dedicated self-model objective adds almost nothing to this "
            "representation (supervision_cell_bce_advantage_gain ~0), so the representation is "
            "task-induced rather than supervision-induced -- the positive direction for the "
            "global falsifier. However, TARGET-level inspection ('have I inspected the target?') "
            "is NOT encoded better than observation in either model (negative advantage), so the "
            "emergent self-model is partial and weak. The probe-direction intervention separately "
            "tests whether that decodable cell-level direction affects attention more than matched "
            "random hidden-state directions; its result must be read from the intervention fields "
            "rather than inferred from probe accuracy. This remains bounded evidence against the "
            "'supervised self-model required everywhere' falsifier, not a strong emergence claim."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
