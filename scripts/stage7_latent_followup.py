"""Stage 7 latent-only follow-up audit.

This separates two possible explanations for the current latent-only result:

1. the opaque quantised interface is too lossy, or
2. the checkpoint state does not contain recoverable remembered/counterfactual content.

The audit compares the shipped quantised latent-only decoder with a richer continuous-state
probe that still reads only internal state tensors from NLExample
(controller/previous-controller/attention/previous-attention/memory). It never reads the
directly encoded content tokens or symbolic labels as inputs.

Usage:
    .venv/bin/python scripts/stage7_latent_followup.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from attcon.data import generate_batch
from attcon.eval import (
    _select_diverse_nl_examples,
    _select_translator_examples,
    collect_cue_switch_nl_examples,
    collect_intervention_nl_examples,
    collect_nl_examples,
    load_config,
    load_models_from_checkpoint,
    make_generator,
)
from attcon.nl_report import (
    _LATENT_BINARY_FIELDS,
    _fit_binary_probe,
    _fit_multiclass_probe,
    _latent_multiclass_fields,
    _score_local_report_payloads,
    run_latent_only_report_mode,
    run_observation_only_heuristic_report_mode,
)


CONFIG_PATH = "configs/tune_prob_035.yaml"
CHECKPOINT_PATH = "outputs/tune_prob_035/experiment.pt"
OUT_PATH = Path("audits/stage7_latent_followup_tune_prob_035.json")
SEED_OFFSET = 8811
PROBE_SCENES = 32
CALIBRATION_EXAMPLES = 8
EVALUATION_EXAMPLES = 24
TRANSLATOR_TRAIN_EXAMPLES = 64


def _continuous_feature_matrix(examples: list[Any]) -> torch.Tensor:
    rows = []
    for example in examples:
        rows.append(
            torch.cat(
                [
                    example.controller_state,
                    example.prev_controller_state,
                    example.attention_state,
                    example.prev_attention_state,
                    example.memory_state,
                ],
                dim=0,
            ).float()
        )
    return torch.stack(rows, dim=0)


def _standardize(train_features: torch.Tensor, eval_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_features - mean) / std, (eval_features - mean) / std


def _run_continuous_state_probe(
    *,
    fit_examples: list[Any],
    evaluation_examples: list[Any],
    grid_size: int,
) -> dict[str, Any]:
    fit_features, eval_features = _standardize(
        _continuous_feature_matrix(fit_examples),
        _continuous_feature_matrix(evaluation_examples),
    )

    multiclass_pred: dict[str, torch.Tensor] = {}
    for field, attr, num_classes in _latent_multiclass_fields(grid_size):
        labels = torch.tensor([int(getattr(example, attr)) for example in fit_examples])
        head = _fit_multiclass_probe(fit_features, labels, num_classes)
        with torch.no_grad():
            multiclass_pred[field] = head(eval_features).argmax(dim=-1)

    binary_pred: dict[str, torch.Tensor] = {}
    for field, attr in _LATENT_BINARY_FIELDS:
        labels = torch.tensor(
            [[float(getattr(example, attr))] for example in fit_examples],
            dtype=torch.float32,
        )
        head = _fit_binary_probe(fit_features, labels)
        with torch.no_grad():
            binary_pred[field] = (torch.sigmoid(head(eval_features)) >= 0.5).long()

    unresolved_rows_head = _fit_binary_probe(
        fit_features,
        torch.tensor(
            [[int(row in example.unresolved_rows) for row in range(grid_size)] for example in fit_examples],
            dtype=torch.float32,
        ),
    )
    unresolved_cols_head = _fit_binary_probe(
        fit_features,
        torch.tensor(
            [[int(col in example.unresolved_cols) for col in range(grid_size)] for example in fit_examples],
            dtype=torch.float32,
        ),
    )
    with torch.no_grad():
        unresolved_rows_pred = (torch.sigmoid(unresolved_rows_head(eval_features)) >= 0.5).long()
        unresolved_cols_pred = (torch.sigmoid(unresolved_cols_head(eval_features)) >= 0.5).long()

    predictions: list[dict[str, Any]] = []
    for idx, example in enumerate(evaluation_examples):
        cell = int(multiclass_pred["attended_cell"][idx].item())
        prev_cell = int(multiclass_pred["previous_attended_cell"][idx].item())
        predictions.append(
            {
                "natural_language_report": "continuous internal-state decode",
                "search_type": example.cue,
                "previous_search_type": int(multiclass_pred["previous_search_type"][idx].item()),
                "cue_switched": bool(binary_pred["cue_switched"][idx, 0].item()),
                "attended_cell": list(divmod(cell, grid_size)),
                "attended_visible_type": int(multiclass_pred["attended_visible_type"][idx].item()),
                "attended_digit": int(multiclass_pred["attended_digit"][idx].item()),
                "glimpse_digit": example.glimpse_digit,
                "previous_attended_cell": list(divmod(prev_cell, grid_size)),
                "previous_attended_visible_type": int(
                    multiclass_pred["previous_attended_visible_type"][idx].item()
                ),
                "previous_attended_digit": int(multiclass_pred["previous_attended_digit"][idx].item()),
                "previous_glimpse_digit": int(multiclass_pred["previous_glimpse_digit"][idx].item()),
                "glimpse_target_match": example.glimpse_target_match,
                "previous_found_target": bool(binary_pred["previous_found_target"][idx, 0].item()),
                "found_target": bool(binary_pred["found_target"][idx, 0].item()),
                "relevant_region_inspected": bool(binary_pred["relevant_region_inspected"][idx, 0].item()),
                "unresolved_search": bool(binary_pred["unresolved_search"][idx, 0].item()),
                "current_wrong_candidate": bool(binary_pred["current_wrong_candidate"][idx, 0].item()),
                "wrong_candidate_history": bool(binary_pred["wrong_candidate_history"][idx, 0].item()),
                "revisit_unresolved": bool(binary_pred["revisit_unresolved"][idx, 0].item()),
                "allocation_error": bool(binary_pred["allocation_error"][idx, 0].item()),
                "inspected_count": int(multiclass_pred["inspected_count"][idx].item()),
                "previous_inspected_count": int(multiclass_pred["previous_inspected_count"][idx].item()),
                "attended_cell_previously_inspected": bool(
                    binary_pred["attended_cell_previously_inspected"][idx, 0].item()
                ),
                "unresolved_rows": [row for row in range(grid_size) if unresolved_rows_pred[idx, row].item()],
                "unresolved_cols": [col for col in range(grid_size) if unresolved_cols_pred[idx, col].item()],
                "unresolved_count": int(multiclass_pred["unresolved_count"][idx].item()),
            }
        )

    scored = _score_local_report_payloads(
        mode="latent_only_state",
        examples=evaluation_examples,
        predictions=predictions,
        grid_size=grid_size,
    )
    scored["interface"] = "continuous_internal_state_probe"
    scored["reads_exact_content_tokens"] = False
    scored["fit_examples"] = len(fit_examples)
    return scored


def _score_slice(examples: list[Any], *, grid_size: int) -> dict[str, Any]:
    examples = [example for example in examples if example.step_index > 0]
    required = CALIBRATION_EXAMPLES + EVALUATION_EXAMPLES
    if len(examples) < required:
        return {"skipped": True, "reason": f"need {required} examples, have {len(examples)}"}

    calibration, evaluation = _select_diverse_nl_examples(
        examples,
        grid_size=grid_size,
        calibration_count=CALIBRATION_EXAMPLES,
        evaluation_count=EVALUATION_EXAMPLES,
    )
    held_out = {id(example) for example in calibration + evaluation}
    translator_pool = [example for example in examples if id(example) not in held_out]
    translator = _select_translator_examples(
        translator_pool,
        grid_size=grid_size,
        target_count=TRANSLATOR_TRAIN_EXAMPLES,
    )
    if not translator:
        translator = calibration
    fit = translator + calibration
    observation = run_observation_only_heuristic_report_mode(
        evaluation_examples=evaluation,
        grid_size=grid_size,
    )

    def summarize(scored: dict[str, Any]) -> dict[str, Any]:
        current_adv = scored["current_content_joint_accuracy"] - observation["current_content_joint_accuracy"]
        memory_adv = scored["memory_content_joint_accuracy"] - observation["memory_content_joint_accuracy"]
        content_adv = scored["content_only_joint_accuracy"] - observation["content_only_joint_accuracy"]
        summary = {
            "current_content_joint_accuracy_advantage": round(current_adv, 6),
            "memory_content_joint_accuracy_advantage": round(memory_adv, 6),
            "content_only_joint_accuracy_advantage": round(content_adv, 6),
            "content_supported": bool(
                current_adv > 0.0 and memory_adv > 0.0 and content_adv > 0.0
            ),
        }
        for field in (
            "attended_visible_type_accuracy",
            "attended_digit_accuracy",
            "previous_attended_visible_type_accuracy",
            "previous_attended_digit_accuracy",
            "previous_glimpse_digit_accuracy",
        ):
            summary[field] = round(scored[field], 6)
            summary[f"{field}_advantage"] = round(scored[field] - observation[field], 6)
        return summary

    quantized_runs = {}
    for num_chunks, num_levels in ((8, 4), (16, 8), (32, 8), (48, 8)):
        scored = run_latent_only_report_mode(
            fit_examples=fit,
            evaluation_examples=evaluation,
            grid_size=grid_size,
            num_chunks=num_chunks,
            num_levels=num_levels,
        )
        quantized_runs[f"{num_chunks}x{num_levels}"] = summarize(scored)

    continuous = _run_continuous_state_probe(
        fit_examples=fit,
        evaluation_examples=evaluation,
        grid_size=grid_size,
    )
    return {
        "fit_examples": len(fit),
        "evaluation_examples": len(evaluation),
        "observation": {
            "current_content_joint_accuracy": observation["current_content_joint_accuracy"],
            "memory_content_joint_accuracy": observation["memory_content_joint_accuracy"],
            "content_only_joint_accuracy": observation["content_only_joint_accuracy"],
            "attended_visible_type_accuracy": observation["attended_visible_type_accuracy"],
            "attended_digit_accuracy": observation["attended_digit_accuracy"],
            "previous_attended_visible_type_accuracy": observation[
                "previous_attended_visible_type_accuracy"
            ],
            "previous_attended_digit_accuracy": observation["previous_attended_digit_accuracy"],
            "previous_glimpse_digit_accuracy": observation["previous_glimpse_digit_accuracy"],
        },
        "quantized_opaque_levels": quantized_runs,
        "continuous_internal_state_probe": summarize(continuous),
    }


def main() -> None:
    device = torch.device("cpu")
    cfg = load_config(CONFIG_PATH)
    _, task_cfg, models = load_models_from_checkpoint(CHECKPOINT_PATH, device, cfg)
    model = models["recurrent"]
    model.eval()

    generator = make_generator(int(cfg["seed"]) + SEED_OFFSET, device)
    batch = generate_batch(
        PROBE_SCENES,
        task_cfg.num_steps,
        task_cfg,
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        outputs = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
        )

    default_examples = collect_nl_examples(model, task_cfg, batch, outputs)
    cue_switch_examples = collect_cue_switch_nl_examples(
        model,
        task_cfg,
        batch,
        switch_step=int(cfg["evaluation"]["cue_switch"].get("switch_step", 3)),
    )
    intervention = collect_intervention_nl_examples(
        model,
        task_cfg,
        batch,
        intervention_step=int(cfg["evaluation"]["intervention_test"].get("step", 5)),
    )

    result = {
        "note": (
            "Stage 7 latent-only follow-up. Quantised opaque levels and a richer continuous "
            "internal-state-only probe are evaluated on the same held-out examples. The "
            "continuous probe is not a reporter interface claim; it is a bottleneck diagnostic."
        ),
        "config_base": CONFIG_PATH,
        "checkpoint": CHECKPOINT_PATH,
        "probe_scenes": PROBE_SCENES,
        "calibration_examples": CALIBRATION_EXAMPLES,
        "evaluation_examples": EVALUATION_EXAMPLES,
        "translator_train_examples": TRANSLATOR_TRAIN_EXAMPLES,
        "slices": {
            "default": _score_slice(default_examples, grid_size=task_cfg.grid_size),
            "cue_switch": _score_slice(cue_switch_examples, grid_size=task_cfg.grid_size),
            "intervention_baseline": _score_slice(
                intervention["baseline_examples"],
                grid_size=task_cfg.grid_size,
            ),
            "intervention_intervened": _score_slice(
                intervention["intervened_examples"],
                grid_size=task_cfg.grid_size,
            ),
        },
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
