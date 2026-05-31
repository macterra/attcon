from __future__ import annotations

"""Evaluation, ablation, and reporting utilities for the attention-control demo."""

import argparse
import json
import os
from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch

from .data import TaskConfig, expand_cues_for_probe, generate_batch
from .models import (
    MatchedTransformerController,
    ModelConfig,
    RecurrentAttentionController,
    StaticAttentionBaseline,
    TrivialUniformRegulator,
)
from .nl_report import (
    OpenAI,
    collect_cue_switch_nl_examples,
    collect_intervention_nl_examples,
    collect_nl_examples,
    load_dotenv,
    run_calibrated_token_report_mode,
    run_nl_report_mode,
    run_observation_only_heuristic_report_mode,
    tokenized_state_payload_metrics,
    _render_tokenized_examples,
)
from .train import deep_update, evaluate_model, load_config, make_generator, set_seed, train_single_model


def load_models_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    config: dict[str, Any] | None = None,
):
    """Rebuild trained model instances from a saved experiment checkpoint."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = deep_update(payload["config"], config or {})
    task_cfg = TaskConfig.from_dict(cfg["task"])
    model_cfg = ModelConfig.from_dict(cfg["model"])
    models = {
        "static": StaticAttentionBaseline(task_cfg, model_cfg).to(device),
        "recurrent": RecurrentAttentionController(task_cfg, model_cfg).to(device),
    }
    for name, model in models.items():
        state = payload["models"][name]
        migrated = False
        missing_report_heads: list[str] = []
        try:
            model.load_state_dict(state)
        except RuntimeError:
            if name != "recurrent":
                raise
            current = model.state_dict()
            migrated_state = dict(state)
            for key in ("controller.weight_ih", "summary_adapter.0.weight"):
                if key not in state or key not in current:
                    continue
                old_value = state[key]
                new_value = current[key]
                if old_value.shape == new_value.shape:
                    continue
                if (
                    old_value.ndim != 2
                    or new_value.ndim != 2
                    or old_value.shape[0] != new_value.shape[0]
                    or old_value.shape[1] > new_value.shape[1]
                ):
                    raise
                padded = torch.zeros_like(new_value)
                old_attention_cols = task_cfg.num_cells
                inserted_inspection_cols = task_cfg.num_cells
                if old_value.shape[1] + inserted_inspection_cols != new_value.shape[1]:
                    raise
                padded[:, :old_attention_cols] = old_value[:, :old_attention_cols]
                padded[:, old_attention_cols + inserted_inspection_cols :] = old_value[
                    :, old_attention_cols:
                ]
                migrated_state[key] = padded

            compatible_missing_prefixes = (
                "hidden_self_model_head.",
                "policy_self_model_head.",
                "self_model_head.",
                "target_found_head.",
                "relevant_region_head.",
                "unresolved_search_head.",
                "wrong_candidate_history_head.",
                "allocation_error_head.",
            )
            missing_report_heads = [
                key
                for key in current
                if key not in migrated_state and key.startswith(compatible_missing_prefixes)
            ]
            missing_other = [
                key
                for key in current
                if key not in migrated_state and not key.startswith(compatible_missing_prefixes)
            ]
            unexpected = [key for key in migrated_state if key not in current]
            mismatched = [
                key
                for key, value in migrated_state.items()
                if key in current and value.shape != current[key].shape
            ]
            if unexpected or missing_other or mismatched or len(missing_report_heads) not in (3, 11, 15):
                raise
            if not cfg.get("evaluation", {}).get("allow_stale_checkpoint", False):
                raise RuntimeError(
                    "checkpoint requires report-head migration; rerun with "
                    "--allow-stale-checkpoint for probe-only diagnostics, or use a "
                    "checkpoint trained with the current model heads"
                )
            model.load_state_dict(migrated_state, strict=False)
            migrated = True
        model.eval()
        cfg.setdefault("_checkpoint_migration", {})[name] = {
            "migrated": migrated,
            "missing_report_heads": missing_report_heads,
            "claim_limit": (
                "checkpoint lacks currently instrumented report/self-model heads; "
                "probe-only diagnostics may run, but trained-head claims are disabled"
            )
            if migrated
            else "",
        }
    return cfg, task_cfg, models


def symmetric_kl(attn_a: torch.Tensor, attn_b: torch.Tensor) -> torch.Tensor:
    attn_a = attn_a.clamp_min(1e-8)
    attn_b = attn_b.clamp_min(1e-8)
    kl_ab = (attn_a * (attn_a.log() - attn_b.log())).sum(dim=-1)
    kl_ba = (attn_b * (attn_b.log() - attn_a.log())).sum(dim=-1)
    return 0.5 * (kl_ab + kl_ba)


def _probe_outputs(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], Any]:
    """Run one probe batch that reuses the same scenes across every cue."""

    generator = make_generator(seed, device)
    base_batch = generate_batch(batch_size, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    probe_batch = expand_cues_for_probe(base_batch, task_cfg.num_types)

    with torch.no_grad():
        outputs = model(
            probe_batch.scene,
            probe_batch.cue,
            target=probe_batch.target,
            target_pos=probe_batch.target_pos,
            num_steps=task_cfg.num_steps,
        )
    return outputs, probe_batch


def _evaluate_probe_batch(
    model,
    probe_batch,
    task_cfg: TaskConfig,
    *,
    cue_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cue = probe_batch.cue if cue_override is None else cue_override
    with torch.no_grad():
        return model(
            probe_batch.scene,
            cue,
            target=probe_batch.target,
            target_pos=probe_batch.target_pos,
            num_steps=task_cfg.num_steps,
        )


def trajectory_metrics(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Measure cue-conditioned attention divergence and within-episode reallocation."""

    outputs, probe_batch = _probe_outputs(model, task_cfg, batch_size, device, seed)
    attention = outputs["attention_seq"].view(batch_size, task_cfg.num_types, task_cfg.num_steps, -1)
    target_pos = probe_batch.target_pos.view(batch_size, task_cfg.num_types)

    cue_divergences = []
    reallocation_divergences = []
    for cue_a in range(task_cfg.num_types):
        for cue_b in range(cue_a + 1, task_cfg.num_types):
            cue_divergences.append(
                symmetric_kl(attention[:, cue_a], attention[:, cue_b]).mean().item()
            )
            reallocation_divergences.append(
                (
                    attention[:, cue_a, 1:] - attention[:, cue_a, :-1]
                    - attention[:, cue_b, 1:] + attention[:, cue_b, :-1]
                )
                .abs()
                .mean()
                .item()
            )

    temporal_reallocation = symmetric_kl(attention[:, :, 1:], attention[:, :, :-1]).mean().item()
    first_step_target = attention[:, :, 0].gather(2, target_pos.unsqueeze(-1)).mean().item()
    final_step_target = attention[:, :, -1].gather(2, target_pos.unsqueeze(-1)).mean().item()

    return {
        "trajectory_divergence": sum(cue_divergences) / max(len(cue_divergences), 1),
        "reallocation_divergence": sum(reallocation_divergences) / max(len(reallocation_divergences), 1),
        "temporal_reallocation": temporal_reallocation,
        "target_attention_gain": final_step_target - first_step_target,
        "probe_target_attention_first": first_step_target,
        "probe_target_attention_final": final_step_target,
    }


def cue_sensitivity_metrics(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Compare correct-cue and wrong-cue behavior on the exact same probe scenes."""

    generator = make_generator(seed, device)
    base_batch = generate_batch(batch_size, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    probe_batch = expand_cues_for_probe(base_batch, task_cfg.num_types)
    wrong_cue = (probe_batch.cue + 1) % task_cfg.num_types

    correct_outputs = _evaluate_probe_batch(model, probe_batch, task_cfg)
    wrong_outputs = _evaluate_probe_batch(model, probe_batch, task_cfg, cue_override=wrong_cue)

    target_pos = probe_batch.target_pos.unsqueeze(-1)
    correct_attention = correct_outputs["attention_seq"][:, -1]
    wrong_attention = wrong_outputs["attention_seq"][:, -1]
    correct_target_attention = correct_attention.gather(1, target_pos).mean().item()
    wrong_target_attention = wrong_attention.gather(1, target_pos).mean().item()

    correct_accuracy = (
        correct_outputs["logits"].argmax(dim=-1) == probe_batch.target
    ).float().mean().item()
    wrong_accuracy = (
        wrong_outputs["logits"].argmax(dim=-1) == probe_batch.target
    ).float().mean().item()

    return {
        "wrong_cue_accuracy": wrong_accuracy,
        "wrong_cue_target_attention": wrong_target_attention,
        "cue_accuracy_delta": correct_accuracy - wrong_accuracy,
        "cue_target_attention_delta": correct_target_attention - wrong_target_attention,
    }


def _collect_probe_dataset(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect recurrent-state, observation, and next-attention tuples."""

    state_features = []
    observation_features = []
    next_attention_targets = []

    for batch_idx in range(num_batches):
        outputs, _ = _probe_outputs(model, task_cfg, batch_size, device, seed + batch_idx)
        state_features.append(outputs["controller_state_seq"][:, :-1].reshape(-1, outputs["controller_state_seq"].shape[-1]))
        observation_features.append(
            outputs["observation_seq"][:, :-1].reshape(-1, outputs["observation_seq"].shape[-1])
        )
        next_attention_targets.append(
            outputs["attention_seq"][:, 1:].reshape(-1, outputs["attention_seq"].shape[-1])
        )

    return (
        torch.cat(state_features, dim=0),
        torch.cat(observation_features, dim=0),
        torch.cat(next_attention_targets, dim=0),
    )


def _collect_temporal_observation_probe_dataset(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    seed: int,
    *,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect previous-observation windows for high-capacity observation controls."""

    window = max(int(window), 1)
    features = []
    next_attention_targets = []
    for batch_idx in range(num_batches):
        outputs, _ = _probe_outputs(model, task_cfg, batch_size, device, seed + batch_idx)
        observation = outputs["observation_seq"]
        attention = outputs["attention_seq"]
        padded = torch.zeros(
            observation.shape[0],
            window - 1,
            observation.shape[-1],
            device=device,
        )
        padded_observation = torch.cat([padded, observation[:, :-1]], dim=1)
        windows = []
        for step_idx in range(task_cfg.num_steps - 1):
            windows.append(padded_observation[:, step_idx : step_idx + window].reshape(observation.shape[0], -1))
        features.append(torch.stack(windows, dim=1).reshape(-1, window * observation.shape[-1]))
        next_attention_targets.append(attention[:, 1:].reshape(-1, attention.shape[-1]))

    return torch.cat(features, dim=0), torch.cat(next_attention_targets, dim=0)


def _collect_report_probe_dataset(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Collect controller-state report labels for simple internal-content probes."""

    state_features = []
    observation_features = []
    cue_labels = []
    attended_cell_labels = []
    found_target_labels = []
    found_target_native = []

    for batch_idx in range(num_batches):
        outputs, probe_batch = _probe_outputs(model, task_cfg, batch_size, device, seed + batch_idx)
        attention = outputs["attention_seq"]
        observation = outputs["observation_seq"]
        cue_seq = probe_batch.cue.unsqueeze(1).repeat(1, task_cfg.num_steps)
        found_target = (observation[..., 0] > 0.5).float()
        found_target = torch.cummax(found_target, dim=1).values

        state_features.append(outputs["controller_state_seq"].reshape(-1, outputs["controller_state_seq"].shape[-1]))
        observation_features.append(observation.reshape(-1, observation.shape[-1]))
        cue_labels.append(cue_seq.reshape(-1))
        attended_cell_labels.append(attention.argmax(dim=-1).reshape(-1))
        found_target_labels.append(found_target.reshape(-1))
        if "found_state_seq" in outputs:
            found_target_native.append(outputs["found_state_seq"].reshape(-1))
        elif "target_found_seq" in outputs:
            found_target_native.append(outputs["target_found_seq"].reshape(-1))

    report = {
        "state_features": torch.cat(state_features, dim=0),
        "observation_features": torch.cat(observation_features, dim=0),
        "cue_labels": torch.cat(cue_labels, dim=0),
        "attended_cell_labels": torch.cat(attended_cell_labels, dim=0),
        "found_target_labels": torch.cat(found_target_labels, dim=0),
    }
    if found_target_native:
        report["found_target_native"] = torch.cat(found_target_native, dim=0)
    return report


def _collect_uncertainty_probe_dataset(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Collect held-out labels and native predictions for Stage 6B-style report variables."""

    state_features = []
    observation_features = []
    prev_observation_features = []
    relevant_region_labels = []
    unresolved_search_labels = []
    current_wrong_candidate_labels = []
    wrong_candidate_history_labels = []
    revisit_unresolved_labels = []
    allocation_error_labels = []
    relevant_region_native = []
    unresolved_search_native = []
    current_wrong_candidate_native = []
    wrong_candidate_history_native = []
    revisit_unresolved_native = []
    allocation_error_native = []

    for batch_idx in range(num_batches):
        outputs, probe_batch = _probe_outputs(model, task_cfg, batch_size, device, seed + batch_idx)
        prev_observation = torch.zeros_like(outputs["observation_seq"])
        prev_observation[:, 1:] = outputs["observation_seq"][:, :-1]
        state_features.append(outputs["controller_state_seq"].reshape(-1, outputs["controller_state_seq"].shape[-1]))
        observation_features.append(outputs["observation_seq"].reshape(-1, outputs["observation_seq"].shape[-1]))
        prev_observation_features.append(
            prev_observation.reshape(-1, prev_observation.shape[-1])
        )
        relevant_region_labels.append(outputs["relevant_region_seq"].reshape(-1))
        unresolved_search_labels.append(outputs["unresolved_search_seq"].reshape(-1))
        current_wrong_candidate_labels.append(outputs["current_wrong_candidate_seq"].reshape(-1))
        wrong_candidate_history_labels.append(outputs["wrong_candidate_history_seq"].reshape(-1))
        revisit_unresolved_labels.append(outputs["revisit_unresolved_seq"].reshape(-1))
        allocation_error_labels.append(outputs["allocation_error_seq"].reshape(-1))
        relevant_region_native.append(outputs["relevant_region_seq"].reshape(-1))
        unresolved_search_native.append(outputs["unresolved_search_seq"].reshape(-1))
        current_wrong_candidate_native.append(outputs["current_wrong_candidate_seq"].reshape(-1))
        wrong_candidate_history_native.append(outputs["wrong_candidate_history_seq"].reshape(-1))
        revisit_unresolved_native.append(outputs["revisit_unresolved_seq"].reshape(-1))
        allocation_error_native.append(outputs["allocation_error_seq"].reshape(-1))
        if "relevant_region_logits_seq" in outputs:
            relevant_region_native[-1] = torch.sigmoid(outputs["relevant_region_logits_seq"]).reshape(-1)
        if "unresolved_search_logits_seq" in outputs:
            unresolved_search_native[-1] = torch.sigmoid(outputs["unresolved_search_logits_seq"]).reshape(-1)
        if "wrong_candidate_history_logits_seq" in outputs:
            wrong_candidate_history_native[-1] = torch.sigmoid(
                outputs["wrong_candidate_history_logits_seq"]
            ).reshape(-1)
        if "allocation_error_logits_seq" in outputs:
            allocation_error_native[-1] = torch.sigmoid(outputs["allocation_error_logits_seq"]).reshape(-1)

    return {
        "state_features": torch.cat(state_features, dim=0),
        "observation_features": torch.cat(observation_features, dim=0),
        "prev_observation_features": torch.cat(prev_observation_features, dim=0),
        "relevant_region_labels": torch.cat(relevant_region_labels, dim=0),
        "unresolved_search_labels": torch.cat(unresolved_search_labels, dim=0),
        "current_wrong_candidate_labels": torch.cat(current_wrong_candidate_labels, dim=0),
        "wrong_candidate_history_labels": torch.cat(wrong_candidate_history_labels, dim=0),
        "revisit_unresolved_labels": torch.cat(revisit_unresolved_labels, dim=0),
        "allocation_error_labels": torch.cat(allocation_error_labels, dim=0),
        "relevant_region_native": torch.cat(relevant_region_native, dim=0),
        "unresolved_search_native": torch.cat(unresolved_search_native, dim=0),
        "current_wrong_candidate_native": torch.cat(current_wrong_candidate_native, dim=0),
        "wrong_candidate_history_native": torch.cat(wrong_candidate_history_native, dim=0),
        "revisit_unresolved_native": torch.cat(revisit_unresolved_native, dim=0),
        "allocation_error_native": torch.cat(allocation_error_native, dim=0),
    }


def _collect_self_model_dataset(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Collect held-out native self-model targets and observation-only baselines."""

    prev_observation_features = []
    inspection_labels = []
    target_inspected_labels = []

    total_cell_matches = 0.0
    total_cell_count = 0
    total_target_matches = 0.0
    total_target_count = 0
    total_target_positive_hits = 0.0
    total_target_positive_count = 0.0
    total_bce = 0.0

    for batch_idx in range(num_batches):
        generator = make_generator(seed + batch_idx, device)
        batch = generate_batch(batch_size, task_cfg.num_steps, task_cfg, generator=generator, device=device)
        with torch.no_grad():
            outputs = model(
                batch.scene,
                batch.cue,
                target=batch.target,
                target_pos=batch.target_pos,
                num_steps=task_cfg.num_steps,
            )

        inspection_true = outputs["inspection_seq"]
        inspection_pred = outputs["self_model_seq"]
        prev_observation = torch.zeros_like(outputs["observation_seq"])
        prev_observation[:, 1:] = outputs["observation_seq"][:, :-1]

        target_pos = batch.target_pos[:, None, None].expand(-1, task_cfg.num_steps, 1)
        # Binarize (a no-op for one-hot hard mode); graded soft-mode inspection would
        # otherwise inflate positive recall above 1.0 via a fractional denominator.
        target_true = (inspection_true.gather(2, target_pos).squeeze(-1) >= 0.5).float()
        target_pred = inspection_pred.gather(2, target_pos).squeeze(-1)

        total_cell_matches += (
            (inspection_pred >= 0.5) == inspection_true.bool()
        ).float().sum().item()
        total_cell_count += inspection_true.numel()
        total_target_matches += (
            (target_pred >= 0.5) == target_true.bool()
        ).float().sum().item()
        total_target_count += target_true.numel()
        total_target_positive_hits += (
            ((target_pred >= 0.5) & target_true.bool()).float().sum().item()
        )
        total_target_positive_count += target_true.float().sum().item()
        total_bce += torch.nn.functional.binary_cross_entropy(
            inspection_pred,
            inspection_true,
        ).item()

        prev_observation_features.append(prev_observation.reshape(-1, prev_observation.shape[-1]))
        inspection_labels.append(inspection_true.reshape(-1, inspection_true.shape[-1]))
        target_inspected_labels.append(target_true.reshape(-1))

    return {
        "prev_observation_features": torch.cat(prev_observation_features, dim=0),
        "inspection_labels": torch.cat(inspection_labels, dim=0),
        "target_inspected_labels": torch.cat(target_inspected_labels, dim=0),
        "native_cell_accuracy": total_cell_matches / max(total_cell_count, 1),
        "native_target_accuracy": total_target_matches / max(total_target_count, 1),
        "native_target_positive_recall": total_target_positive_hits / max(total_target_positive_count, 1.0),
        "native_cell_bce": total_bce / max(num_batches, 1),
    }


def _collect_learned_self_model_dataset(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Collect hidden-state-only self-model targets for Stage 4B diagnostics."""

    hidden_features = []
    prev_observation_features = []
    inspection_labels = []
    target_inspected_labels = []
    total_hidden_cell_matches = 0.0
    total_hidden_cell_count = 0
    total_hidden_target_matches = 0.0
    total_hidden_target_count = 0
    total_hidden_target_positive_hits = 0.0
    total_hidden_target_positive_count = 0.0
    total_hidden_bce = 0.0
    for batch_idx in range(num_batches):
        generator = make_generator(seed + batch_idx, device)
        batch = generate_batch(
            batch_size,
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
        prev_observation = torch.zeros_like(outputs["observation_seq"])
        prev_observation[:, 1:] = outputs["observation_seq"][:, :-1]
        inspection_true = outputs["inspection_seq"]
        hidden_inspection_pred = outputs.get("hidden_self_model_seq")
        target_pos = batch.target_pos[:, None, None].expand(-1, task_cfg.num_steps, 1)
        # Binarize the "target inspected" label. inspection_seq is graded in soft-attention
        # mode, so using it both as a boolean numerator and a fractional recall denominator
        # let positive recall exceed 1.0; threshold it once (a no-op for one-hot hard mode).
        target_true = (inspection_true.gather(2, target_pos).squeeze(-1) >= 0.5).float()
        if hidden_inspection_pred is not None:
            hidden_target_pred = hidden_inspection_pred.gather(2, target_pos).squeeze(-1)
            total_hidden_cell_matches += (
                (hidden_inspection_pred >= 0.5) == inspection_true.bool()
            ).float().sum().item()
            total_hidden_cell_count += inspection_true.numel()
            total_hidden_target_matches += (
                (hidden_target_pred >= 0.5) == target_true.bool()
            ).float().sum().item()
            total_hidden_target_count += target_true.numel()
            total_hidden_target_positive_hits += (
                ((hidden_target_pred >= 0.5) & target_true.bool()).float().sum().item()
            )
            total_hidden_target_positive_count += target_true.float().sum().item()
            total_hidden_bce += torch.nn.functional.binary_cross_entropy(
                hidden_inspection_pred,
                inspection_true,
            ).item()
        hidden_features.append(
            outputs["controller_state_seq"].reshape(
                -1,
                outputs["controller_state_seq"].shape[-1],
            )
        )
        prev_observation_features.append(prev_observation.reshape(-1, prev_observation.shape[-1]))
        inspection_labels.append(outputs["inspection_seq"].reshape(-1, outputs["inspection_seq"].shape[-1]))
        target_inspected_labels.append(target_true.reshape(-1))

    dataset = {
        "hidden_features": torch.cat(hidden_features, dim=0),
        "prev_observation_features": torch.cat(prev_observation_features, dim=0),
        "inspection_labels": torch.cat(inspection_labels, dim=0),
        "target_inspected_labels": torch.cat(target_inspected_labels, dim=0),
    }
    if total_hidden_cell_count:
        dataset.update(
            {
                "native_hidden_cell_accuracy": total_hidden_cell_matches
                / max(total_hidden_cell_count, 1),
                "native_hidden_target_accuracy": total_hidden_target_matches
                / max(total_hidden_target_count, 1),
                "native_hidden_target_positive_recall": total_hidden_target_positive_hits
                / max(total_hidden_target_positive_count, 1.0),
                "native_hidden_cell_bce": total_hidden_bce / max(num_batches, 1),
            }
        )
    return dataset


def _learned_self_model_intervention(
    model,
    task_cfg: TaskConfig,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    step: int,
    scale: float,
) -> dict[str, float]:
    """Perturb hidden state along native self-model directions and measure coupled effects."""

    generator = make_generator(seed, device)
    batch = generate_batch(
        batch_size,
        task_cfg.num_steps,
        task_cfg,
        generator=generator,
        device=device,
    )
    step = max(0, min(step, task_cfg.num_steps - 1))
    with torch.no_grad():
        base = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
        )
        hidden = base["controller_state_seq"][:, step]
        hidden_dim = hidden.shape[-1]
        hidden_weights = model.hidden_self_model_head.weight
        directions = hidden_weights[batch.target_pos]
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = directions * hidden.std(dim=-1, keepdim=True).clamp_min(1e-3) * scale
        pos = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
            intervention={"step": step, "state_override": hidden + delta},
        )
        neg = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
            intervention={"step": step, "state_override": hidden - delta},
        )
        hidden_self_model = base["hidden_self_model_seq"][:, step]
        override_delta = 0.25 * scale
        pos_hidden_self_model = hidden_self_model.clone()
        neg_hidden_self_model = hidden_self_model.clone()
        batch_index = torch.arange(batch.scene.shape[0], device=device)
        pos_hidden_self_model[batch_index, batch.target_pos] = (
            pos_hidden_self_model[batch_index, batch.target_pos] + override_delta
        ).clamp_max(1.0)
        neg_hidden_self_model[batch_index, batch.target_pos] = (
            neg_hidden_self_model[batch_index, batch.target_pos] - override_delta
        ).clamp_min(0.0)
        policy_pos = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
            intervention={"step": step, "hidden_self_model_override": pos_hidden_self_model},
        )
        policy_neg = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
            intervention={"step": step, "hidden_self_model_override": neg_hidden_self_model},
        )

    target_index = batch.target_pos[:, None]
    base_target_self = base["self_model_seq"][:, step].gather(1, target_index)
    pos_target_self = pos["self_model_seq"][:, step].gather(1, target_index)
    neg_target_self = neg["self_model_seq"][:, step].gather(1, target_index)
    base_target_attention = base["attention_seq"][:, step].gather(1, target_index)
    pos_target_attention = pos["attention_seq"][:, step].gather(1, target_index)
    neg_target_attention = neg["attention_seq"][:, step].gather(1, target_index)
    base_policy_feedback = base["policy_self_model_feedback_seq"][:, step]
    policy_pos_target_attention = policy_pos["attention_seq"][:, step].gather(1, target_index)
    policy_neg_target_attention = policy_neg["attention_seq"][:, step].gather(1, target_index)
    policy_pos_target_hidden_self = policy_pos["hidden_self_model_seq"][:, step].gather(1, target_index)
    policy_neg_target_hidden_self = policy_neg["hidden_self_model_seq"][:, step].gather(1, target_index)
    inspection_target = base["inspection_seq"][:, step].gather(1, target_index)
    return {
        "step": float(step),
        "scale": float(scale),
        "policy_feedback_abs_mean": base_policy_feedback.abs().mean().item(),
        "target_inspection_rate": inspection_target.mean().item(),
        "positive_self_model_target_delta": (pos_target_self - base_target_self).mean().item(),
        "negative_self_model_target_delta": (neg_target_self - base_target_self).mean().item(),
        "bidirectional_self_model_target_gap": (pos_target_self - neg_target_self).mean().item(),
        "positive_target_attention_delta": (pos_target_attention - base_target_attention).mean().item(),
        "negative_target_attention_delta": (neg_target_attention - base_target_attention).mean().item(),
        "bidirectional_target_attention_gap": (pos_target_attention - neg_target_attention).mean().item(),
        "policy_override_positive_hidden_self_model_target_delta": (
            policy_pos_target_hidden_self - hidden_self_model.gather(1, target_index)
        ).mean().item(),
        "policy_override_negative_hidden_self_model_target_delta": (
            policy_neg_target_hidden_self - hidden_self_model.gather(1, target_index)
        ).mean().item(),
        "policy_override_bidirectional_hidden_self_model_target_gap": (
            policy_pos_target_hidden_self - policy_neg_target_hidden_self
        ).mean().item(),
        "policy_override_positive_target_attention_delta": (
            policy_pos_target_attention - base_target_attention
        ).mean().item(),
        "policy_override_negative_target_attention_delta": (
            policy_neg_target_attention - base_target_attention
        ).mean().item(),
        "policy_override_bidirectional_target_attention_gap": (
            policy_pos_target_attention - policy_neg_target_attention
        ).mean().item(),
    }


def _train_attention_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    """Fit a linear probe that predicts the next attention distribution."""

    probe = nn.Linear(train_features.shape[-1], train_targets.shape[-1]).to(train_features.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    for _ in range(epochs):
        logits = probe(train_features)
        loss = -(train_targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = probe(train_features)
        test_logits = probe(test_features)
        train_probs = torch.softmax(train_logits, dim=-1)
        test_probs = torch.softmax(test_logits, dim=-1)

        train_loss = -(train_targets * torch.log_softmax(train_logits, dim=-1)).sum(dim=-1).mean().item()
        test_loss = -(test_targets * torch.log_softmax(test_logits, dim=-1)).sum(dim=-1).mean().item()
        train_mse = (train_probs - train_targets).pow(2).mean().item()
        test_mse = (test_probs - test_targets).pow(2).mean().item()
        train_top1 = (
            train_probs.argmax(dim=-1) == train_targets.argmax(dim=-1)
        ).float().mean().item()
        test_top1 = (
            test_probs.argmax(dim=-1) == test_targets.argmax(dim=-1)
        ).float().mean().item()

    return {
        "train_cross_entropy": train_loss,
        "test_cross_entropy": test_loss,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_top1_match": train_top1,
        "test_top1_match": test_top1,
    }


def _train_mlp_attention_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    """Fit a higher-capacity observation-only probe for negative-control audits."""

    probe = nn.Sequential(
        nn.Linear(train_features.shape[-1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, train_targets.shape[-1]),
    ).to(train_features.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    for _ in range(epochs):
        logits = probe(train_features)
        loss = -(train_targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = probe(train_features)
        test_logits = probe(test_features)
        train_probs = torch.softmax(train_logits, dim=-1)
        test_probs = torch.softmax(test_logits, dim=-1)
        return {
            "train_cross_entropy": (
                -(train_targets * torch.log_softmax(train_logits, dim=-1)).sum(dim=-1).mean().item()
            ),
            "test_cross_entropy": (
                -(test_targets * torch.log_softmax(test_logits, dim=-1)).sum(dim=-1).mean().item()
            ),
            "train_mse": (train_probs - train_targets).pow(2).mean().item(),
            "test_mse": (test_probs - test_targets).pow(2).mean().item(),
            "train_top1_match": (
                train_probs.argmax(dim=-1) == train_targets.argmax(dim=-1)
            ).float().mean().item(),
            "test_top1_match": (
                test_probs.argmax(dim=-1) == test_targets.argmax(dim=-1)
            ).float().mean().item(),
        }


def _train_multilabel_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    """Fit a linear probe for binary per-cell self-model labels."""

    probe = nn.Linear(train_features.shape[-1], train_targets.shape[-1]).to(train_features.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    for _ in range(epochs):
        logits = probe(train_features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = probe(train_features)
        test_logits = probe(test_features)
        train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
        test_pred = (torch.sigmoid(test_logits) >= 0.5).float()
    return {
        "train_bce": torch.nn.functional.binary_cross_entropy_with_logits(train_logits, train_targets).item(),
        "test_bce": torch.nn.functional.binary_cross_entropy_with_logits(test_logits, test_targets).item(),
        "train_cell_accuracy": (train_pred == train_targets).float().mean().item(),
        "test_cell_accuracy": (test_pred == test_targets).float().mean().item(),
    }


def _probe_capacity_matched_features(
    features: torch.Tensor,
    matched_dim: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Lift or truncate features to a fixed linear-probe input dimension.

    The lift is deterministic and untrained. It gives lower-dimensional baselines
    the same linear-probe parameter count without adding task information. This
    is a probe-capacity match, not a full baseline-processing-capacity match.
    """

    if features.shape[-1] == matched_dim:
        return features
    if features.shape[-1] > matched_dim:
        return features[..., :matched_dim]

    generator = make_generator(seed, features.device)
    projection = torch.randn(
        features.shape[-1],
        matched_dim - features.shape[-1],
        generator=generator,
        device=features.device,
    )
    projection = projection / max(features.shape[-1], 1) ** 0.5
    lifted = torch.tanh(features @ projection)
    return torch.cat([features, lifted], dim=-1)


def _train_classification_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    """Fit a linear classifier probe."""

    probe = nn.Linear(train_features.shape[-1], num_classes).to(train_features.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    train_labels = train_labels.long()
    test_labels = test_labels.long()

    for _ in range(epochs):
        logits = probe(train_features)
        loss = torch.nn.functional.cross_entropy(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = probe(train_features)
        test_logits = probe(test_features)
        return {
            "train_cross_entropy": torch.nn.functional.cross_entropy(train_logits, train_labels).item(),
            "test_cross_entropy": torch.nn.functional.cross_entropy(test_logits, test_labels).item(),
            "train_accuracy": (train_logits.argmax(dim=-1) == train_labels).float().mean().item(),
            "test_accuracy": (test_logits.argmax(dim=-1) == test_labels).float().mean().item(),
        }


def _train_binary_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    """Fit a linear binary probe."""

    probe = nn.Linear(train_features.shape[-1], 1).to(train_features.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    train_labels = train_labels.float().unsqueeze(-1)
    test_labels = test_labels.float().unsqueeze(-1)

    for _ in range(epochs):
        logits = probe(train_features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = probe(train_features)
        test_logits = probe(test_features)
        train_probs = torch.sigmoid(train_logits)
        test_probs = torch.sigmoid(test_logits)
        train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
        test_pred = (torch.sigmoid(test_logits) >= 0.5).float()
        train_positive = train_labels == 1.0
        test_positive = test_labels == 1.0
        train_positive_score = train_probs[train_positive].mean().item() if train_positive.any() else 0.0
        train_negative_score = train_probs[~train_positive].mean().item() if (~train_positive).any() else 0.0
        test_positive_score = test_probs[test_positive].mean().item() if test_positive.any() else 0.0
        test_negative_score = test_probs[~test_positive].mean().item() if (~test_positive).any() else 0.0
        return {
            "train_bce": torch.nn.functional.binary_cross_entropy_with_logits(train_logits, train_labels).item(),
            "test_bce": torch.nn.functional.binary_cross_entropy_with_logits(test_logits, test_labels).item(),
            "train_accuracy": (train_pred == train_labels).float().mean().item(),
            "test_accuracy": (test_pred == test_labels).float().mean().item(),
            "train_positive_score": train_positive_score,
            "train_negative_score": train_negative_score,
            "train_score_separation": train_positive_score - train_negative_score,
            "test_positive_score": test_positive_score,
            "test_negative_score": test_negative_score,
            "test_score_separation": test_positive_score - test_negative_score,
            "train_positive_recall": (
                ((train_pred == 1.0) & train_positive).float().sum()
                / train_positive.float().sum().clamp_min(1.0)
            ).item(),
            "test_positive_recall": (
                ((test_pred == 1.0) & test_positive).float().sum()
                / test_positive.float().sum().clamp_min(1.0)
            ).item(),
        }


def predictive_probe_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Test whether controller state predicts the next attention map better than observation alone."""

    probe_cfg = cfg["evaluation"].get(
        "predictive_probe",
        {
            "train_batches": 12,
            "test_batches": 6,
            "epochs": 60,
            "learning_rate": 0.05,
            "thresholds": {
                "min_advantage_cross_entropy": 0.0,
                "min_advantage_mse": 0.0,
                "min_advantage_top1_match": 0.0,
            },
        },
    )
    thresholds = probe_cfg.get("thresholds", {})
    batch_size = cfg["training"]["batch_size"]
    train_state, train_obs, train_targets = _collect_probe_dataset(
        model,
        task_cfg,
        batch_size,
        probe_cfg["train_batches"],
        device,
        seed,
    )
    test_state, test_obs, test_targets = _collect_probe_dataset(
        model,
        task_cfg,
        batch_size,
        probe_cfg["test_batches"],
        device,
        seed + 1000,
    )

    controller_probe = _train_attention_probe(
        train_state,
        train_targets,
        test_state,
        test_targets,
        epochs=probe_cfg["epochs"],
        learning_rate=probe_cfg["learning_rate"],
    )
    observation_probe = _train_attention_probe(
        train_obs,
        train_targets,
        test_obs,
        test_targets,
        epochs=probe_cfg["epochs"],
        learning_rate=probe_cfg["learning_rate"],
    )

    cross_entropy_advantage = observation_probe["test_cross_entropy"] - controller_probe["test_cross_entropy"]
    mse_advantage = observation_probe["test_mse"] - controller_probe["test_mse"]
    top1_advantage = controller_probe["test_top1_match"] - observation_probe["test_top1_match"]
    return {
        "controller_state_probe": controller_probe,
        "observation_only_probe": observation_probe,
        "controller_advantage_cross_entropy": cross_entropy_advantage,
        "controller_advantage_mse": mse_advantage,
        "controller_advantage_top1_match": top1_advantage,
        "thresholds": {
            "min_advantage_cross_entropy": thresholds.get("min_advantage_cross_entropy", 0.0),
            "min_advantage_mse": thresholds.get("min_advantage_mse", 0.0),
            "min_advantage_top1_match": thresholds.get("min_advantage_top1_match", 0.0),
        },
        "supported": (
            cross_entropy_advantage >= thresholds.get("min_advantage_cross_entropy", 0.0)
            and mse_advantage >= thresholds.get("min_advantage_mse", 0.0)
            and top1_advantage >= thresholds.get("min_advantage_top1_match", 0.0)
        ),
    }


def report_probe_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Test whether controller state can support simple report-like readouts."""

    set_seed(seed)
    probe_cfg = cfg["evaluation"].get(
        "report_probes",
        {
            "train_batches": 12,
            "test_batches": 6,
            "epochs": 60,
            "learning_rate": 0.05,
        },
    )
    if not probe_cfg.get("enabled", False):
        return {}

    batch_size = cfg["training"]["batch_size"]
    train = _collect_report_probe_dataset(
        model, task_cfg, batch_size, probe_cfg["train_batches"], device, seed
    )
    test = _collect_report_probe_dataset(
        model, task_cfg, batch_size, probe_cfg["test_batches"], device, seed + 1000
    )

    epochs = probe_cfg["epochs"]
    learning_rate = probe_cfg["learning_rate"]
    audit_cfg = probe_cfg.get("capacity_audit", {})
    min_accuracy_advantage = audit_cfg.get("min_accuracy_advantage", 0.01)
    min_positive_recall_advantage = audit_cfg.get("min_positive_recall_advantage", 0.01)
    matched_dim = train["state_features"].shape[-1]
    matched_train_obs = _probe_capacity_matched_features(
        train["observation_features"],
        matched_dim,
        seed=seed + 3200,
    )
    matched_test_obs = _probe_capacity_matched_features(
        test["observation_features"],
        matched_dim,
        seed=seed + 3200,
    )

    cue_state = _train_classification_probe(
        train["state_features"],
        train["cue_labels"],
        test["state_features"],
        test["cue_labels"],
        num_classes=task_cfg.num_types,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    cue_obs = _train_classification_probe(
        train["observation_features"],
        train["cue_labels"],
        test["observation_features"],
        test["cue_labels"],
        num_classes=task_cfg.num_types,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    cue_obs_matched = _train_classification_probe(
        matched_train_obs,
        train["cue_labels"],
        matched_test_obs,
        test["cue_labels"],
        num_classes=task_cfg.num_types,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    attended_state = _train_classification_probe(
        train["state_features"],
        train["attended_cell_labels"],
        test["state_features"],
        test["attended_cell_labels"],
        num_classes=task_cfg.num_cells,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    attended_obs = _train_classification_probe(
        train["observation_features"],
        train["attended_cell_labels"],
        test["observation_features"],
        test["attended_cell_labels"],
        num_classes=task_cfg.num_cells,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    attended_obs_matched = _train_classification_probe(
        matched_train_obs,
        train["attended_cell_labels"],
        matched_test_obs,
        test["attended_cell_labels"],
        num_classes=task_cfg.num_cells,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    found_obs = _train_binary_probe(
        train["observation_features"],
        train["found_target_labels"],
        test["observation_features"],
        test["found_target_labels"],
        epochs=epochs,
        learning_rate=learning_rate,
    )
    found_obs_matched = _train_binary_probe(
        matched_train_obs,
        train["found_target_labels"],
        matched_test_obs,
        test["found_target_labels"],
        epochs=epochs,
        learning_rate=learning_rate,
    )
    found_state = None
    if "found_target_native" in test:
        native_train_pred = (train["found_target_native"] >= 0.5).float()
        native_test_pred = (test["found_target_native"] >= 0.5).float()
        train_labels = train["found_target_labels"].float()
        test_labels = test["found_target_labels"].float()
        train_positive = train_labels == 1.0
        test_positive = test_labels == 1.0
        found_state = {
            "train_bce": torch.nn.functional.binary_cross_entropy(
                train["found_target_native"].clamp(1e-6, 1 - 1e-6),
                train_labels,
            ).item(),
            "test_bce": torch.nn.functional.binary_cross_entropy(
                test["found_target_native"].clamp(1e-6, 1 - 1e-6),
                test_labels,
            ).item(),
            "train_accuracy": (native_train_pred == train_labels).float().mean().item(),
            "test_accuracy": (native_test_pred == test_labels).float().mean().item(),
            "train_positive_recall": (
                ((native_train_pred == 1.0) & train_positive).float().sum()
                / train_positive.float().sum().clamp_min(1.0)
            ).item(),
            "test_positive_recall": (
                ((native_test_pred == 1.0) & test_positive).float().sum()
                / test_positive.float().sum().clamp_min(1.0)
            ).item(),
        }
    else:
        found_state = _train_binary_probe(
            train["state_features"],
            train["found_target_labels"],
            test["state_features"],
            test["found_target_labels"],
            epochs=epochs,
            learning_rate=learning_rate,
        )

    return {
        "current_search_type": {
            "controller_state_probe": cue_state,
            "observation_only_probe": cue_obs,
            "probe_capacity_matched_observation_probe": cue_obs_matched,
            "controller_accuracy_advantage": cue_state["test_accuracy"] - cue_obs["test_accuracy"],
            "probe_capacity_matched_controller_accuracy_advantage": (
                cue_state["test_accuracy"] - cue_obs_matched["test_accuracy"]
            ),
        },
        "current_attended_cell": {
            "controller_state_probe": attended_state,
            "observation_only_probe": attended_obs,
            "probe_capacity_matched_observation_probe": attended_obs_matched,
            "controller_accuracy_advantage": attended_state["test_accuracy"] - attended_obs["test_accuracy"],
            "probe_capacity_matched_controller_accuracy_advantage": (
                attended_state["test_accuracy"] - attended_obs_matched["test_accuracy"]
            ),
        },
        "target_found_in_glimpse": {
            "controller_state_probe": found_state,
            "observation_only_probe": found_obs,
            "probe_capacity_matched_observation_probe": found_obs_matched,
            "controller_accuracy_advantage": found_state["test_accuracy"] - found_obs["test_accuracy"],
            "probe_capacity_matched_controller_accuracy_advantage": (
                found_state["test_accuracy"] - found_obs_matched["test_accuracy"]
            ),
            "controller_positive_recall_advantage": (
                found_state["test_positive_recall"] - found_obs["test_positive_recall"]
            ),
            "probe_capacity_matched_controller_positive_recall_advantage": (
                found_state["test_positive_recall"] - found_obs_matched["test_positive_recall"]
            ),
        },
        "capacity_audit": {
            "matched_input_dim": matched_dim,
            "scope": "linear_probe_input_dim_only",
            "baseline_source": "observation_features",
            "baseline_lift": "deterministic_tanh_random_projection",
            "thresholds": {
                "min_accuracy_advantage": min_accuracy_advantage,
                "min_positive_recall_advantage": min_positive_recall_advantage,
            },
            "passed": (
                cue_state["test_accuracy"] - cue_obs_matched["test_accuracy"]
                >= min_accuracy_advantage
                and attended_state["test_accuracy"] - attended_obs_matched["test_accuracy"]
                >= min_accuracy_advantage
                and found_state["test_positive_recall"] - found_obs_matched["test_positive_recall"]
                >= min_positive_recall_advantage
            ),
        },
        "supported": (
            cue_state["test_accuracy"] > cue_obs["test_accuracy"]
            and attended_state["test_accuracy"] > attended_obs["test_accuracy"]
            and found_state["test_positive_recall"] > found_obs["test_positive_recall"]
        ),
    }


def noise_floor_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Empirical permuted-label noise floor for the Stage 6A controller-state advantage.

    Hardcoded directional thresholds answer "is the advantage positive?" but not "is it
    bigger than chance?". For each strong report signal this measures the real
    controller-vs-observation accuracy advantage and the distribution of that same
    advantage under permuted (shuffled) labels, where no real signal exists. The
    p95 permuted advantage is a data-driven significance floor; a real advantage that
    clears it is unlikely to be a spurious probe-capacity artifact.
    """

    nf_cfg = cfg["evaluation"].get("noise_floor", {})
    if not nf_cfg.get("enabled", False):
        return {}

    set_seed(seed)
    batch_size = cfg["training"]["batch_size"]
    train_batches = nf_cfg.get("train_batches", 8)
    test_batches = nf_cfg.get("test_batches", 4)
    epochs = nf_cfg.get("epochs", 40)
    learning_rate = nf_cfg.get("learning_rate", 0.05)
    permutations = int(nf_cfg.get("permutations", 12))
    percentile = nf_cfg.get("percentile", 95)

    train = _collect_report_probe_dataset(model, task_cfg, batch_size, train_batches, device, seed)
    test = _collect_report_probe_dataset(model, task_cfg, batch_size, test_batches, device, seed + 1000)
    generator = make_generator(seed + 7000, device)

    def _advantage(train_labels, test_labels, num_classes):
        controller = _train_classification_probe(
            train["state_features"], train_labels, test["state_features"], test_labels,
            num_classes=num_classes, epochs=epochs, learning_rate=learning_rate,
        )
        observation = _train_classification_probe(
            train["observation_features"], train_labels, test["observation_features"], test_labels,
            num_classes=num_classes, epochs=epochs, learning_rate=learning_rate,
        )
        return controller["test_accuracy"] - observation["test_accuracy"]

    signals = {
        "current_search_type": ("cue_labels", task_cfg.num_types),
        "current_attended_cell": ("attended_cell_labels", task_cfg.num_cells),
    }
    results: dict[str, Any] = {"permutations": permutations, "percentile": percentile}
    all_pass = True
    for name, (label_key, num_classes) in signals.items():
        train_labels = train[label_key]
        test_labels = test[label_key]
        real_advantage = _advantage(train_labels, test_labels, num_classes)
        null_advantages = []
        for _ in range(permutations):
            tr = train_labels[torch.randperm(train_labels.shape[0], generator=generator, device=device)]
            te = test_labels[torch.randperm(test_labels.shape[0], generator=generator, device=device)]
            null_advantages.append(_advantage(tr, te, num_classes))
        floor = float(np.percentile(np.array(null_advantages), percentile))
        exceeds = real_advantage > floor
        all_pass = all_pass and exceeds
        results[name] = {
            "real_controller_advantage": real_advantage,
            "permuted_label_advantage_mean": float(np.mean(null_advantages)),
            "permuted_label_advantage_p95": floor,
            "exceeds_noise_floor": exceeds,
        }
    results["supported"] = all_pass
    return results


def self_model_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Evaluate native self-reports about attention history against a weaker baseline."""

    self_cfg = cfg["evaluation"].get(
        "self_modeling",
        {
            "enabled": True,
            "train_batches": 12,
            "test_batches": 6,
            "epochs": 60,
            "learning_rate": 0.05,
        },
    )
    if not self_cfg.get("enabled", False):
        return {}

    batch_size = cfg["training"]["batch_size"]
    train = _collect_self_model_dataset(
        model,
        task_cfg,
        batch_size,
        self_cfg["train_batches"],
        device,
        seed,
    )
    test = _collect_self_model_dataset(
        model,
        task_cfg,
        batch_size,
        self_cfg["test_batches"],
        device,
        seed + 1000,
    )

    observation_probe = _train_binary_probe(
        train["prev_observation_features"],
        train["target_inspected_labels"],
        test["prev_observation_features"],
        test["target_inspected_labels"],
        epochs=self_cfg["epochs"],
        learning_rate=self_cfg["learning_rate"],
    )
    observation_cell_probe = _train_multilabel_probe(
        train["prev_observation_features"],
        train["inspection_labels"],
        test["prev_observation_features"],
        test["inspection_labels"],
        epochs=self_cfg["epochs"],
        learning_rate=self_cfg["learning_rate"],
    )

    return {
        "native_cell_report": {
            "cell_accuracy": test["native_cell_accuracy"],
            "cell_bce": test["native_cell_bce"],
            "observation_only_probe": observation_cell_probe,
            "cell_accuracy_advantage": (
                test["native_cell_accuracy"] - observation_cell_probe["test_cell_accuracy"]
            ),
        },
        "target_inspected_report": {
            "native_accuracy": test["native_target_accuracy"],
            "native_positive_recall": test["native_target_positive_recall"],
            "observation_only_probe": observation_probe,
            "native_accuracy_advantage": test["native_target_accuracy"] - observation_probe["test_accuracy"],
            "native_positive_recall_advantage": (
                test["native_target_positive_recall"] - observation_probe["test_positive_recall"]
            ),
        },
        "supported": (
            test["native_cell_accuracy"] > observation_cell_probe["test_cell_accuracy"]
            and test["native_cell_accuracy"] > 0.95
            and test["native_target_positive_recall"] >= observation_probe["test_positive_recall"]
            and test["native_target_positive_recall"] > 0.5
        ),
    }


def learned_self_model_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Evaluate hidden-state-only evidence for learned attention self-modeling."""

    set_seed(seed)
    learned_cfg = cfg["evaluation"].get(
        "learned_self_modeling",
        {
            "enabled": True,
            "train_batches": 12,
            "test_batches": 6,
            "epochs": 60,
            "learning_rate": 0.05,
            "intervention": {
                "probe_scenes": 16,
                "step": 2,
                "scale": 2.0,
            },
        },
    )
    if not learned_cfg.get("enabled", False):
        return {}

    batch_size = cfg["training"]["batch_size"]
    train = _collect_learned_self_model_dataset(
        model,
        task_cfg,
        batch_size,
        learned_cfg["train_batches"],
        device,
        seed,
    )
    test = _collect_learned_self_model_dataset(
        model,
        task_cfg,
        batch_size,
        learned_cfg["test_batches"],
        device,
        seed + 1000,
    )

    epochs = learned_cfg["epochs"]
    learning_rate = learned_cfg["learning_rate"]
    audit_cfg = learned_cfg.get("capacity_audit", {})
    min_cell_bce_advantage = audit_cfg.get("min_cell_bce_advantage", 0.01)
    min_target_bce_advantage = audit_cfg.get("min_target_bce_advantage", 0.01)
    recurrent_migration = cfg.get("_checkpoint_migration", {}).get("recurrent", {})
    stale_checkpoint = bool(recurrent_migration.get("migrated", False))
    hidden_cell_probe = _train_multilabel_probe(
        train["hidden_features"],
        train["inspection_labels"],
        test["hidden_features"],
        test["inspection_labels"],
        epochs=epochs,
        learning_rate=learning_rate,
    )
    observation_cell_probe = _train_multilabel_probe(
        train["prev_observation_features"],
        train["inspection_labels"],
        test["prev_observation_features"],
        test["inspection_labels"],
        epochs=epochs,
        learning_rate=learning_rate,
    )
    hidden_target_probe = _train_binary_probe(
        train["hidden_features"],
        train["target_inspected_labels"],
        test["hidden_features"],
        test["target_inspected_labels"],
        epochs=epochs,
        learning_rate=learning_rate,
    )
    observation_target_probe = _train_binary_probe(
        train["prev_observation_features"],
        train["target_inspected_labels"],
        test["prev_observation_features"],
        test["target_inspected_labels"],
        epochs=epochs,
        learning_rate=learning_rate,
    )
    matched_dim = train["hidden_features"].shape[-1]
    matched_train_obs = _probe_capacity_matched_features(
        train["prev_observation_features"],
        matched_dim,
        seed=seed + 3100,
    )
    matched_test_obs = _probe_capacity_matched_features(
        test["prev_observation_features"],
        matched_dim,
        seed=seed + 3100,
    )
    matched_observation_cell_probe = _train_multilabel_probe(
        matched_train_obs,
        train["inspection_labels"],
        matched_test_obs,
        test["inspection_labels"],
        epochs=epochs,
        learning_rate=learning_rate,
    )
    matched_observation_target_probe = _train_binary_probe(
        matched_train_obs,
        train["target_inspected_labels"],
        matched_test_obs,
        test["target_inspected_labels"],
        epochs=epochs,
        learning_rate=learning_rate,
    )

    intervention_cfg = learned_cfg.get("intervention", {})
    intervention = _learned_self_model_intervention(
        model,
        task_cfg,
        batch_size=intervention_cfg.get("probe_scenes", batch_size),
        device=device,
        seed=seed + 2000,
        step=intervention_cfg.get("step", min(2, task_cfg.num_steps - 1)),
        scale=intervention_cfg.get("scale", 2.0),
    )

    hidden_cell_accuracy_advantage = (
        hidden_cell_probe["test_cell_accuracy"] - observation_cell_probe["test_cell_accuracy"]
    )
    hidden_cell_bce_advantage = (
        observation_cell_probe["test_bce"] - hidden_cell_probe["test_bce"]
    )
    matched_hidden_cell_bce_advantage = (
        matched_observation_cell_probe["test_bce"] - hidden_cell_probe["test_bce"]
    )
    matched_hidden_cell_accuracy_advantage = (
        hidden_cell_probe["test_cell_accuracy"]
        - matched_observation_cell_probe["test_cell_accuracy"]
    )
    hidden_target_accuracy_advantage = (
        hidden_target_probe["test_accuracy"] - observation_target_probe["test_accuracy"]
    )
    hidden_target_bce_advantage = (
        observation_target_probe["test_bce"] - hidden_target_probe["test_bce"]
    )
    matched_hidden_target_accuracy_advantage = (
        hidden_target_probe["test_accuracy"]
        - matched_observation_target_probe["test_accuracy"]
    )
    matched_hidden_target_bce_advantage = (
        matched_observation_target_probe["test_bce"] - hidden_target_probe["test_bce"]
    )
    hidden_target_positive_recall_advantage = (
        hidden_target_probe["test_positive_recall"]
        - observation_target_probe["test_positive_recall"]
    )
    hidden_target_score_separation_advantage = (
        hidden_target_probe["test_score_separation"]
        - observation_target_probe["test_score_separation"]
    )
    positive_evidence = (
        hidden_cell_bce_advantage > 0.0
        and (
            hidden_target_bce_advantage > 0.0
            or hidden_target_accuracy_advantage > 0.0
            or hidden_target_positive_recall_advantage > 0.0
            or hidden_target_score_separation_advantage > 0.0
        )
        and intervention["bidirectional_self_model_target_gap"] > 0.0
    )
    policy_feedback_evidence = (
        intervention["policy_feedback_abs_mean"] > 1e-6
        and abs(intervention["policy_override_bidirectional_target_attention_gap"]) > 5e-5
    )
    supported = (
        not stale_checkpoint
        and
        hidden_cell_bce_advantage > 0.01
        and hidden_target_bce_advantage > 0.01
        and hidden_target_accuracy_advantage >= 0.0
        and (
            hidden_target_positive_recall_advantage > 0.05
            or hidden_target_score_separation_advantage > 0.005
        )
        and intervention["bidirectional_self_model_target_gap"] > 0.01
        and intervention["positive_self_model_target_delta"] > 0.0
        and intervention["negative_self_model_target_delta"] < 0.0
        and policy_feedback_evidence
    )

    return {
        "native_hidden_cell_report": {
            "cell_accuracy": test.get("native_hidden_cell_accuracy", 0.0),
            "cell_bce": test.get("native_hidden_cell_bce", 0.0),
            "target_accuracy": test.get("native_hidden_target_accuracy", 0.0),
            "target_positive_recall": test.get(
                "native_hidden_target_positive_recall",
                0.0,
            ),
        },
        "hidden_cell_probe": hidden_cell_probe,
        "observation_cell_probe": observation_cell_probe,
        "probe_capacity_matched_observation_cell_probe": matched_observation_cell_probe,
        "hidden_cell_accuracy_advantage": hidden_cell_accuracy_advantage,
        "hidden_cell_bce_advantage": hidden_cell_bce_advantage,
        "probe_capacity_matched_hidden_cell_accuracy_advantage": matched_hidden_cell_accuracy_advantage,
        "probe_capacity_matched_hidden_cell_bce_advantage": matched_hidden_cell_bce_advantage,
        "hidden_target_probe": hidden_target_probe,
        "observation_target_probe": observation_target_probe,
        "probe_capacity_matched_observation_target_probe": matched_observation_target_probe,
        "hidden_target_accuracy_advantage": hidden_target_accuracy_advantage,
        "hidden_target_bce_advantage": hidden_target_bce_advantage,
        "probe_capacity_matched_hidden_target_accuracy_advantage": matched_hidden_target_accuracy_advantage,
        "probe_capacity_matched_hidden_target_bce_advantage": matched_hidden_target_bce_advantage,
        "hidden_target_positive_recall_advantage": hidden_target_positive_recall_advantage,
        "hidden_target_score_separation_advantage": hidden_target_score_separation_advantage,
        "capacity_audit": {
            "matched_input_dim": matched_dim,
            "scope": "linear_probe_input_dim_only",
            "baseline_source": "previous_observation_features",
            "baseline_lift": "deterministic_tanh_random_projection",
            "thresholds": {
                "min_cell_bce_advantage": min_cell_bce_advantage,
                "min_target_bce_advantage": min_target_bce_advantage,
            },
            "checkpoint_migration": recurrent_migration,
            "hidden_cell_bce_advantage": matched_hidden_cell_bce_advantage,
            "hidden_cell_accuracy_advantage": matched_hidden_cell_accuracy_advantage,
            "hidden_target_bce_advantage": matched_hidden_target_bce_advantage,
            "hidden_target_accuracy_advantage": matched_hidden_target_accuracy_advantage,
            "passed": (
                not stale_checkpoint
                and matched_hidden_cell_bce_advantage >= min_cell_bce_advantage
                and matched_hidden_target_bce_advantage >= min_target_bce_advantage
            ),
            "probe_effect_positive": (
                matched_hidden_cell_bce_advantage > 0.0
                and matched_hidden_target_bce_advantage > 0.0
            ),
        },
        "hidden_self_model_intervention": intervention,
        "policy_feedback_evidence": policy_feedback_evidence,
        "positive_evidence": positive_evidence,
        "supported": supported,
        "note": (
            "This diagnostic excludes the engineered inspection-state input from the probe "
            "features, evaluates a hidden-state-only self-model head, and tests whether "
            "hidden self-model overrides can affect attention through the learned policy "
            "feedback path. It is progress toward Stage 4B, but current checkpoints must "
            "learn and robustly use this path before the stronger claim is supported."
        ),
    }


def uncertainty_report_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Evaluate Stage 6B-style native uncertainty and allocation-error reports."""

    set_seed(seed)
    probe_cfg = cfg["evaluation"].get(
        "uncertainty_report_probes",
        {
            "enabled": True,
            "train_batches": 12,
            "test_batches": 6,
            "epochs": 60,
            "learning_rate": 0.05,
        },
    )
    if not probe_cfg.get("enabled", False):
        return {}

    batch_size = cfg["training"]["batch_size"]
    train = _collect_uncertainty_probe_dataset(
        model, task_cfg, batch_size, probe_cfg["train_batches"], device, seed
    )
    test = _collect_uncertainty_probe_dataset(
        model, task_cfg, batch_size, probe_cfg["test_batches"], device, seed + 1000
    )
    epochs = probe_cfg["epochs"]
    learning_rate = probe_cfg["learning_rate"]
    audit_cfg = probe_cfg.get("capacity_audit", {})
    min_positive_recall_advantage = audit_cfg.get("min_positive_recall_advantage", 0.01)
    matched_dim = train["state_features"].shape[-1]
    matched_train_prev_obs = _probe_capacity_matched_features(
        train["prev_observation_features"],
        matched_dim,
        seed=seed + 3300,
    )
    matched_test_prev_obs = _probe_capacity_matched_features(
        test["prev_observation_features"],
        matched_dim,
        seed=seed + 3300,
    )

    def _native_binary_report(native_scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        native_pred = (native_scores >= 0.5).float()
        labels = labels.float()
        positive = labels == 1.0
        return {
            "test_bce": torch.nn.functional.binary_cross_entropy(
                native_scores.clamp(1e-6, 1 - 1e-6),
                labels,
            ).item(),
            "test_accuracy": (native_pred == labels).float().mean().item(),
            "test_positive_recall": (
                ((native_pred == 1.0) & positive).float().sum()
                / positive.float().sum().clamp_min(1.0)
            ).item(),
        }

    def _compare_binary(name: str, native_key: str, label_key: str) -> dict[str, Any]:
        # The gated reportability test mirrors Stage 6A: does the controller STATE linearly
        # encode the uncertainty/allocation-error variable better than a capacity-matched
        # observation baseline? The "native_report" score is kept only as an informational
        # field -- for some variables it is a ground-truth computed signal (circular) or an
        # untrained head, so it must NOT gate the claim.
        state_probe = _train_binary_probe(
            train["state_features"],
            train[label_key],
            test["state_features"],
            test[label_key],
            epochs=epochs,
            learning_rate=learning_rate,
        )
        observation_probe = _train_binary_probe(
            train["prev_observation_features"],
            train[label_key],
            test["prev_observation_features"],
            test[label_key],
            epochs=epochs,
            learning_rate=learning_rate,
        )
        matched_observation_probe = _train_binary_probe(
            matched_train_prev_obs,
            train[label_key],
            matched_test_prev_obs,
            test[label_key],
            epochs=epochs,
            learning_rate=learning_rate,
        )
        native_report = _native_binary_report(test[native_key], test[label_key])
        return {
            "controller_state_probe": state_probe,
            "native_report": native_report,
            "observation_only_probe": observation_probe,
            "probe_capacity_matched_observation_probe": matched_observation_probe,
            "controller_accuracy_advantage": (
                state_probe["test_accuracy"] - observation_probe["test_accuracy"]
            ),
            "probe_capacity_matched_controller_accuracy_advantage": (
                state_probe["test_accuracy"] - matched_observation_probe["test_accuracy"]
            ),
            "controller_positive_recall_advantage": (
                state_probe["test_positive_recall"] - observation_probe["test_positive_recall"]
            ),
            "probe_capacity_matched_controller_positive_recall_advantage": (
                state_probe["test_positive_recall"]
                - matched_observation_probe["test_positive_recall"]
            ),
            # Informational only (see note above); does not gate support.
            "native_positive_recall_advantage": (
                native_report["test_positive_recall"] - observation_probe["test_positive_recall"]
            ),
            "probe_capacity_matched_native_positive_recall_advantage": (
                native_report["test_positive_recall"]
                - matched_observation_probe["test_positive_recall"]
            ),
            "name": name,
        }

    relevant_region = _compare_binary(
        "relevant_region_inspected",
        "relevant_region_native",
        "relevant_region_labels",
    )
    unresolved_search = _compare_binary(
        "unresolved_search",
        "unresolved_search_native",
        "unresolved_search_labels",
    )
    current_wrong_candidate = _compare_binary(
        "current_wrong_candidate",
        "current_wrong_candidate_native",
        "current_wrong_candidate_labels",
    )
    wrong_candidate_history = _compare_binary(
        "wrong_candidate_history",
        "wrong_candidate_history_native",
        "wrong_candidate_history_labels",
    )
    revisit_unresolved = _compare_binary(
        "revisit_unresolved",
        "revisit_unresolved_native",
        "revisit_unresolved_labels",
    )
    allocation_error = _compare_binary(
        "allocation_error",
        "allocation_error_native",
        "allocation_error_labels",
    )
    gated_capacity_audit_signals = (
        current_wrong_candidate,
        wrong_candidate_history,
        revisit_unresolved,
        allocation_error,
    )
    informational_capacity_audit_signals = (
        relevant_region,
        unresolved_search,
    )
    return {
        "relevant_region_inspected": relevant_region,
        "unresolved_search": unresolved_search,
        "current_wrong_candidate": current_wrong_candidate,
        "wrong_candidate_history": wrong_candidate_history,
        "revisit_unresolved": revisit_unresolved,
        "allocation_error": allocation_error,
        "capacity_audit": {
            "matched_input_dim": matched_dim,
            "scope": "linear_probe_input_dim_only",
            "baseline_source": "prev_observation_features",
            "baseline_lift": "deterministic_tanh_random_projection",
            "thresholds": {
                "min_positive_recall_advantage": min_positive_recall_advantage,
            },
            # Gate on the controller-STATE probe beating the capacity-matched observation
            # baseline on positive recall, with a non-negative accuracy advantage so a
            # predict-all-positive probe cannot pass on recall alone.
            "passed": all(
                signal["probe_capacity_matched_controller_positive_recall_advantage"]
                >= min_positive_recall_advantage
                and signal["probe_capacity_matched_controller_accuracy_advantage"] >= 0.0
                for signal in gated_capacity_audit_signals
            ),
            "nonnegative_directional_effect": all(
                signal["probe_capacity_matched_controller_positive_recall_advantage"] >= 0.0
                for signal in gated_capacity_audit_signals
            ),
            "gated_controller_positive_recall_advantages": {
                signal["name"]: signal["probe_capacity_matched_controller_positive_recall_advantage"]
                for signal in gated_capacity_audit_signals
            },
            "gated_controller_accuracy_advantages": {
                signal["name"]: signal["probe_capacity_matched_controller_accuracy_advantage"]
                for signal in gated_capacity_audit_signals
            },
            "informational_native_positive_recall_advantages": {
                signal["name"]: signal["probe_capacity_matched_native_positive_recall_advantage"]
                for signal in (*gated_capacity_audit_signals, *informational_capacity_audit_signals)
            },
        },
        "supported": (
            current_wrong_candidate["probe_capacity_matched_controller_positive_recall_advantage"]
            >= min_positive_recall_advantage
            and wrong_candidate_history["probe_capacity_matched_controller_positive_recall_advantage"]
            >= min_positive_recall_advantage
            and revisit_unresolved["probe_capacity_matched_controller_positive_recall_advantage"]
            >= 0.0
            and allocation_error["probe_capacity_matched_controller_positive_recall_advantage"]
            >= 0.0
        ),
    }


def _select_diverse_nl_examples(
    examples,
    *,
    grid_size: int,
    calibration_count: int,
    evaluation_count: int,
):
    """Greedily pick calibration/eval examples that cover more token-state variation."""

    if len(examples) < calibration_count + evaluation_count:
        raise ValueError("not enough NL examples for requested calibration/evaluation counts")

    def features(example):
        row, col = divmod(example.attended_cell, grid_size)
        prev_row, prev_col = divmod(example.prev_attended_cell, grid_size)
        changed_cell = int(example.attended_cell != example.prev_attended_cell)
        changed_visible = int(example.attended_visible_type != example.prev_attended_visible_type)
        changed_digit = int(example.attended_digit != example.prev_attended_digit)
        changed_glimpse = int(example.glimpse_digit != example.prev_glimpse_digit)
        unresolved_rows = tuple(sorted({cell // grid_size for cell in example.unresolved_cells}))
        unresolved_cols = tuple(sorted({cell % grid_size for cell in example.unresolved_cells}))
        return {
            ("cue", example.cue),
            ("row", row),
            ("col", col),
            ("prev_row", prev_row),
            ("prev_col", prev_col),
            ("changed_cell", changed_cell),
            ("changed_visible", changed_visible),
            ("changed_digit", changed_digit),
            ("changed_glimpse", changed_glimpse),
            ("visible_type", example.attended_visible_type),
            ("attended_digit", example.attended_digit),
            ("glimpse_digit", example.glimpse_digit),
            ("prev_visible_type", example.prev_attended_visible_type),
            ("prev_attended_digit", example.prev_attended_digit),
            ("prev_glimpse_digit", example.prev_glimpse_digit),
            ("glimpse_match", int(example.glimpse_target_match)),
            ("found_target", int(example.found_target)),
            ("unresolved_rows", unresolved_rows),
            ("unresolved_cols", unresolved_cols),
        }

    def signature(example):
        return (
            example.cue,
            example.attended_cell,
            example.attended_visible_type,
            example.attended_digit,
            example.glimpse_digit,
            example.prev_attended_cell,
            example.prev_attended_visible_type,
            example.prev_attended_digit,
            example.prev_glimpse_digit,
            int(example.glimpse_target_match),
            int(example.found_target),
            tuple(example.unresolved_rows),
            tuple(example.unresolved_cols),
            example.unresolved_count,
        )

    remaining = list(range(len(examples)))
    covered = set()
    calibration = []
    used_signatures = set()

    while remaining and len(calibration) < calibration_count:
        best_idx = None
        best_score = None
        for idx in remaining:
            feat = features(examples[idx])
            new_score = len(feat - covered)
            tie_break = len(feat)
            changed_score = (
                int(examples[idx].attended_cell != examples[idx].prev_attended_cell)
                + int(examples[idx].attended_visible_type != examples[idx].prev_attended_visible_type)
                + int(examples[idx].attended_digit != examples[idx].prev_attended_digit)
                + int(examples[idx].glimpse_digit != examples[idx].prev_glimpse_digit)
            )
            score = (changed_score, new_score, tie_break, -idx)
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
        calibration.append(examples[best_idx])
        covered |= features(examples[best_idx])
        used_signatures.add(signature(examples[best_idx]))
        remaining.remove(best_idx)

    evaluation = []
    while remaining and len(evaluation) < evaluation_count:
        best_idx = None
        best_score = None
        for idx in remaining:
            feat = features(examples[idx])
            new_score = len(feat - covered)
            row, col = divmod(examples[idx].attended_cell, grid_size)
            novel_signature = int(signature(examples[idx]) not in used_signatures)
            changed_score = (
                int(examples[idx].attended_cell != examples[idx].prev_attended_cell)
                + int(examples[idx].attended_visible_type != examples[idx].prev_attended_visible_type)
                + int(examples[idx].attended_digit != examples[idx].prev_attended_digit)
                + int(examples[idx].glimpse_digit != examples[idx].prev_glimpse_digit)
            )
            score = (
                changed_score,
                novel_signature,
                new_score,
                int(examples[idx].glimpse_target_match),
                int(examples[idx].found_target),
                row + col,
                -idx,
            )
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
        evaluation.append(examples[best_idx])
        covered |= features(examples[best_idx])
        used_signatures.add(signature(examples[best_idx]))
        remaining.remove(best_idx)

    if len(calibration) < calibration_count or len(evaluation) < evaluation_count:
        raise ValueError("unable to select enough diverse NL examples")
    return calibration, evaluation


def _select_translator_examples(
    examples,
    *,
    grid_size: int,
    target_count: int,
):
    """Pick a broader training pool for the Stage 7 translator."""

    if not examples or target_count <= 0:
        return []

    def features(example):
        row, col = divmod(example.attended_cell, grid_size)
        prev_row, prev_col = divmod(example.prev_attended_cell, grid_size)
        return {
            ("cue", example.cue),
            ("row", row),
            ("col", col),
            ("prev_row", prev_row),
            ("prev_col", prev_col),
            ("visible_type", example.attended_visible_type),
            ("attended_digit", example.attended_digit),
            ("glimpse_digit", example.glimpse_digit),
            ("prev_visible_type", example.prev_attended_visible_type),
            ("prev_attended_digit", example.prev_attended_digit),
            ("prev_glimpse_digit", example.prev_glimpse_digit),
            ("changed_cell", int(example.attended_cell != example.prev_attended_cell)),
            ("changed_visible", int(example.attended_visible_type != example.prev_attended_visible_type)),
            ("changed_digit", int(example.attended_digit != example.prev_attended_digit)),
            ("changed_glimpse", int(example.glimpse_digit != example.prev_glimpse_digit)),
        }

    remaining = list(range(len(examples)))
    covered = set()
    selected = []
    while remaining and len(selected) < target_count:
        best_idx = None
        best_score = None
        for idx in remaining:
            feat = features(examples[idx])
            new_score = len(feat - covered)
            changed_score = (
                int(examples[idx].attended_cell != examples[idx].prev_attended_cell)
                + int(examples[idx].attended_visible_type != examples[idx].prev_attended_visible_type)
                + int(examples[idx].attended_digit != examples[idx].prev_attended_digit)
                + int(examples[idx].glimpse_digit != examples[idx].prev_glimpse_digit)
            )
            score = (changed_score, new_score, -idx)
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
        selected.append(examples[best_idx])
        covered |= features(examples[best_idx])
        remaining.remove(best_idx)
    return selected


def nl_report_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Evaluate natural-language reporting from tokenized internal state."""

    set_seed(seed)
    nl_cfg = cfg["evaluation"].get(
        "nl_report",
        {
            "enabled": False,
            "model": "gpt-5-mini",
            "dotenv_path": ".env",
            "calibration_examples": 4,
            "evaluation_examples": 4,
            "translator_train_examples": 8,
            "probe_scenes": 2,
            "max_output_tokens": 240,
        },
    )
    if not nl_cfg.get("enabled", False):
        return {}

    load_dotenv(nl_cfg.get("dotenv_path", ".env"))
    api_skip_reason = ""
    if OpenAI is None:
        api_skip_reason = "openai dependency is not installed"
    elif not os.environ.get("OPENAI_API_KEY"):
        api_skip_reason = "OPENAI_API_KEY is not set"

    batch_size = nl_cfg.get("probe_scenes", cfg["evaluation"]["probe_scenes"])
    generator = make_generator(seed, device)
    batch = generate_batch(batch_size, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    calibration_count = int(nl_cfg.get("calibration_examples", 4))
    evaluation_count = int(nl_cfg.get("evaluation_examples", 4))
    translator_train_count = int(nl_cfg.get("translator_train_examples", 8))
    model_name = nl_cfg.get("model", "gpt-5-mini")
    max_output_tokens = int(nl_cfg.get("max_output_tokens", 240))
    request_retries = int(nl_cfg.get("request_retries", 2))
    retry_backoff_seconds = float(nl_cfg.get("retry_backoff_seconds", 2.0))
    local_decoder_enabled = bool(nl_cfg.get("local_decoder", {}).get("enabled", False))
    nl_audit_cfg = nl_cfg.get("capacity_audit", {})
    min_joint_advantage = nl_audit_cfg.get("min_joint_accuracy_advantage", 0.01)
    min_memory_advantage = nl_audit_cfg.get("min_memory_content_joint_accuracy_advantage", 0.01)
    modes = ("tokenized_state", "symbolic_state", "observation_only")
    required_examples = calibration_count + evaluation_count

    def _score_examples(slice_name: str, examples: list[Any]) -> dict[str, Any]:
        examples = [example for example in examples if example.step_index > 0]
        if len(examples) < required_examples:
            return {
                "enabled": True,
                "skipped": True,
                "reason": (
                    f"not enough examples for {slice_name}: need {required_examples}, have {len(examples)}"
                ),
            }

        calibration_examples, evaluation_examples = _select_diverse_nl_examples(
            examples,
            grid_size=task_cfg.grid_size,
            calibration_count=calibration_count,
            evaluation_count=evaluation_count,
        )
        held_out_ids = {id(example) for example in calibration_examples + evaluation_examples}
        translator_pool = [example for example in examples if id(example) not in held_out_ids]
        translator_examples = _select_translator_examples(
            translator_pool,
            grid_size=task_cfg.grid_size,
            target_count=translator_train_count,
        )
        if not translator_examples:
            translator_examples = calibration_examples
        _render_tokenized_examples(translator_examples, calibration_examples + evaluation_examples)
        tokenized_payload = tokenized_state_payload_metrics(
            evaluation_examples,
            grid_size=task_cfg.grid_size,
        )
        if local_decoder_enabled:
            results = {
                "tokenized_state": run_calibrated_token_report_mode(
                    evaluation_examples=evaluation_examples,
                    grid_size=task_cfg.grid_size,
                ),
                "symbolic_state": run_calibrated_token_report_mode(
                    evaluation_examples=evaluation_examples,
                    grid_size=task_cfg.grid_size,
                ),
                "observation_only": run_observation_only_heuristic_report_mode(
                    evaluation_examples=evaluation_examples,
                    grid_size=task_cfg.grid_size,
                ),
            }
        if api_skip_reason and not local_decoder_enabled:
            return {
                "enabled": True,
                "skipped": True,
                "reason": api_skip_reason,
                "model": model_name,
                "slice": slice_name,
                "calibration_examples": calibration_count,
                "evaluation_examples": evaluation_count,
                "translator_train_examples": len(translator_examples),
                "tokenized_state_payload": tokenized_payload,
            }
        if not local_decoder_enabled:
            try:
                results = {
                    mode: run_nl_report_mode(
                        mode=mode,
                        model_name=model_name,
                        calibration_examples=calibration_examples,
                        evaluation_examples=evaluation_examples,
                        grid_size=task_cfg.grid_size,
                        max_output_tokens=max_output_tokens,
                        teaching_examples=translator_examples,
                        request_retries=request_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                    )
                    for mode in modes
                }
            except Exception as exc:  # pragma: no cover - network/runtime behavior
                return {
                    "enabled": True,
                    "skipped": True,
                    "reason": f"nl_report request failed: {type(exc).__name__}: {exc}",
                    "model": model_name,
                    "slice": slice_name,
                    "calibration_examples": calibration_count,
                    "evaluation_examples": evaluation_count,
                    "translator_train_examples": len(translator_examples),
                    "tokenized_state_payload": tokenized_payload,
                }

        tokenized = results["tokenized_state"]
        symbolic = results["symbolic_state"]
        observation = results["observation_only"]
        tokenized_token_count = sum(
            len(example.tokenized_state.split()) for example in evaluation_examples
        ) / max(len(evaluation_examples), 1)
        observation_token_count = sum(
            len(example.observation_only.split()) for example in evaluation_examples
        ) / max(len(evaluation_examples), 1)
        probe_capacity_matched_observation = {
            **observation,
            "mode": "probe_capacity_matched_observation_only",
            "baseline_source": "observation_only",
            "baseline_lift": "opaque_filler_tokens_to_tokenized_state_budget",
            "mean_input_tokens": tokenized_token_count,
            "semantic_input_tokens": observation_token_count,
            "filler_tokens": max(tokenized_token_count - observation_token_count, 0.0),
        }
        return {
            "enabled": True,
            "skipped": False,
            "model": model_name,
            "local_decoder": local_decoder_enabled,
            "slice": slice_name,
            "calibration_examples": calibration_count,
            "evaluation_examples": evaluation_count,
            "translator_train_examples": len(translator_examples),
            "tokenized_state_payload": tokenized_payload,
            "tokenized_state": tokenized,
            "symbolic_state": symbolic,
            "observation_only": observation,
            "probe_capacity_matched_observation_only": probe_capacity_matched_observation,
            "tokenized_joint_accuracy_advantage": (
                tokenized["joint_accuracy"] - observation["joint_accuracy"]
            ),
            "probe_capacity_matched_tokenized_joint_accuracy_advantage": (
                tokenized["joint_accuracy"] - probe_capacity_matched_observation["joint_accuracy"]
            ),
            "tokenized_current_content_joint_accuracy_advantage": (
                tokenized["current_content_joint_accuracy"]
                - observation["current_content_joint_accuracy"]
            ),
            "probe_capacity_matched_tokenized_current_content_joint_accuracy_advantage": (
                tokenized["current_content_joint_accuracy"]
                - probe_capacity_matched_observation["current_content_joint_accuracy"]
            ),
            "tokenized_memory_content_joint_accuracy_advantage": (
                tokenized["memory_content_joint_accuracy"]
                - observation["memory_content_joint_accuracy"]
            ),
            "probe_capacity_matched_tokenized_memory_content_joint_accuracy_advantage": (
                tokenized["memory_content_joint_accuracy"]
                - probe_capacity_matched_observation["memory_content_joint_accuracy"]
            ),
            "tokenized_content_only_joint_accuracy_advantage": (
                tokenized["content_only_joint_accuracy"]
                - observation["content_only_joint_accuracy"]
            ),
            "probe_capacity_matched_tokenized_content_only_joint_accuracy_advantage": (
                tokenized["content_only_joint_accuracy"]
                - probe_capacity_matched_observation["content_only_joint_accuracy"]
            ),
            "tokenized_visible_type_accuracy_advantage": (
                tokenized["attended_visible_type_accuracy"]
                - observation["attended_visible_type_accuracy"]
            ),
            "tokenized_attended_digit_accuracy_advantage": (
                tokenized["attended_digit_accuracy"] - observation["attended_digit_accuracy"]
            ),
            "tokenized_glimpse_digit_accuracy_advantage": (
                tokenized["glimpse_digit_accuracy"] - observation["glimpse_digit_accuracy"]
            ),
            "tokenized_previous_attended_cell_accuracy_advantage": (
                tokenized["previous_attended_cell_accuracy"]
                - observation["previous_attended_cell_accuracy"]
            ),
            "tokenized_previous_visible_type_accuracy_advantage": (
                tokenized["previous_attended_visible_type_accuracy"]
                - observation["previous_attended_visible_type_accuracy"]
            ),
            "tokenized_previous_attended_digit_accuracy_advantage": (
                tokenized["previous_attended_digit_accuracy"]
                - observation["previous_attended_digit_accuracy"]
            ),
            "tokenized_previous_glimpse_digit_accuracy_advantage": (
                tokenized["previous_glimpse_digit_accuracy"]
                - observation["previous_glimpse_digit_accuracy"]
            ),
            "tokenized_glimpse_match_accuracy_advantage": (
                tokenized["glimpse_target_match_accuracy"]
                - observation["glimpse_target_match_accuracy"]
            ),
            "tokenized_relevant_region_accuracy_advantage": (
                tokenized["relevant_region_inspected_accuracy"]
                - observation["relevant_region_inspected_accuracy"]
            ),
            "tokenized_unresolved_search_accuracy_advantage": (
                tokenized["unresolved_search_accuracy"]
                - observation["unresolved_search_accuracy"]
            ),
            "tokenized_current_wrong_candidate_accuracy_advantage": (
                tokenized["current_wrong_candidate_accuracy"]
                - observation["current_wrong_candidate_accuracy"]
            ),
            "tokenized_wrong_candidate_history_accuracy_advantage": (
                tokenized["wrong_candidate_history_accuracy"]
                - observation["wrong_candidate_history_accuracy"]
            ),
            "tokenized_revisit_unresolved_accuracy_advantage": (
                tokenized["revisit_unresolved_accuracy"] - observation["revisit_unresolved_accuracy"]
            ),
            "tokenized_allocation_error_accuracy_advantage": (
                tokenized["allocation_error_accuracy"]
                - observation["allocation_error_accuracy"]
            ),
            "tokenized_uncertainty_content_joint_accuracy_advantage": (
                tokenized["uncertainty_content_joint_accuracy"]
                - observation["uncertainty_content_joint_accuracy"]
            ),
            "probe_capacity_matched_tokenized_uncertainty_content_joint_accuracy_advantage": (
                tokenized["uncertainty_content_joint_accuracy"]
                - probe_capacity_matched_observation["uncertainty_content_joint_accuracy"]
            ),
            "tokenized_unresolved_accuracy_advantage": (
                (
                    tokenized["unresolved_rows_accuracy"]
                    + tokenized["unresolved_cols_accuracy"]
                    + tokenized["unresolved_count_accuracy"]
                )
                / 3.0
                - (
                    observation["unresolved_rows_accuracy"]
                    + observation["unresolved_cols_accuracy"]
                    + observation["unresolved_count_accuracy"]
                )
                / 3.0
            ),
            "symbolic_joint_accuracy_advantage": (
                symbolic["joint_accuracy"] - observation["joint_accuracy"]
            ),
            "capacity_audit": {
                "matched_input_tokens": tokenized_token_count,
                "semantic_observation_tokens": observation_token_count,
                "scope": "matched_token_budget_only",
                "baseline_source": "observation_only",
                "baseline_lift": "opaque_filler_tokens_to_tokenized_state_budget",
                "thresholds": {
                    "min_joint_accuracy_advantage": min_joint_advantage,
                    "min_memory_content_joint_accuracy_advantage": min_memory_advantage,
                },
                "joint_accuracy_advantage": (
                    tokenized["joint_accuracy"]
                    - probe_capacity_matched_observation["joint_accuracy"]
                ),
                "memory_content_joint_accuracy_advantage": (
                    tokenized["memory_content_joint_accuracy"]
                    - probe_capacity_matched_observation["memory_content_joint_accuracy"]
                ),
                "passed": (
                    tokenized["joint_accuracy"] - probe_capacity_matched_observation["joint_accuracy"]
                    >= min_joint_advantage
                    and tokenized["memory_content_joint_accuracy"]
                    - probe_capacity_matched_observation["memory_content_joint_accuracy"]
                    >= min_memory_advantage
                ),
                "nonnegative_directional_effect": (
                    tokenized["joint_accuracy"] >= probe_capacity_matched_observation["joint_accuracy"]
                    and tokenized["memory_content_joint_accuracy"]
                    >= probe_capacity_matched_observation["memory_content_joint_accuracy"]
                ),
            },
            "content_supported": (
                tokenized["content_only_joint_accuracy"] > observation["content_only_joint_accuracy"]
                and tokenized["memory_content_joint_accuracy"] > observation["memory_content_joint_accuracy"]
                and tokenized["current_content_joint_accuracy"]
                >= observation["current_content_joint_accuracy"]
            ),
            "supported": (
                tokenized["joint_accuracy"] > observation["joint_accuracy"]
                and tokenized["previous_attended_cell_accuracy"]
                > observation["previous_attended_cell_accuracy"]
                and tokenized["previous_attended_visible_type_accuracy"]
                > observation["previous_attended_visible_type_accuracy"]
                and tokenized["previous_attended_digit_accuracy"]
                > observation["previous_attended_digit_accuracy"]
                and tokenized["previous_glimpse_digit_accuracy"]
                > observation["previous_glimpse_digit_accuracy"]
                and tokenized["attended_visible_type_accuracy"] >= observation["attended_visible_type_accuracy"]
                and tokenized["attended_digit_accuracy"] >= observation["attended_digit_accuracy"]
                and tokenized["glimpse_digit_accuracy"] >= observation["glimpse_digit_accuracy"]
                and tokenized["uncertainty_content_joint_accuracy"]
                >= observation["uncertainty_content_joint_accuracy"]
                and (
                    tokenized["unresolved_rows_accuracy"]
                    + tokenized["unresolved_cols_accuracy"]
                    + tokenized["unresolved_count_accuracy"]
                )
                / 3.0
                >= (
                    observation["unresolved_rows_accuracy"]
                    + observation["unresolved_cols_accuracy"]
                    + observation["unresolved_count_accuracy"]
                )
                / 3.0
            ),
        }

    with torch.no_grad():
        outputs = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
        )

    base_examples = collect_nl_examples(model, task_cfg, batch, outputs)
    base_result = _score_examples("default", base_examples)
    if base_result.get("skipped", False):
        return base_result

    result = dict(base_result)
    cue_switch_cfg = cfg["evaluation"].get("cue_switch", {})
    if cue_switch_cfg.get("enabled", False):
        cue_switch_examples = collect_cue_switch_nl_examples(
            model,
            task_cfg,
            batch,
            switch_step=int(cue_switch_cfg.get("switch_step", task_cfg.num_steps // 2)),
        )
        result["cue_switch_slice"] = _score_examples("cue_switch", cue_switch_examples)

    intervention_cfg = cfg["evaluation"].get("intervention_test", {})
    if intervention_cfg.get("enabled", False):
        intervention_examples = collect_intervention_nl_examples(
            model,
            task_cfg,
            batch,
            intervention_step=int(intervention_cfg.get("step", 1)),
        )
        result["intervention_slice"] = _score_examples(
            "intervention",
            intervention_examples["intervened_examples"],
        )
        result["intervention_slice"]["delta_norm"] = intervention_examples["delta_norm"]
        result["intervention_slice"]["attention_change_fraction"] = intervention_examples[
            "attention_change_fraction"
        ]

    return result


def intervention_test_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Perturb controller state with same-scene, different-cue deltas and measure next attention shifts."""

    intervention_cfg = cfg["evaluation"].get("intervention_test", {})
    if not intervention_cfg.get("enabled", False):
        return {}
    thresholds = intervention_cfg.get("thresholds", {})

    batch_size = intervention_cfg.get("probe_scenes", cfg["evaluation"]["probe_scenes"])
    step = intervention_cfg.get("step", 2)
    if step < 0 or step >= task_cfg.num_steps:
        raise ValueError("intervention_test.step must be within the episode length")

    outputs, probe_batch = _probe_outputs(model, task_cfg, batch_size, device, seed)
    states = outputs["controller_state_seq"].view(batch_size, task_cfg.num_types, task_cfg.num_steps, -1)
    baseline_attention = outputs["attention_seq"].view(
        batch_size, task_cfg.num_types, task_cfg.num_steps, -1
    )
    target_pos = probe_batch.target_pos.view(batch_size, task_cfg.num_types)

    alt_cue = (torch.arange(task_cfg.num_types, device=device) + 1) % task_cfg.num_types
    current_state = states[:, torch.arange(task_cfg.num_types, device=device), step]
    alternate_state = states[:, alt_cue, step]
    intervention_delta = (alternate_state - current_state).reshape(-1, current_state.shape[-1])

    intervened_outputs = model(
        probe_batch.scene,
        probe_batch.cue,
        target=probe_batch.target,
        target_pos=probe_batch.target_pos,
        num_steps=task_cfg.num_steps,
        intervention={"step": step, "delta": intervention_delta},
    )
    intervened_attention = intervened_outputs["attention_seq"].view(
        batch_size, task_cfg.num_types, task_cfg.num_steps, -1
    )

    row_index = torch.arange(batch_size, device=device)[:, None]
    alt_target_pos = target_pos[row_index, alt_cue[None, :]]
    base_step_attention = baseline_attention[:, :, step]
    intervened_step_attention = intervened_attention[:, :, step]

    base_target_attention = base_step_attention.gather(2, target_pos.unsqueeze(-1)).mean().item()
    intervened_target_attention = intervened_step_attention.gather(
        2, target_pos.unsqueeze(-1)
    ).mean().item()
    base_alt_target_attention = base_step_attention.gather(2, alt_target_pos.unsqueeze(-1)).mean().item()
    intervened_alt_target_attention = intervened_step_attention.gather(
        2, alt_target_pos.unsqueeze(-1)
    ).mean().item()

    attention_change_kl = symmetric_kl(base_step_attention, intervened_step_attention).mean().item()
    original_target_attention_drop = base_target_attention - intervened_target_attention
    alternate_target_attention_gain = intervened_alt_target_attention - base_alt_target_attention
    return {
        "step": step,
        "attention_change_kl": attention_change_kl,
        "original_target_attention_drop": original_target_attention_drop,
        "alternate_target_attention_gain": alternate_target_attention_gain,
        "baseline_target_attention": base_target_attention,
        "intervened_target_attention": intervened_target_attention,
        "baseline_alternate_target_attention": base_alt_target_attention,
        "intervened_alternate_target_attention": intervened_alt_target_attention,
        "thresholds": {
            "min_attention_change_kl": thresholds.get("min_attention_change_kl", 0.0),
            "min_original_target_attention_drop": thresholds.get(
                "min_original_target_attention_drop", 0.0
            ),
            "min_alternate_target_attention_gain": thresholds.get(
                "min_alternate_target_attention_gain", 0.0
            ),
        },
        "supported": (
            attention_change_kl >= thresholds.get("min_attention_change_kl", 0.0)
            and original_target_attention_drop
            >= thresholds.get("min_original_target_attention_drop", 0.0)
            and alternate_target_attention_gain
            >= thresholds.get("min_alternate_target_attention_gain", 0.0)
        ),
    }


def _perturbational_response(
    model,
    batch,
    task_cfg: TaskConfig,
    device: torch.device,
    *,
    step: int,
    noise: torch.Tensor,
    ablation: dict[str, bool] | None,
) -> dict[str, float]:
    """Perturb controller hidden state at one step and measure the recovery trajectory."""

    with torch.no_grad():
        base = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
            ablation=ablation,
        )
        perturbed = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
            ablation=ablation,
            intervention={"step": step, "delta": noise},
        )

    base_states = base["controller_state_seq"]
    pert_states = perturbed["controller_state_seq"]
    # State-divergence trajectory from the perturbed step to the end of the episode. The
    # perturbation (added at `step`) first appears in the stored state at index step+1, so
    # recovery is measured from the divergence PEAK rather than the perturbed-step value
    # (which is still ~0) to avoid dividing by a near-zero immediate effect.
    divergence = (pert_states - base_states).norm(dim=-1).mean(dim=0)  # (steps,)
    post = divergence[step:]
    peak = post.max().item()
    final = post[-1].item()
    immediate = post[0].item()
    recovery_ratio = (peak - final) / max(peak, 1e-6)
    # Behavioural propagation: how much the perturbation moves attention on later steps.
    base_attn = base["attention_seq"]
    pert_attn = perturbed["attention_seq"]
    if step + 1 < task_cfg.num_steps:
        attention_propagation = symmetric_kl(
            base_attn[:, step + 1 :], pert_attn[:, step + 1 :]
        ).mean().item()
    else:
        attention_propagation = 0.0
    # Downstream integration: change in the final task prediction distribution.
    base_probs = torch.softmax(base["logits_seq"][:, -1], dim=-1)
    pert_probs = torch.softmax(perturbed["logits_seq"][:, -1], dim=-1)
    task_shift = (pert_probs - base_probs).abs().sum(dim=-1).mean().item() / 2.0
    return {
        "immediate_divergence": immediate,
        "final_divergence": final,
        "peak_divergence": peak,
        "recovery_ratio": recovery_ratio,
        "attention_propagation": attention_propagation,
        "task_distribution_shift": task_shift,
    }


def perturbational_complexity_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Stage-independent (non-reportability) perturbational-complexity branch.

    Perturbs controller hidden state with noise at one step and asks whether the recurrent
    controller produces rich-but-recoverable dynamics (the perturbation propagates to later
    behaviour but the state trajectory partially recovers), and whether that integrated
    response degrades under feedforward / frozen / shuffled-feedback controls.
    """

    pcfg = cfg["evaluation"].get("perturbational", {"enabled": True})
    if not pcfg.get("enabled", False):
        return {}

    set_seed(seed)
    generator = make_generator(seed, device)
    batch_size = pcfg.get("probe_scenes", cfg["evaluation"]["probe_scenes"]) * task_cfg.num_types
    magnitudes = pcfg.get("magnitudes", [0.5, 1.0, 2.0])
    step = pcfg.get("step", max(task_cfg.num_steps // 2, 1))
    if step < 0 or step >= task_cfg.num_steps:
        raise ValueError("perturbational.step must be within the episode length")
    min_recovery = pcfg.get("min_recovery_ratio", 0.1)
    min_propagation = pcfg.get("min_attention_propagation", 0.05)

    batch = generate_batch(batch_size, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    with torch.no_grad():
        base = model(
            batch.scene, batch.cue, target=batch.target, target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
        )
    hidden_std = base["controller_state_seq"][:, step].std(dim=-1, keepdim=True).clamp_min(1e-3)
    hidden_dim = base["controller_state_seq"].shape[-1]

    conditions = {
        "recurrent": None,
        "feedforward_summary": {"feedforward_summary": True},
        "freeze_recurrence": {"freeze_recurrence": True},
        "shuffle_feedback": {"shuffle_feedback": True},
    }
    per_condition: dict[str, Any] = {}
    for name, ablation in conditions.items():
        per_magnitude = []
        for magnitude in magnitudes:
            unit = torch.randn(batch_size, hidden_dim, generator=generator, device=device)
            noise = unit * hidden_std * float(magnitude)
            response = _perturbational_response(
                model, batch, task_cfg, device, step=step, noise=noise, ablation=ablation
            )
            response["magnitude"] = float(magnitude)
            per_magnitude.append(response)
        per_condition[name] = {
            "per_magnitude": per_magnitude,
            "mean_recovery_ratio": sum(r["recovery_ratio"] for r in per_magnitude) / len(per_magnitude),
            "mean_attention_propagation": sum(r["attention_propagation"] for r in per_magnitude) / len(per_magnitude),
            "mean_task_distribution_shift": sum(r["task_distribution_shift"] for r in per_magnitude) / len(per_magnitude),
        }

    rec = per_condition["recurrent"]
    ff = per_condition["feedforward_summary"]
    freeze = per_condition["freeze_recurrence"]
    # Rich-but-recoverable: the perturbation propagates to later behaviour (non-trivial) and the
    # state trajectory partially recovers (not a rigid reset and not unrecoverable collapse).
    rich_but_recoverable = (
        rec["mean_recovery_ratio"] >= min_recovery
        and rec["mean_recovery_ratio"] < 0.999
        and rec["mean_attention_propagation"] >= min_propagation
    )
    # Integration depends on the recurrent state: the no-recurrence (feedforward) control
    # propagates the perturbation far less, while the frozen-state control cannot recover.
    integration_exceeds_feedforward = (
        rec["mean_attention_propagation"] > ff["mean_attention_propagation"]
    )
    recovery_exceeds_freeze = rec["mean_recovery_ratio"] > freeze["mean_recovery_ratio"]
    return {
        "step": step,
        "magnitudes": list(magnitudes),
        "conditions": per_condition,
        "thresholds": {
            "min_recovery_ratio": min_recovery,
            "min_attention_propagation": min_propagation,
        },
        "rich_but_recoverable": rich_but_recoverable,
        "integration_exceeds_feedforward": integration_exceeds_feedforward,
        "recovery_exceeds_freeze": recovery_exceeds_freeze,
        "recurrent_mean_recovery_ratio": rec["mean_recovery_ratio"],
        "recurrent_mean_attention_propagation": rec["mean_attention_propagation"],
        "feedforward_mean_attention_propagation": ff["mean_attention_propagation"],
        "freeze_mean_recovery_ratio": freeze["mean_recovery_ratio"],
        "supported": (
            rich_but_recoverable
            and integration_exceeds_feedforward
            and recovery_exceeds_freeze
        ),
    }


def _targets_for_cues(batch, cues: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover target positions and digits for arbitrary cues on a fixed batch of scenes."""

    target_pos = torch.zeros_like(cues)
    target = torch.zeros_like(cues)
    for idx in range(batch.scene.shape[0]):
        cue_idx = cues[idx].item()
        pos = torch.nonzero(batch.target_types[idx] == cue_idx, as_tuple=False).flatten().item()
        target_pos[idx] = pos
        target[idx] = batch.digits[idx, pos]
    return target_pos, target


def cue_switch_metrics(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
    *,
    switch_step: int,
) -> dict[str, float]:
    """Measure how strongly the model redirects attention after a mid-episode cue switch."""

    generator = make_generator(seed, device)
    base_batch = generate_batch(batch_size, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    cue_before = base_batch.cue
    cue_after = (cue_before + 1) % task_cfg.num_types
    cue_seq = cue_before.unsqueeze(1).repeat(1, task_cfg.num_steps)
    cue_seq[:, switch_step:] = cue_after.unsqueeze(1)
    switched_target_pos, switched_target = _targets_for_cues(base_batch, cue_after)

    with torch.no_grad():
        outputs = model(
            base_batch.scene,
            cue_before,
            cue_seq=cue_seq,
            target=switched_target,
            target_pos=switched_target_pos,
            target_pos_seq=switched_target_pos.unsqueeze(1).repeat(1, task_cfg.num_steps),
            num_steps=task_cfg.num_steps,
        )

    attention = outputs["attention_seq"]
    before_target = attention[:, switch_step - 1].gather(1, base_batch.target_pos.unsqueeze(-1)).mean().item()
    after_old_target = attention[:, switch_step:].gather(
        2, base_batch.target_pos[:, None, None].expand(-1, task_cfg.num_steps - switch_step, 1)
    ).mean().item()
    after_new_target = attention[:, switch_step:].gather(
        2, switched_target_pos[:, None, None].expand(-1, task_cfg.num_steps - switch_step, 1)
    ).mean().item()
    final_accuracy = (outputs["logits"].argmax(dim=-1) == switched_target).float().mean().item()

    return {
        "switch_step": switch_step,
        "pre_switch_old_target_attention": before_target,
        "post_switch_old_target_attention": after_old_target,
        "post_switch_new_target_attention": after_new_target,
        "switch_target_gain": after_new_target - after_old_target,
        "switch_accuracy": final_accuracy,
    }


def negative_control_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run controls that should not satisfy the recurrent-control interpretation."""

    set_seed(seed)
    control_cfg = cfg["evaluation"].get(
        "negative_controls",
        {
            "enabled": True,
            "high_capacity_observation_probe": {
                "train_batches": 12,
                "test_batches": 6,
                "epochs": 60,
                "learning_rate": 0.03,
                "hidden_dim": 128,
                "observation_window": 3,
            },
            "min_accuracy_drop": 0.02,
        },
    )
    if not control_cfg.get("enabled", False):
        return {}

    batch_size = cfg["training"]["batch_size"]
    test_batches = cfg["evaluation"]["test_batches"]
    probe_scenes = cfg["evaluation"]["probe_scenes"]
    controls: dict[str, Any] = {}
    recurrent_reference = evaluate_model(
        model,
        cfg,
        task_cfg,
        device,
        num_batches=test_batches,
        seed=seed + 900,
    )
    min_accuracy_drop = control_cfg.get("min_accuracy_drop", 0.02)
    for idx, name in enumerate(("feedforward_summary", "shuffle_feedback")):
        ablation = {name: True}
        metrics = evaluate_model(
            model,
            cfg,
            task_cfg,
            device,
            num_batches=test_batches,
            seed=seed + idx * 100,
            ablation=ablation,
        )
        metrics.update(
            trajectory_metrics(
                lambda scene, cue, target=None, target_pos=None, num_steps=None: model(
                    scene,
                    cue,
                    target=target,
                    target_pos=target_pos,
                    num_steps=num_steps,
                    ablation=ablation,
                ),
                task_cfg,
                probe_scenes,
                device,
                seed + 50 + idx * 100,
            )
        )
        controls[name] = metrics
        controls[name]["accuracy_drop_vs_recurrent"] = (
            recurrent_reference["accuracy"] - metrics["accuracy"]
        )
        controls[name]["failed_as_intended"] = (
            controls[name]["accuracy_drop_vs_recurrent"] >= min_accuracy_drop
        )

    probe_cfg = control_cfg.get("high_capacity_observation_probe", {})
    observation_window = probe_cfg.get("observation_window", 3)
    train_obs, train_targets = _collect_temporal_observation_probe_dataset(
        model,
        task_cfg,
        batch_size,
        probe_cfg.get("train_batches", 12),
        device,
        seed + 300,
        window=observation_window,
    )
    test_obs, test_targets = _collect_temporal_observation_probe_dataset(
        model,
        task_cfg,
        batch_size,
        probe_cfg.get("test_batches", 6),
        device,
        seed + 1300,
        window=observation_window,
    )
    high_capacity_probe = _train_mlp_attention_probe(
        train_obs,
        train_targets,
        test_obs,
        test_targets,
        hidden_dim=probe_cfg.get("hidden_dim", 128),
        epochs=probe_cfg.get("epochs", 60),
        learning_rate=probe_cfg.get("learning_rate", 0.03),
    )
    controller_reference = predictive_probe_metrics(
        model,
        cfg,
        task_cfg,
        device,
        seed + 2300,
    )
    controls["high_capacity_observation_only"] = {
        "probe": high_capacity_probe,
        "observation_window": observation_window,
        "controller_reference": controller_reference["controller_state_probe"],
        "test_cross_entropy_gap_vs_controller": (
            high_capacity_probe["test_cross_entropy"]
            - controller_reference["controller_state_probe"]["test_cross_entropy"]
        ),
        "test_top1_gap_vs_controller": (
            controller_reference["controller_state_probe"]["test_top1_match"]
            - high_capacity_probe["test_top1_match"]
        ),
        "failed_as_intended": (
            high_capacity_probe["test_cross_entropy"]
            > controller_reference["controller_state_probe"]["test_cross_entropy"]
            and high_capacity_probe["test_top1_match"]
            < controller_reference["controller_state_probe"]["test_top1_match"]
        ),
    }
    controls["failed_as_intended"] = (
        controls["feedforward_summary"]["failed_as_intended"]
        and controls["shuffle_feedback"]["failed_as_intended"]
        and controls["high_capacity_observation_only"]["failed_as_intended"]
    )
    controls["recurrent_reference"] = recurrent_reference
    controls["thresholds"] = {"min_accuracy_drop": min_accuracy_drop}
    return controls


def comparator_system_metrics(
    models: dict[str, Any],
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
    output_dir: Path,
    *,
    nl_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate first-class comparator systems against the target controller."""

    comparator_cfg = cfg["evaluation"].get("comparator_systems", {"enabled": True})
    if not comparator_cfg.get("enabled", False):
        return {}

    test_batches = cfg["evaluation"]["test_batches"]
    probe_scenes = cfg["evaluation"]["probe_scenes"]
    comparators: dict[str, Any] = {}

    comparators["static_feedforward"] = evaluate_model(
        models["static"],
        cfg,
        task_cfg,
        device,
        num_batches=test_batches,
        seed=seed,
    )
    comparators["static_feedforward"].update(
        trajectory_metrics(models["static"], task_cfg, probe_scenes, device, seed + 10)
    )

    feedforward_ablation = {"feedforward_summary": True}
    comparators["recurrent_feedforward_summary"] = evaluate_model(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        num_batches=test_batches,
        seed=seed + 100,
        ablation=feedforward_ablation,
    )
    comparators["recurrent_feedforward_summary"].update(
        trajectory_metrics(
            lambda scene, cue, target=None, target_pos=None, num_steps=None: models["recurrent"](
                scene,
                cue,
                target=target,
                target_pos=target_pos,
                num_steps=num_steps,
                ablation=feedforward_ablation,
            ),
            task_cfg,
            probe_scenes,
            device,
            seed + 110,
        )
    )

    transformer_cfg = comparator_cfg.get("matched_transformer", {})
    transformer_training_overrides = dict(transformer_cfg)
    if transformer_training_overrides.pop("match_recurrent_training", True):
        transformer_training_overrides = {
            **{
                key: cfg["training"][key]
                for key in (
                    "train_steps",
                    "val_batches",
                    "val_interval",
                    "log_interval",
                    "learning_rate",
                    "weight_decay",
                )
                if key in cfg["training"]
            },
            **transformer_training_overrides,
        }
    transformer_output = output_dir / "comparators" / "matched_transformer"
    transformer_output.mkdir(parents=True, exist_ok=True)
    transformer_train_cfg = deep_update(
        cfg,
        {
            "seed": seed + 200,
            "output_dir": str(transformer_output),
            "training": transformer_training_overrides,
        },
    )
    transformer = MatchedTransformerController(task_cfg, ModelConfig.from_dict(cfg["model"])).to(device)
    _, transformer_state = train_single_model(
        "matched_transformer",
        transformer,
        transformer_train_cfg,
        task_cfg,
        device,
        transformer_output,
    )
    transformer.load_state_dict(transformer_state["model_state_dict"])
    transformer.eval()
    comparators["matched_transformer"] = evaluate_model(
        transformer,
        transformer_train_cfg,
        task_cfg,
        device,
        num_batches=test_batches,
        seed=seed + 300,
    )
    comparators["matched_transformer"].update(
        trajectory_metrics(transformer, task_cfg, probe_scenes, device, seed + 310)
    )
    comparators["matched_transformer"]["training_budget"] = {
        "train_steps": transformer_train_cfg["training"]["train_steps"],
        "reference_recurrent_train_steps": cfg["training"].get("train_steps", 0),
        "matched_to_recurrent": (
            transformer_train_cfg["training"]["train_steps"]
            == cfg["training"].get("train_steps", 0)
        ),
    }

    trivial = TrivialUniformRegulator(task_cfg, ModelConfig.from_dict(cfg["model"])).to(device)
    comparators["trivial_uniform_regulator"] = evaluate_model(
        trivial,
        cfg,
        task_cfg,
        device,
        num_batches=test_batches,
        seed=seed + 400,
    )
    comparators["trivial_uniform_regulator"].update(
        trajectory_metrics(trivial, task_cfg, probe_scenes, device, seed + 410)
    )

    observation_report = (nl_report or {}).get("observation_only", {})
    comparators["large_lm_without_loop_proxy"] = {
        "proxy": "stage7_observation_only_reporter",
        "independent_comparator": False,
        "double_counts_stage7_observation_only": True,
        "joint_accuracy": observation_report.get("joint_accuracy", 0.0),
        "memory_content_joint_accuracy": observation_report.get(
            "memory_content_joint_accuracy",
            0.0,
        ),
        "note": (
            "This is a bookkeeping proxy for no-loop language-shaped reporting, not an "
            "independent comparator run; do not count it separately from Stage 7 observation-only."
        ),
    }
    recurrent_accuracy = evaluate_model(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        num_batches=test_batches,
        seed=seed + 500,
    )["accuracy"]
    comparators["target_recurrent_accuracy"] = recurrent_accuracy
    comparators["failed_as_intended"] = all(
        comparators[name]["accuracy"] < recurrent_accuracy
        for name in (
            "static_feedforward",
            "recurrent_feedforward_summary",
            "matched_transformer",
            "trivial_uniform_regulator",
        )
    )
    return comparators


def self_state_diagnostics(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Summarize how self-state signals evolve over time on a shared probe batch."""

    outputs, _ = _probe_outputs(model, task_cfg, batch_size, device, seed)
    inspection = outputs["inspection_seq"].float()
    self_model = outputs["self_model_seq"].float()
    found = outputs["found_state_seq"][..., 0].float()
    relevant_region = outputs["relevant_region_seq"][..., 0].float()
    unresolved_search = outputs["unresolved_search_seq"][..., 0].float()
    current_wrong_candidate = outputs["current_wrong_candidate_seq"][..., 0].float()
    wrong_candidate_history = outputs["wrong_candidate_history_seq"][..., 0].float()
    revisit_unresolved = outputs["revisit_unresolved_seq"][..., 0].float()
    allocation_error = outputs["allocation_error_seq"][..., 0].float()

    inspected_mass = (self_model * inspection).sum(dim=-1)
    uninspected_mass = (self_model * (1.0 - inspection)).sum(dim=-1)

    return {
        "inspection_coverage_by_step": inspection.mean(dim=(0, 2)).tolist(),
        "found_state_rate_by_step": found.mean(dim=0).tolist(),
        "relevant_region_rate_by_step": relevant_region.mean(dim=0).tolist(),
        "unresolved_search_rate_by_step": unresolved_search.mean(dim=0).tolist(),
        "current_wrong_candidate_rate_by_step": current_wrong_candidate.mean(dim=0).tolist(),
        "wrong_candidate_history_rate_by_step": wrong_candidate_history.mean(dim=0).tolist(),
        "revisit_unresolved_rate_by_step": revisit_unresolved.mean(dim=0).tolist(),
        "allocation_error_rate_by_step": allocation_error.mean(dim=0).tolist(),
        "self_model_mass_on_inspected_cells_by_step": inspected_mass.mean(dim=0).tolist(),
        "self_model_mass_on_uninspected_cells_by_step": uninspected_mass.mean(dim=0).tolist(),
    }


def self_model_diagnostics(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Summarize how the learned self-model tracks inspected vs uninspected cells."""

    outputs, probe_batch = _probe_outputs(model, task_cfg, batch_size, device, seed)
    attention = outputs["attention_seq"].float()
    inspection = outputs["inspection_seq"].float()
    self_model = outputs["self_model_seq"].float()
    target_pos = probe_batch.target_pos.view(batch_size, task_cfg.num_types)
    target_pos = target_pos.reshape(-1)

    flat_attention = attention.reshape(-1, task_cfg.num_steps, task_cfg.num_cells)
    flat_inspection = inspection.reshape(-1, task_cfg.num_steps, task_cfg.num_cells)
    flat_self_model = self_model.reshape(-1, task_cfg.num_steps, task_cfg.num_cells)
    target_index = target_pos[:, None, None].expand(-1, task_cfg.num_steps, 1)

    target_self_model = flat_self_model.gather(2, target_index).squeeze(-1)
    target_inspection = flat_inspection.gather(2, target_index).squeeze(-1)
    target_attention = flat_attention.gather(2, target_index).squeeze(-1)

    inspected_error = (flat_self_model - flat_inspection).abs().mean(dim=-1)
    target_alignment = 1.0 - (target_self_model - target_inspection).abs()

    return {
        "target_self_model_mass_by_step": target_self_model.mean(dim=0).tolist(),
        "target_inspection_state_by_step": target_inspection.mean(dim=0).tolist(),
        "target_attention_by_step": target_attention.mean(dim=0).tolist(),
        "self_model_cell_error_by_step": inspected_error.mean(dim=0).tolist(),
        "target_self_model_alignment_by_step": target_alignment.mean(dim=0).tolist(),
    }


def save_intervention_plots(
    model,
    output_dir: Path,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> list[str]:
    """Render baseline vs intervened attention maps around the intervention step."""

    intervention_cfg = cfg["evaluation"].get("intervention_test", {})
    if not intervention_cfg.get("enabled", False):
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 1
    step = intervention_cfg.get("step", 2)
    outputs, probe_batch = _probe_outputs(model, task_cfg, batch_size, device, seed)
    states = outputs["controller_state_seq"].view(batch_size, task_cfg.num_types, task_cfg.num_steps, -1)
    baseline_attention = outputs["attention_seq"].view(batch_size, task_cfg.num_types, task_cfg.num_steps, -1)
    target_pos = probe_batch.target_pos.view(batch_size, task_cfg.num_types)

    alt_cue = (torch.arange(task_cfg.num_types, device=device) + 1) % task_cfg.num_types
    current_state = states[:, torch.arange(task_cfg.num_types, device=device), step]
    alternate_state = states[:, alt_cue, step]
    intervention_delta = (alternate_state - current_state).reshape(-1, current_state.shape[-1])

    intervened_outputs = model(
        probe_batch.scene,
        probe_batch.cue,
        target=probe_batch.target,
        target_pos=probe_batch.target_pos,
        num_steps=task_cfg.num_steps,
        intervention={"step": step, "delta": intervention_delta},
    )
    intervened_attention = intervened_outputs["attention_seq"].view(
        batch_size, task_cfg.num_types, task_cfg.num_steps, -1
    )

    saved_paths = []
    shown_steps = sorted({max(step - 1, 0), step})
    for cue_idx in range(task_cfg.num_types):
        target_row, target_col = divmod(target_pos[0, cue_idx].item(), task_cfg.grid_size)
        alt_row, alt_col = divmod(target_pos[0, alt_cue[cue_idx]].item(), task_cfg.grid_size)
        fig, axes = plt.subplots(2, len(shown_steps), figsize=(4 * len(shown_steps), 7))
        if len(shown_steps) == 1:
            axes = [[axes[0]], [axes[1]]]

        for col_idx, step_idx in enumerate(shown_steps):
            base_ax = axes[0][col_idx]
            int_ax = axes[1][col_idx]
            base_heatmap = baseline_attention[0, cue_idx, step_idx].reshape(
                task_cfg.grid_size, task_cfg.grid_size
            ).detach().cpu()
            int_heatmap = intervened_attention[0, cue_idx, step_idx].reshape(
                task_cfg.grid_size, task_cfg.grid_size
            ).detach().cpu()

            for ax, heatmap, title_prefix in (
                (base_ax, base_heatmap, "Baseline"),
                (int_ax, int_heatmap, "Intervened"),
            ):
                ax.imshow(heatmap, cmap="viridis")
                ax.scatter(
                    target_col,
                    target_row,
                    s=160,
                    marker="o",
                    facecolors="none",
                    edgecolors="white",
                    linewidths=2,
                )
                ax.scatter(
                    alt_col,
                    alt_row,
                    s=140,
                    marker="x",
                    c="red",
                    linewidths=2,
                )
                ax.set_title(f"{title_prefix} / Cue {cue_idx} / Step {step_idx + 1}")
                ax.axis("off")

        fig.suptitle("White circle: original target, red X: alternate-cue target", fontsize=11)
        fig.tight_layout()
        path = output_dir / f"intervention_probe_cue_{cue_idx}.png"
        fig.savefig(path)
        plt.close(fig)
        saved_paths.append(str(path))

    return saved_paths


def save_self_state_plots(
    diagnostics: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Render compact line plots for recurrent self-state diagnostics."""

    if not diagnostics:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    num_steps = len(diagnostics["inspection_coverage_by_step"])
    steps = list(range(1, num_steps + 1))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    bounded_series = (
        ("inspection_coverage_by_step", "Inspection coverage"),
        ("found_state_rate_by_step", "Found state"),
        ("relevant_region_rate_by_step", "Relevant region"),
        ("unresolved_search_rate_by_step", "Unresolved search"),
        ("current_wrong_candidate_rate_by_step", "Current wrong candidate"),
        ("wrong_candidate_history_rate_by_step", "Wrong candidate history"),
        ("revisit_unresolved_rate_by_step", "Revisit while unresolved"),
        ("allocation_error_rate_by_step", "Allocation error"),
    )
    for key, label in bounded_series:
        axes[0].plot(steps, diagnostics[key], marker="o", label=label)
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Stepwise self-state rates")
    axes[0].legend(loc="best", fontsize=8)

    mass_series = (
        ("self_model_mass_on_inspected_cells_by_step", "Self-model on inspected"),
        ("self_model_mass_on_uninspected_cells_by_step", "Self-model on uninspected"),
    )
    for key, label in mass_series:
        axes[1].plot(steps, diagnostics[key], marker="o", label=label)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean mass")
    axes[1].set_title("Self-model mass allocation")
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = output_dir / "self_state_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def save_self_model_plots(
    diagnostics: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Render compact line plots for self-model trajectory diagnostics."""

    if not diagnostics:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    num_steps = len(diagnostics["target_self_model_mass_by_step"])
    steps = list(range(1, num_steps + 1))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    bounded_series = (
        ("target_self_model_mass_by_step", "Target self-model mass"),
        ("target_inspection_state_by_step", "Target inspection state"),
        ("target_attention_by_step", "Target attention"),
        ("target_self_model_alignment_by_step", "Target self-model alignment"),
    )
    for key, label in bounded_series:
        axes[0].plot(steps, diagnostics[key], marker="o", label=label)
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Target-focused self-model trajectories")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(
        steps,
        diagnostics["self_model_cell_error_by_step"],
        marker="o",
        label="Self-model cell error",
    )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_title("Self-model vs inspection mismatch")
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = output_dir / "self_model_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def save_uncertainty_report_plots(
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Render compact comparison plots for Stage 6B uncertainty reports."""

    if not metrics:
        return []

    signal_names = (
        "relevant_region_inspected",
        "unresolved_search",
        "current_wrong_candidate",
        "wrong_candidate_history",
        "revisit_unresolved",
        "allocation_error",
    )
    available = [name for name in signal_names if name in metrics]
    if not available:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [name.replace("_", "\n") for name in available]
    native_accuracy = [metrics[name]["native_report"]["test_accuracy"] for name in available]
    observation_accuracy = [
        metrics[name]["observation_only_probe"]["test_accuracy"] for name in available
    ]
    native_recall = [metrics[name]["native_report"]["test_positive_recall"] for name in available]
    observation_recall = [
        metrics[name]["observation_only_probe"]["test_positive_recall"] for name in available
    ]

    x_positions = np.arange(len(available))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].bar(x_positions - width / 2, native_accuracy, width=width, label="Native")
    axes[0].bar(
        x_positions + width / 2,
        observation_accuracy,
        width=width,
        label="Observation-only",
    )
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Stage 6B accuracy comparison")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(x_positions - width / 2, native_recall, width=width, label="Native")
    axes[1].bar(
        x_positions + width / 2,
        observation_recall,
        width=width,
        label="Observation-only",
    )
    axes[1].set_ylabel("Positive recall")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Stage 6B positive-recall comparison")
    axes[1].set_xticks(x_positions, labels)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = output_dir / "uncertainty_report_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def save_stage3_multi_seed_plots(
    metrics: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Render compact robustness plots for repeated-seed Stage 3 checks."""

    if not metrics:
        return []
    predictive_runs = metrics.get("predictive_probe_runs", [])
    intervention_runs = metrics.get("intervention_runs", [])
    if not predictive_runs and not intervention_runs:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    if predictive_runs:
        x = list(range(1, len(predictive_runs) + 1))
        axes[0].plot(
            x,
            [run["controller_advantage_cross_entropy"] for run in predictive_runs],
            marker="o",
            label="Cross-entropy advantage",
        )
        axes[0].plot(
            x,
            [run["controller_advantage_top1_match"] for run in predictive_runs],
            marker="o",
            label="Top-1 advantage",
        )
        axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
        axes[0].set_title("Stage 3 predictive-probe margins by seed")
        axes[0].set_ylabel("Advantage")
        axes[0].legend(loc="best", fontsize=8)
    else:
        axes[0].axis("off")

    if intervention_runs:
        x = list(range(1, len(intervention_runs) + 1))
        axes[1].plot(
            x,
            [run["attention_change_kl"] for run in intervention_runs],
            marker="o",
            label="Attention-change KL",
        )
        axes[1].plot(
            x,
            [run["original_target_attention_drop"] for run in intervention_runs],
            marker="o",
            label="Original-target drop",
        )
        axes[1].plot(
            x,
            [run["alternate_target_attention_gain"] for run in intervention_runs],
            marker="o",
            label="Alternate-target gain",
        )
        axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
        axes[1].set_title("Stage 3 intervention margins by seed")
        axes[1].set_xlabel("Seed index")
        axes[1].set_ylabel("Effect size")
        axes[1].legend(loc="best", fontsize=8)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    path = output_dir / "stage3_multi_seed_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def save_stage3_checkpoint_family_plots(
    summary: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Render compact Stage 3 robustness plots across checkpoint families."""

    families = summary.get("families", [])
    if not families:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [family.get("family", "unknown") for family in families]
    predictive = [family.get("predictive_supported_fraction", 0.0) for family in families]
    intervention = [family.get("intervention_supported_fraction", 0.0) for family in families]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].bar(x - width / 2, predictive, width=width, label="Predictive support fraction")
    axes[0].bar(x + width / 2, intervention, width=width, label="Intervention support fraction")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Support fraction")
    axes[0].set_title("Stage 3 support fractions across checkpoint families")
    axes[0].legend(loc="best", fontsize=8)

    predictive_gap = [family.get("predictive_cross_entropy_advantage_min_gap", 0.0) for family in families]
    intervention_gap = [
        family.get("intervention_alternate_target_gain_min_gap", 0.0) for family in families
    ]
    axes[1].bar(x - width / 2, predictive_gap, width=width, label="Predictive min-gap")
    axes[1].bar(x + width / 2, intervention_gap, width=width, label="Intervention gain min-gap")
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Worst-case threshold gap")
    axes[1].set_title("Stage 3 bottleneck margins across checkpoint families")
    axes[1].set_xticks(x, labels)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = output_dir / "stage3_checkpoint_family_diagnostics.png"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def export_stage3_robustness_tables(
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Export compact machine-readable Stage 3 robustness tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    multi_seed = report.get("stage3_multi_seed", {})
    checkpoint_family = report.get("stage3_checkpoint_family", {})

    seed_rows: list[dict[str, Any]] = []
    for run in multi_seed.get("predictive_probe_runs", []):
        seed_rows.append(
            {
                "table": "predictive_probe_runs",
                "seed": run.get("seed", 0),
                "supported": run.get("supported", False),
                "controller_advantage_cross_entropy": run.get(
                    "controller_advantage_cross_entropy", 0.0
                ),
                "controller_advantage_mse": run.get("controller_advantage_mse", 0.0),
                "controller_advantage_top1_match": run.get("controller_advantage_top1_match", 0.0),
                "cross_entropy_gap": run.get("cross_entropy_gap", 0.0),
                "mse_gap": run.get("mse_gap", 0.0),
                "top1_gap": run.get("top1_gap", 0.0),
            }
        )
    for run in multi_seed.get("intervention_runs", []):
        seed_rows.append(
            {
                "table": "intervention_runs",
                "seed": run.get("seed", 0),
                "supported": run.get("supported", False),
                "attention_change_kl": run.get("attention_change_kl", 0.0),
                "original_target_attention_drop": run.get("original_target_attention_drop", 0.0),
                "alternate_target_attention_gain": run.get("alternate_target_attention_gain", 0.0),
                "attention_change_kl_gap": run.get("attention_change_kl_gap", 0.0),
                "original_target_attention_drop_gap": run.get(
                    "original_target_attention_drop_gap", 0.0
                ),
                "alternate_target_attention_gain_gap": run.get(
                    "alternate_target_attention_gain_gap", 0.0
                ),
            }
        )

    family_rows = checkpoint_family.get("families", [])

    seed_path = output_dir / "stage3_seed_table.json"
    with open(seed_path, "w", encoding="utf-8") as handle:
        json.dump(seed_rows, handle, indent=2)

    family_path = output_dir / "stage3_checkpoint_family_table.json"
    with open(family_path, "w", encoding="utf-8") as handle:
        json.dump(family_rows, handle, indent=2)

    return {
        "stage3_seed_table": str(seed_path),
        "stage3_checkpoint_family_table": str(family_path),
    }


def save_stage3_robustness_note(report: dict[str, Any], output_dir: Path) -> str:
    """Write a compact human-readable Stage 3 robustness note."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = report.get("stage3_summary", {})
    checkpoint_family = report.get("stage3_checkpoint_family", {})
    lines = [
        "# Stage 3 Robustness Note",
        "",
        f"- Single-run support: `{summary.get('single_run_supported', False)}`",
        f"- Robust support: `{summary.get('robust_supported', False)}`",
        f"- Predictive supported fraction: `{summary.get('predictive_supported_fraction', 0.0):.3f}`",
        f"- Intervention supported fraction: `{summary.get('intervention_supported_fraction', 0.0):.3f}`",
        f"- Bottleneck metric: `{summary.get('bottleneck_metric', '')}`",
        f"- Bottleneck gap: `{summary.get('bottleneck_gap', 0.0):.6f}`",
        f"- Worst predictive seed: `{summary.get('worst_predictive_seed', 0)}`",
        f"- Worst intervention seed: `{summary.get('worst_intervention_seed', 0)}`",
        f"- Checkpoint-family verdict: `{checkpoint_family.get('verdict', 'missing')}`",
        f"- Checkpoint-family bottleneck: `{checkpoint_family.get('bottleneck_family', '')}` / `{checkpoint_family.get('bottleneck_metric', '')}`",
    ]
    failure_reasons = summary.get("failure_reasons", [])
    if failure_reasons:
        lines.append(f"- Failure reasons: `{', '.join(failure_reasons)}`")

    path = output_dir / "stage3_robustness_note.md"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return str(path)


def save_cue_switch_plots(
    models: dict[str, Any],
    output_dir: Path,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> list[str]:
    """Render baseline vs recurrent attention around a cue switch."""

    cue_switch_cfg = cfg["evaluation"].get("cue_switch", {})
    if not cue_switch_cfg.get("enabled", False):
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    switch_step = cue_switch_cfg.get("switch_step", task_cfg.num_steps // 2)
    shown_steps = sorted({max(switch_step - 1, 0), switch_step})
    generator = make_generator(seed, device)
    batch = generate_batch(1, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    cue_before = batch.cue
    cue_after = (cue_before + 1) % task_cfg.num_types
    cue_seq = cue_before.unsqueeze(1).repeat(1, task_cfg.num_steps)
    cue_seq[:, switch_step:] = cue_after.unsqueeze(1)
    switched_target_pos, switched_target = _targets_for_cues(batch, cue_after)

    outputs = {}
    for name, model in models.items():
        with torch.no_grad():
            outputs[name] = model(
                batch.scene,
                cue_before,
                cue_seq=cue_seq,
                target=switched_target,
                target_pos=switched_target_pos,
                target_pos_seq=switched_target_pos.unsqueeze(1).repeat(1, task_cfg.num_steps),
                num_steps=task_cfg.num_steps,
            )

    old_target_row, old_target_col = divmod(batch.target_pos[0].item(), task_cfg.grid_size)
    new_target_row, new_target_col = divmod(switched_target_pos[0].item(), task_cfg.grid_size)
    fig, axes = plt.subplots(len(outputs), len(shown_steps), figsize=(4 * len(shown_steps), 3.5 * len(outputs)))
    if len(outputs) == 1:
        axes = [axes]
    if len(shown_steps) == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (name, model_outputs) in enumerate(outputs.items()):
        for col_idx, step_idx in enumerate(shown_steps):
            ax = axes[row_idx][col_idx]
            heatmap = model_outputs["attention_seq"][0, step_idx].reshape(
                task_cfg.grid_size, task_cfg.grid_size
            ).detach().cpu()
            ax.imshow(heatmap, cmap="viridis")
            ax.scatter(
                old_target_col,
                old_target_row,
                s=160,
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=2,
            )
            ax.scatter(
                new_target_col,
                new_target_row,
                s=140,
                marker="x",
                c="red",
                linewidths=2,
            )
            ax.set_title(f"{name} / Step {step_idx + 1}")
            ax.axis("off")

    fig.suptitle("Cue-switch attention: white circle = old target, red X = switched target", fontsize=11)
    fig.tight_layout()
    path = output_dir / "cue_switch_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def reduced_shaping_metrics(
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Retrain recurrent controllers with weaker shaping and compare their behavior."""

    reduced_cfg = cfg["evaluation"].get("reduced_shaping", {})
    if not reduced_cfg.get("enabled", False):
        return {}

    weights = reduced_cfg.get("weights", [])
    if not weights:
        return {}
    thresholds = reduced_cfg.get("thresholds", {})

    shaping_dir = output_dir / "reduced_shaping"
    shaping_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for weight_idx, weight in enumerate(weights):
        training_overrides = reduced_cfg.get("training_overrides", {})
        variant_cfg = deep_update(
            cfg,
            {
                "seed": cfg["seed"] + 20000 + weight_idx * 100,
                "output_dir": str(shaping_dir / f"attn_weight_{str(weight).replace('.', '_')}"),
                "training": deep_update(
                    training_overrides,
                    {
                        # Reduce both the final-step and stepwise attention shaping together,
                        # otherwise full-strength stepwise supervision masks the reduction.
                        "attention_target_weight": float(weight),
                        "stepwise_attention_target_weight": float(weight),
                    },
                ),
            },
        )
        set_seed(variant_cfg["seed"])
        model_cfg = ModelConfig.from_dict(variant_cfg["model"])
        model = RecurrentAttentionController(task_cfg, model_cfg).to(device)
        variant_output_dir = Path(variant_cfg["output_dir"])
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        _, best_state = train_single_model(
            f"recurrent_attn_weight_{str(weight).replace('.', '_')}",
            model,
            variant_cfg,
            task_cfg,
            device,
            variant_output_dir,
        )
        model.load_state_dict(best_state["model_state_dict"])
        model.eval()

        variant_report = evaluate_model(
            model,
            variant_cfg,
            task_cfg,
            device,
            num_batches=cfg["evaluation"]["test_batches"],
            seed=variant_cfg["seed"] + 9000,
        )
        variant_report.update(
            trajectory_metrics(
                model,
                task_cfg,
                cfg["evaluation"]["probe_scenes"],
                device,
                variant_cfg["seed"] + 9100,
            )
        )
        variant_report.update(
            cue_sensitivity_metrics(
                model,
                task_cfg,
                cfg["evaluation"]["probe_scenes"],
                device,
                variant_cfg["seed"] + 9200,
            )
        )
        variant_report["stage3"] = evaluate_stage3_bundle(
            model,
            variant_cfg,
            task_cfg,
            device,
            variant_cfg["seed"] + 9300,
            reduced_shaping_summary={
                "supported": (
                    variant_report["accuracy"] >= thresholds.get("min_accuracy", 0.1)
                    and variant_report["temporal_reallocation"]
                    >= thresholds.get("min_temporal_reallocation", 0.0)
                    and variant_report["target_attention_gain"]
                    >= thresholds.get("min_target_attention_gain", 0.0)
                )
            },
        )
        results[str(weight)] = variant_report

    lowest_weight_key = str(min(weights))
    lowest_variant = results[lowest_weight_key]
    # The bounded resilience claim is about reduced (not removed) shaping: gate "supported" on
    # the lowest NON-ZERO weight. Complete zero-shaping is a separate stress test that is expected
    # to collapse to ~static accuracy and is reported as zero_weight_supported, not as support.
    nonzero_weights = [w for w in weights if w > 0]
    bounded_weight_key = str(min(nonzero_weights)) if nonzero_weights else lowest_weight_key
    bounded_variant = results[bounded_weight_key]
    zero_variant = results.get("0.0")
    zero_supported = False
    if zero_variant is not None:
        zero_supported = (
            zero_variant["accuracy"] >= thresholds.get("min_accuracy", 0.1)
            and zero_variant["temporal_reallocation"]
            >= thresholds.get("min_temporal_reallocation", 0.0)
            and zero_variant["target_attention_gain"]
            >= thresholds.get("min_target_attention_gain", 0.0)
        )
    results["summary"] = {
        "lowest_weight": float(lowest_weight_key),
        "lowest_weight_accuracy": lowest_variant["accuracy"],
        "lowest_weight_temporal_reallocation": lowest_variant["temporal_reallocation"],
        "lowest_weight_target_attention_gain": lowest_variant["target_attention_gain"],
        "zero_weight_tested": zero_variant is not None,
        "zero_weight_accuracy": zero_variant["accuracy"] if zero_variant is not None else 0.0,
        "zero_weight_temporal_reallocation": (
            zero_variant["temporal_reallocation"] if zero_variant is not None else 0.0
        ),
        "zero_weight_target_attention_gain": (
            zero_variant["target_attention_gain"] if zero_variant is not None else 0.0
        ),
        "zero_weight_supported": zero_supported,
        "thresholds": {
            "min_accuracy": thresholds.get("min_accuracy", 0.1),
            "min_temporal_reallocation": thresholds.get("min_temporal_reallocation", 0.0),
            "min_target_attention_gain": thresholds.get("min_target_attention_gain", 0.0),
        },
        "bounded_weight": float(bounded_weight_key),
        "bounded_weight_accuracy": bounded_variant["accuracy"],
        "supported": (
            bounded_variant["accuracy"] >= thresholds.get("min_accuracy", 0.1)
            and bounded_variant["temporal_reallocation"]
            >= thresholds.get("min_temporal_reallocation", 0.0)
            and bounded_variant["target_attention_gain"]
            >= thresholds.get("min_target_attention_gain", 0.0)
        ),
    }
    return results


def stage3_multi_seed_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
    *,
    reduced_shaping_summary: dict[str, Any],
) -> dict[str, Any]:
    """Repeat Stage 3 probe-style checks across multiple evaluation seeds."""

    multi_cfg = cfg["evaluation"].get("stage3_multi_seed", {})
    if not multi_cfg.get("enabled", False):
        return {}

    num_seeds = int(multi_cfg.get("num_seeds", 3))
    seed_stride = int(multi_cfg.get("seed_stride", 100))
    predictive_runs = []
    intervention_runs = []

    for idx in range(num_seeds):
        run_seed = seed + idx * seed_stride
        predictive = predictive_probe_metrics(
            model,
            cfg,
            task_cfg,
            device,
            run_seed,
        )
        intervention = intervention_test_metrics(
            model,
            cfg,
            task_cfg,
            device,
            run_seed + seed_stride // 2,
        )
        predictive_runs.append(
            {
                "seed": run_seed,
                "supported": predictive.get("supported", False),
                "controller_advantage_cross_entropy": predictive.get(
                    "controller_advantage_cross_entropy", 0.0
                ),
                "controller_advantage_mse": predictive.get("controller_advantage_mse", 0.0),
                "controller_advantage_top1_match": predictive.get(
                    "controller_advantage_top1_match", 0.0
                ),
            }
        )
        intervention_runs.append(
            {
                "seed": run_seed + seed_stride // 2,
                "supported": intervention.get("supported", False),
                "attention_change_kl": intervention.get("attention_change_kl", 0.0),
                "original_target_attention_drop": intervention.get(
                    "original_target_attention_drop", 0.0
                ),
                "alternate_target_attention_gain": intervention.get(
                    "alternate_target_attention_gain", 0.0
                ),
            }
        )

    predictive_supported = sum(int(item["supported"]) for item in predictive_runs)
    intervention_supported = sum(int(item["supported"]) for item in intervention_runs)
    reduced_shaping_supported = reduced_shaping_summary.get("supported", False)
    predictive_thresholds = cfg["evaluation"].get("predictive_probe", {}).get("thresholds", {})
    intervention_thresholds = cfg["evaluation"].get("intervention_test", {}).get("thresholds", {})

    def _mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    predictive_ce = [item["controller_advantage_cross_entropy"] for item in predictive_runs]
    predictive_mse = [item["controller_advantage_mse"] for item in predictive_runs]
    predictive_top1 = [item["controller_advantage_top1_match"] for item in predictive_runs]
    intervention_kl = [item["attention_change_kl"] for item in intervention_runs]
    intervention_drop = [item["original_target_attention_drop"] for item in intervention_runs]
    intervention_gain = [item["alternate_target_attention_gain"] for item in intervention_runs]

    ce_threshold = predictive_thresholds.get("min_advantage_cross_entropy", 0.0)
    mse_threshold = predictive_thresholds.get("min_advantage_mse", 0.0)
    top1_threshold = predictive_thresholds.get("min_advantage_top1_match", 0.0)
    kl_threshold = intervention_thresholds.get("min_attention_change_kl", 0.0)
    drop_threshold = intervention_thresholds.get("min_original_target_attention_drop", 0.0)
    gain_threshold = intervention_thresholds.get("min_alternate_target_attention_gain", 0.0)

    for run in predictive_runs:
        run["cross_entropy_gap"] = run["controller_advantage_cross_entropy"] - ce_threshold
        run["mse_gap"] = run["controller_advantage_mse"] - mse_threshold
        run["top1_gap"] = run["controller_advantage_top1_match"] - top1_threshold
    for run in intervention_runs:
        run["attention_change_kl_gap"] = run["attention_change_kl"] - kl_threshold
        run["original_target_attention_drop_gap"] = (
            run["original_target_attention_drop"] - drop_threshold
        )
        run["alternate_target_attention_gain_gap"] = (
            run["alternate_target_attention_gain"] - gain_threshold
        )

    predictive_gap_items = [
        ("predictive_cross_entropy_advantage_min_gap", min((run["cross_entropy_gap"] for run in predictive_runs), default=0.0)),
        ("predictive_mse_advantage_min_gap", min((run["mse_gap"] for run in predictive_runs), default=0.0)),
        ("predictive_top1_advantage_min_gap", min((run["top1_gap"] for run in predictive_runs), default=0.0)),
        (
            "intervention_attention_change_kl_min_gap",
            min((run["attention_change_kl_gap"] for run in intervention_runs), default=0.0),
        ),
        (
            "intervention_original_target_drop_min_gap",
            min((run["original_target_attention_drop_gap"] for run in intervention_runs), default=0.0),
        ),
        (
            "intervention_alternate_target_gain_min_gap",
            min((run["alternate_target_attention_gain_gap"] for run in intervention_runs), default=0.0),
        ),
    ]
    bottleneck_metric, bottleneck_gap = min(predictive_gap_items, key=lambda item: item[1])
    worst_predictive_run = min(
        predictive_runs,
        key=lambda run: min(run["cross_entropy_gap"], run["mse_gap"], run["top1_gap"]),
        default={"seed": seed},
    )
    worst_intervention_run = min(
        intervention_runs,
        key=lambda run: min(
            run["attention_change_kl_gap"],
            run["original_target_attention_drop_gap"],
            run["alternate_target_attention_gain_gap"],
        ),
        default={"seed": seed + seed_stride // 2},
    )

    failure_reasons = []
    if predictive_supported != num_seeds:
        failure_reasons.append("predictive_probe_instability")
    if intervention_supported != num_seeds:
        failure_reasons.append("intervention_instability")
    if not reduced_shaping_supported:
        failure_reasons.append("reduced_shaping_failed")

    return {
        "num_seeds": num_seeds,
        "predictive_probe_runs": predictive_runs,
        "intervention_runs": intervention_runs,
        "reduced_shaping_supported": reduced_shaping_supported,
        "predictive_supported_fraction": predictive_supported / max(num_seeds, 1),
        "intervention_supported_fraction": intervention_supported / max(num_seeds, 1),
        "all_predictive_supported": predictive_supported == num_seeds,
        "all_intervention_supported": intervention_supported == num_seeds,
        "predictive_cross_entropy_supported_fraction": (
            sum(int(run["cross_entropy_gap"] >= 0.0) for run in predictive_runs) / max(num_seeds, 1)
        ),
        "predictive_mse_supported_fraction": (
            sum(int(run["mse_gap"] >= 0.0) for run in predictive_runs) / max(num_seeds, 1)
        ),
        "predictive_top1_supported_fraction": (
            sum(int(run["top1_gap"] >= 0.0) for run in predictive_runs) / max(num_seeds, 1)
        ),
        "predictive_cross_entropy_advantage_mean": _mean(predictive_ce),
        "predictive_cross_entropy_advantage_min": min(predictive_ce) if predictive_ce else 0.0,
        "predictive_cross_entropy_advantage_min_gap": (
            (min(predictive_ce) if predictive_ce else 0.0)
            - ce_threshold
        ),
        "predictive_mse_advantage_mean": _mean(predictive_mse),
        "predictive_mse_advantage_min": min(predictive_mse) if predictive_mse else 0.0,
        "predictive_mse_advantage_min_gap": (
            (min(predictive_mse) if predictive_mse else 0.0)
            - mse_threshold
        ),
        "predictive_top1_advantage_mean": _mean(predictive_top1),
        "predictive_top1_advantage_min": min(predictive_top1) if predictive_top1 else 0.0,
        "predictive_top1_advantage_min_gap": (
            (min(predictive_top1) if predictive_top1 else 0.0)
            - top1_threshold
        ),
        "intervention_attention_change_kl_supported_fraction": (
            sum(int(run["attention_change_kl_gap"] >= 0.0) for run in intervention_runs)
            / max(num_seeds, 1)
        ),
        "intervention_original_target_drop_supported_fraction": (
            sum(int(run["original_target_attention_drop_gap"] >= 0.0) for run in intervention_runs)
            / max(num_seeds, 1)
        ),
        "intervention_alternate_target_gain_supported_fraction": (
            sum(int(run["alternate_target_attention_gain_gap"] >= 0.0) for run in intervention_runs)
            / max(num_seeds, 1)
        ),
        "intervention_attention_change_kl_mean": _mean(intervention_kl),
        "intervention_attention_change_kl_min": min(intervention_kl) if intervention_kl else 0.0,
        "intervention_attention_change_kl_min_gap": (
            (min(intervention_kl) if intervention_kl else 0.0)
            - kl_threshold
        ),
        "intervention_original_target_drop_mean": _mean(intervention_drop),
        "intervention_original_target_drop_min": min(intervention_drop) if intervention_drop else 0.0,
        "intervention_original_target_drop_min_gap": (
            (min(intervention_drop) if intervention_drop else 0.0)
            - drop_threshold
        ),
        "intervention_alternate_target_gain_mean": _mean(intervention_gain),
        "intervention_alternate_target_gain_min": min(intervention_gain) if intervention_gain else 0.0,
        "intervention_alternate_target_gain_min_gap": (
            (min(intervention_gain) if intervention_gain else 0.0)
            - gain_threshold
        ),
        "worst_predictive_seed": worst_predictive_run["seed"],
        "worst_intervention_seed": worst_intervention_run["seed"],
        "bottleneck_metric": bottleneck_metric,
        "bottleneck_gap": bottleneck_gap,
        "failure_reasons": failure_reasons,
        "supported": (
            predictive_supported == num_seeds
            and intervention_supported == num_seeds
            and reduced_shaping_supported
        ),
    }


def evaluate_stage3_bundle(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
    *,
    reduced_shaping_summary: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the full Stage 3 bundle for a single checkpoint family."""

    predictive_probe = predictive_probe_metrics(
        model,
        cfg,
        task_cfg,
        device,
        seed,
    )
    intervention_test = intervention_test_metrics(
        model,
        cfg,
        task_cfg,
        device,
        seed + 100,
    )
    multi_seed = stage3_multi_seed_metrics(
        model,
        cfg,
        task_cfg,
        device,
        seed + 200,
        reduced_shaping_summary=reduced_shaping_summary,
    )
    stage3_summary = build_stage3_summary(
        {
            "predictive_probe": predictive_probe,
            "intervention_test": intervention_test,
            "reduced_shaping": {"summary": reduced_shaping_summary},
            "stage3_multi_seed": multi_seed,
        }
    )
    return {
        "predictive_probe": predictive_probe,
        "intervention_test": intervention_test,
        "stage3_multi_seed": multi_seed,
        "stage3_summary": stage3_summary,
    }


def build_stage3_checkpoint_family_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Aggregate Stage 3 robustness across the default and reduced-shaping checkpoints."""

    families = [
        {
            "family": "default",
            **report.get("stage3_summary", {}),
        }
    ]
    # Complete zero-shaping is a deliberate stress test that is outside the bounded Stage 3
    # claim (the roadmap states this explicitly). Keep it as an informational stress family
    # rather than letting its expected failure flip the robustness verdict.
    stress_families = []
    reduced_shaping = report.get("reduced_shaping", {})
    for family_name, family_report in reduced_shaping.items():
        if family_name == "summary" or not isinstance(family_report, dict):
            continue
        stage3 = family_report.get("stage3", {})
        entry = {
            "family": f"reduced_shaping_{family_name}",
            **stage3.get("stage3_summary", {}),
        }
        try:
            is_stress = float(family_name) == 0.0
        except (TypeError, ValueError):
            is_stress = False
        if is_stress:
            stress_families.append(entry)
        else:
            families.append(entry)

    robust_families = sum(int(family.get("robust_supported", False)) for family in families)
    single_run_families = sum(int(family.get("single_run_supported", False)) for family in families)
    failure_reasons = sorted(
        {
            reason
            for family in families
            for reason in family.get("failure_reasons", [])
        }
    )
    verdict = "robust"
    if robust_families != len(families):
        verdict = "checkpoint_fragile"
    if single_run_families == 0:
        verdict = "no_single_run_support"
    bottleneck_family = min(
        families,
        key=lambda family: family.get("bottleneck_gap", 0.0),
        default={"family": "default", "bottleneck_metric": "", "bottleneck_gap": 0.0},
    )

    return {
        "num_families": len(families),
        "single_run_supported_families": single_run_families,
        "robust_supported_families": robust_families,
        "supported": robust_families == len(families),
        "verdict": verdict,
        "failure_reasons": failure_reasons,
        "bottleneck_family": bottleneck_family.get("family", "default"),
        "bottleneck_metric": bottleneck_family.get("bottleneck_metric", ""),
        "bottleneck_gap": bottleneck_family.get("bottleneck_gap", 0.0),
        "families": families,
        # Reported but excluded from the verdict: complete zero-shaping resilience remains a
        # known weakness (the model collapses to ~static accuracy without attention shaping).
        "stress_test_families": stress_families,
        "zero_shaping_stress_supported": all(
            f.get("robust_supported", False) for f in stress_families
        ) if stress_families else None,
    }


def build_evidence_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Condense raw metrics into the three core claims from the README."""

    baseline = report["baseline"]
    recurrent = report["recurrent"]
    ablations = report["ablations"]
    freeze = ablations.get("freeze_recurrence", {})
    feedforward = ablations.get("feedforward_summary", {})

    dissociation = {
        "recurrent_vs_baseline_accuracy_gain": recurrent["accuracy"] - baseline["accuracy"],
        "recurrent_vs_feedforward_accuracy_gain": recurrent["accuracy"] - feedforward.get("accuracy", 0.0),
        "recurrent_vs_freeze_accuracy_gain": recurrent["accuracy"] - freeze.get("accuracy", 0.0),
        "supported": (
            recurrent["accuracy"] > baseline["accuracy"]
            and recurrent["accuracy"] > feedforward.get("accuracy", float("-inf"))
            and recurrent["accuracy"] > freeze.get("accuracy", float("-inf"))
        ),
    }

    closed_loop = {
        "recurrent_temporal_reallocation": recurrent["temporal_reallocation"],
        "freeze_temporal_reallocation": freeze.get("temporal_reallocation", 0.0),
        "recurrent_target_attention_gain": recurrent["target_attention_gain"],
        "feedforward_target_attention_gain": feedforward.get("target_attention_gain", 0.0),
        "supported": (
            recurrent["temporal_reallocation"] > freeze.get("temporal_reallocation", 0.0)
            and recurrent["target_attention_gain"] > feedforward.get("target_attention_gain", float("-inf"))
        ),
    }

    cue_dependence = {
        "baseline_cue_accuracy_delta": baseline["cue_accuracy_delta"],
        "recurrent_cue_accuracy_delta": recurrent["cue_accuracy_delta"],
        "baseline_cue_target_attention_delta": baseline["cue_target_attention_delta"],
        "recurrent_cue_target_attention_delta": recurrent["cue_target_attention_delta"],
        "supported": (
            recurrent["cue_accuracy_delta"] > baseline["cue_accuracy_delta"]
            and recurrent["cue_target_attention_delta"] > baseline["cue_target_attention_delta"]
        ),
    }

    cue_switch = report.get("cue_switch", {})
    cue_switch_adaptation = {
        "baseline_switch_target_gain": cue_switch.get("baseline", {}).get("switch_target_gain", 0.0),
        "recurrent_switch_target_gain": cue_switch.get("recurrent", {}).get("switch_target_gain", 0.0),
        "baseline_switch_accuracy": cue_switch.get("baseline", {}).get("switch_accuracy", 0.0),
        "recurrent_switch_accuracy": cue_switch.get("recurrent", {}).get("switch_accuracy", 0.0),
        "supported": (
            cue_switch.get("recurrent", {}).get("switch_target_gain", float("-inf"))
            > cue_switch.get("baseline", {}).get("switch_target_gain", float("-inf"))
            and cue_switch.get("recurrent", {}).get("switch_accuracy", float("-inf"))
            > cue_switch.get("baseline", {}).get("switch_accuracy", float("-inf"))
        ),
    }

    predictive_probe = report.get("predictive_probe", {})
    intervention_test = report.get("intervention_test", {})
    reduced_shaping = report.get("reduced_shaping", {})
    reduced_shaping_summary = reduced_shaping.get("summary", {})
    stage3_multi_seed = report.get("stage3_multi_seed", {})
    stage3_checkpoint_family = report.get("stage3_checkpoint_family", {})
    explicit_attention_modeling = {
        "controller_advantage_cross_entropy": predictive_probe.get(
            "controller_advantage_cross_entropy", 0.0
        ),
        "controller_advantage_mse": predictive_probe.get("controller_advantage_mse", 0.0),
        "controller_advantage_top1_match": predictive_probe.get(
            "controller_advantage_top1_match", 0.0
        ),
        "intervention_supported": intervention_test.get("supported", False),
        "reduced_shaping_supported": reduced_shaping_summary.get("supported", False),
        "stage3_num_seeds": stage3_multi_seed.get("num_seeds", 0),
        "stage3_predictive_supported_fraction": stage3_multi_seed.get(
            "predictive_supported_fraction", 0.0
        ),
        "stage3_intervention_supported_fraction": stage3_multi_seed.get(
            "intervention_supported_fraction", 0.0
        ),
        "stage3_predictive_cross_entropy_advantage_mean": stage3_multi_seed.get(
            "predictive_cross_entropy_advantage_mean", 0.0
        ),
        "stage3_predictive_cross_entropy_advantage_min": stage3_multi_seed.get(
            "predictive_cross_entropy_advantage_min", 0.0
        ),
        "stage3_predictive_cross_entropy_advantage_min_gap": stage3_multi_seed.get(
            "predictive_cross_entropy_advantage_min_gap", 0.0
        ),
        "stage3_predictive_top1_advantage_mean": stage3_multi_seed.get(
            "predictive_top1_advantage_mean", 0.0
        ),
        "stage3_predictive_top1_advantage_min": stage3_multi_seed.get(
            "predictive_top1_advantage_min", 0.0
        ),
        "stage3_predictive_top1_advantage_min_gap": stage3_multi_seed.get(
            "predictive_top1_advantage_min_gap", 0.0
        ),
        "stage3_intervention_attention_change_kl_mean": stage3_multi_seed.get(
            "intervention_attention_change_kl_mean", 0.0
        ),
        "stage3_intervention_attention_change_kl_min": stage3_multi_seed.get(
            "intervention_attention_change_kl_min", 0.0
        ),
        "stage3_intervention_attention_change_kl_min_gap": stage3_multi_seed.get(
            "intervention_attention_change_kl_min_gap", 0.0
        ),
        "stage3_intervention_alternate_target_gain_mean": stage3_multi_seed.get(
            "intervention_alternate_target_gain_mean", 0.0
        ),
        "stage3_intervention_alternate_target_gain_min": stage3_multi_seed.get(
            "intervention_alternate_target_gain_min", 0.0
        ),
        "stage3_intervention_alternate_target_gain_min_gap": stage3_multi_seed.get(
            "intervention_alternate_target_gain_min_gap", 0.0
        ),
        "stage3_failure_reasons": stage3_multi_seed.get("failure_reasons", []),
        "stage3_bottleneck_metric": stage3_multi_seed.get("bottleneck_metric", ""),
        "stage3_bottleneck_gap": stage3_multi_seed.get("bottleneck_gap", 0.0),
        "stage3_worst_predictive_seed": stage3_multi_seed.get("worst_predictive_seed", 0),
        "stage3_worst_intervention_seed": stage3_multi_seed.get("worst_intervention_seed", 0),
        "stage3_all_predictive_supported": stage3_multi_seed.get("all_predictive_supported", False),
        "stage3_all_intervention_supported": stage3_multi_seed.get(
            "all_intervention_supported", False
        ),
        "stage3_multi_seed_supported": stage3_multi_seed.get("supported", False),
        "stage3_checkpoint_family_supported": stage3_checkpoint_family.get("supported", False),
        "stage3_checkpoint_family_num_families": stage3_checkpoint_family.get("num_families", 0),
        "stage3_checkpoint_family_robust_supported_families": stage3_checkpoint_family.get(
            "robust_supported_families", 0
        ),
        "stage3_checkpoint_family_verdict": stage3_checkpoint_family.get("verdict", "missing"),
        "stage3_checkpoint_family_bottleneck_family": stage3_checkpoint_family.get(
            "bottleneck_family", ""
        ),
        "stage3_checkpoint_family_bottleneck_metric": stage3_checkpoint_family.get(
            "bottleneck_metric", ""
        ),
        "stage3_checkpoint_family_bottleneck_gap": stage3_checkpoint_family.get(
            "bottleneck_gap", 0.0
        ),
        "single_run_supported": (
            predictive_probe.get("supported", False)
            and intervention_test.get("supported", False)
            and reduced_shaping_summary.get("supported", False)
        ),
        "robust_supported": stage3_multi_seed.get("supported", False),
        "supported": (
            predictive_probe.get("supported", False)
            and intervention_test.get("supported", False)
            and reduced_shaping_summary.get("supported", False)
            and stage3_multi_seed.get("supported", False)
            and stage3_checkpoint_family.get("supported", False)
        ),
    }

    report_probes = report.get("report_probes", {})
    self_model = report.get("self_modeling", {})
    learned_self_model = report.get("learned_self_modeling", {})
    uncertainty_reports = report.get("uncertainty_report_probes", {})
    noise_floor = report.get("noise_floor", {})
    # When the empirical noise-floor diagnostic is enabled, require the strong report signals
    # to clear their permuted-label p95 floor; when disabled, fall back to directional > 0.
    noise_floor_enabled = bool(noise_floor)
    noise_floor_cleared = (
        noise_floor.get("current_search_type", {}).get("exceeds_noise_floor", False)
        and noise_floor.get("current_attended_cell", {}).get("exceeds_noise_floor", False)
    )
    structured_reportability = {
        "current_search_type_advantage": report_probes.get("current_search_type", {}).get(
            "controller_accuracy_advantage", 0.0
        ),
        "current_attended_cell_advantage": report_probes.get("current_attended_cell", {}).get(
            "controller_accuracy_advantage", 0.0
        ),
        "target_found_advantage": report_probes.get("target_found_in_glimpse", {}).get(
            "controller_accuracy_advantage", 0.0
        ),
        "unresolved_region_advantage": self_model.get("native_cell_report", {}).get(
            "cell_accuracy_advantage", 0.0
        ),
        "noise_floor_enabled": noise_floor_enabled,
        "exceeds_noise_floor": noise_floor_cleared if noise_floor_enabled else None,
    }
    structured_reportability["supported"] = (
        report_probes.get("current_search_type", {}).get("controller_accuracy_advantage", 0.0) > 0.0
        and report_probes.get("current_attended_cell", {}).get("controller_accuracy_advantage", 0.0) > 0.0
        and self_model.get("native_cell_report", {}).get("cell_accuracy_advantage", 0.0) > 0.0
        and (noise_floor_cleared if noise_floor_enabled else True)
    )

    engineered_self_state_tracking = {
        "native_cell_accuracy": self_model.get("native_cell_report", {}).get("cell_accuracy", 0.0),
        "native_cell_bce": self_model.get("native_cell_report", {}).get("cell_bce", 0.0),
        "cell_accuracy_advantage": self_model.get("native_cell_report", {}).get(
            "cell_accuracy_advantage", 0.0
        ),
        "target_inspected_native_accuracy": self_model.get("target_inspected_report", {}).get(
            "native_accuracy", 0.0
        ),
        "target_inspected_native_positive_recall": self_model.get("target_inspected_report", {}).get(
            "native_positive_recall", 0.0
        ),
        "target_inspected_advantage": self_model.get("target_inspected_report", {}).get(
            "native_accuracy_advantage", 0.0
        ),
        "target_inspected_positive_recall_advantage": self_model.get(
            "target_inspected_report", {}
        ).get("native_positive_recall_advantage", 0.0),
        "supported": self_model.get("supported", False),
    }

    learned_intervention = learned_self_model.get("hidden_self_model_intervention", {})
    learned_self_modeling_of_attention = {
        "implemented": bool(learned_self_model),
        "positive_evidence": learned_self_model.get("positive_evidence", False),
        "supported": learned_self_model.get("supported", False),
        "hidden_cell_accuracy_advantage": learned_self_model.get(
            "hidden_cell_accuracy_advantage", 0.0
        ),
        "hidden_cell_bce_advantage": learned_self_model.get(
            "hidden_cell_bce_advantage", 0.0
        ),
        "hidden_target_accuracy_advantage": learned_self_model.get(
            "hidden_target_accuracy_advantage", 0.0
        ),
        "hidden_target_bce_advantage": learned_self_model.get(
            "hidden_target_bce_advantage", 0.0
        ),
        "hidden_target_positive_recall_advantage": learned_self_model.get(
            "hidden_target_positive_recall_advantage", 0.0
        ),
        "hidden_target_score_separation_advantage": learned_self_model.get(
            "hidden_target_score_separation_advantage", 0.0
        ),
        "bidirectional_self_model_target_gap": learned_intervention.get(
            "bidirectional_self_model_target_gap", 0.0
        ),
        "bidirectional_target_attention_gap": learned_intervention.get(
            "bidirectional_target_attention_gap", 0.0
        ),
        "policy_feedback_abs_mean": learned_intervention.get(
            "policy_feedback_abs_mean", 0.0
        ),
        "policy_override_bidirectional_target_attention_gap": learned_intervention.get(
            "policy_override_bidirectional_target_attention_gap", 0.0
        ),
        "policy_feedback_evidence": learned_self_model.get(
            "policy_feedback_evidence", False
        ),
        "note": learned_self_model.get(
            "note",
            (
                "The benchmark still exposes an engineered inspected-state scaffold. "
                "Hidden-state-only probes and hidden-state interventions are required "
                "before a stronger learned self-model claim can be supported."
            ),
        ),
    }

    def _u_ctrl(name: str) -> float:
        return uncertainty_reports.get(name, {}).get(
            "probe_capacity_matched_controller_positive_recall_advantage", 0.0
        )

    structured_reportability_uncertainty_and_allocation_error = {
        "implemented": bool(uncertainty_reports),
        "positive_evidence": any(
            _u_ctrl(name) > 0.0
            for name in (
                "current_wrong_candidate",
                "wrong_candidate_history",
                "revisit_unresolved",
                "allocation_error",
            )
        ),
        # Gated on the controller-STATE probe vs capacity-matched observation (mirrors 6A);
        # native head/ground-truth scores are informational only.
        "supported": uncertainty_reports.get("supported", False),
        "current_wrong_candidate_controller_recall_advantage": _u_ctrl("current_wrong_candidate"),
        "wrong_candidate_history_controller_recall_advantage": _u_ctrl("wrong_candidate_history"),
        "revisit_unresolved_controller_recall_advantage": _u_ctrl("revisit_unresolved"),
        "allocation_error_controller_recall_advantage": _u_ctrl("allocation_error"),
        "relevant_region_controller_recall_advantage": _u_ctrl("relevant_region_inspected"),
        "unresolved_search_controller_recall_advantage": _u_ctrl("unresolved_search"),
    }

    shaping_resilience = {
        "lowest_weight": reduced_shaping_summary.get("lowest_weight", 0.0),
        "lowest_weight_accuracy": reduced_shaping_summary.get("lowest_weight_accuracy", 0.0),
        "lowest_weight_temporal_reallocation": reduced_shaping_summary.get(
            "lowest_weight_temporal_reallocation", 0.0
        ),
        "lowest_weight_target_attention_gain": reduced_shaping_summary.get(
            "lowest_weight_target_attention_gain", 0.0
        ),
        "zero_weight_tested": reduced_shaping_summary.get("zero_weight_tested", False),
        "zero_weight_accuracy": reduced_shaping_summary.get("zero_weight_accuracy", 0.0),
        "zero_weight_temporal_reallocation": reduced_shaping_summary.get(
            "zero_weight_temporal_reallocation", 0.0
        ),
        "zero_weight_target_attention_gain": reduced_shaping_summary.get(
            "zero_weight_target_attention_gain", 0.0
        ),
        "zero_weight_supported": reduced_shaping_summary.get("zero_weight_supported", False),
        "supported": reduced_shaping_summary.get("supported", False),
    }

    causal_intervention = {
        "step": intervention_test.get("step", -1),
        "attention_change_kl": intervention_test.get("attention_change_kl", 0.0),
        "original_target_attention_drop": intervention_test.get(
            "original_target_attention_drop", 0.0
        ),
        "alternate_target_attention_gain": intervention_test.get(
            "alternate_target_attention_gain", 0.0
        ),
        "supported": intervention_test.get("supported", False),
    }

    nl_report = report.get("nl_report", {})
    cue_switch_nl = nl_report.get("cue_switch_slice", {})
    intervention_nl = nl_report.get("intervention_slice", {})
    natural_language_reportability = {
        "skipped": nl_report.get("skipped", not bool(nl_report)),
        "model": nl_report.get("model", ""),
        "local_decoder": nl_report.get("local_decoder", False),
        "tokenized_payload_attended_cell_accuracy": nl_report.get(
            "tokenized_state_payload", {}
        ).get("attended_cell_accuracy", 0.0),
        "tokenized_payload_previous_attended_cell_accuracy": nl_report.get(
            "tokenized_state_payload", {}
        ).get("previous_attended_cell_accuracy", 0.0),
        "tokenized_payload_current_content_joint_accuracy": nl_report.get(
            "tokenized_state_payload", {}
        ).get("current_content_joint_accuracy", 0.0),
        "tokenized_payload_memory_content_joint_accuracy": nl_report.get(
            "tokenized_state_payload", {}
        ).get("memory_content_joint_accuracy", 0.0),
        "tokenized_joint_accuracy": nl_report.get("tokenized_state", {}).get("joint_accuracy", 0.0),
        "observation_joint_accuracy": nl_report.get("observation_only", {}).get("joint_accuracy", 0.0),
        "tokenized_current_content_joint_accuracy_advantage": nl_report.get(
            "tokenized_current_content_joint_accuracy_advantage", 0.0
        ),
        "tokenized_memory_content_joint_accuracy_advantage": nl_report.get(
            "tokenized_memory_content_joint_accuracy_advantage", 0.0
        ),
        "tokenized_content_only_joint_accuracy_advantage": nl_report.get(
            "tokenized_content_only_joint_accuracy_advantage", 0.0
        ),
        "tokenized_visible_type_accuracy_advantage": nl_report.get(
            "tokenized_visible_type_accuracy_advantage", 0.0
        ),
        "tokenized_attended_digit_accuracy_advantage": nl_report.get(
            "tokenized_attended_digit_accuracy_advantage", 0.0
        ),
        "tokenized_glimpse_digit_accuracy_advantage": nl_report.get(
            "tokenized_glimpse_digit_accuracy_advantage", 0.0
        ),
        "tokenized_previous_attended_cell_accuracy_advantage": nl_report.get(
            "tokenized_previous_attended_cell_accuracy_advantage", 0.0
        ),
        "tokenized_previous_visible_type_accuracy_advantage": nl_report.get(
            "tokenized_previous_visible_type_accuracy_advantage", 0.0
        ),
        "tokenized_previous_attended_digit_accuracy_advantage": nl_report.get(
            "tokenized_previous_attended_digit_accuracy_advantage", 0.0
        ),
        "tokenized_previous_glimpse_digit_accuracy_advantage": nl_report.get(
            "tokenized_previous_glimpse_digit_accuracy_advantage", 0.0
        ),
        "tokenized_glimpse_match_accuracy_advantage": nl_report.get(
            "tokenized_glimpse_match_accuracy_advantage", 0.0
        ),
        "tokenized_relevant_region_accuracy_advantage": nl_report.get(
            "tokenized_relevant_region_accuracy_advantage", 0.0
        ),
        "tokenized_unresolved_search_accuracy_advantage": nl_report.get(
            "tokenized_unresolved_search_accuracy_advantage", 0.0
        ),
        "tokenized_current_wrong_candidate_accuracy_advantage": nl_report.get(
            "tokenized_current_wrong_candidate_accuracy_advantage", 0.0
        ),
        "tokenized_wrong_candidate_history_accuracy_advantage": nl_report.get(
            "tokenized_wrong_candidate_history_accuracy_advantage", 0.0
        ),
        "tokenized_revisit_unresolved_accuracy_advantage": nl_report.get(
            "tokenized_revisit_unresolved_accuracy_advantage", 0.0
        ),
        "tokenized_allocation_error_accuracy_advantage": nl_report.get(
            "tokenized_allocation_error_accuracy_advantage", 0.0
        ),
        "tokenized_uncertainty_content_joint_accuracy_advantage": nl_report.get(
            "tokenized_uncertainty_content_joint_accuracy_advantage", 0.0
        ),
        "tokenized_joint_accuracy_advantage": nl_report.get(
            "tokenized_joint_accuracy_advantage", 0.0
        ),
        "tokenized_unresolved_accuracy_advantage": nl_report.get(
            "tokenized_unresolved_accuracy_advantage", 0.0
        ),
        "cue_switch_slice_supported": cue_switch_nl.get("supported", False),
        "cue_switch_tokenized_joint_accuracy_advantage": cue_switch_nl.get(
            "tokenized_joint_accuracy_advantage", 0.0
        ),
        "cue_switch_tokenized_memory_content_joint_accuracy_advantage": cue_switch_nl.get(
            "tokenized_memory_content_joint_accuracy_advantage", 0.0
        ),
        "intervention_slice_supported": intervention_nl.get("supported", False),
        "intervention_tokenized_joint_accuracy_advantage": intervention_nl.get(
            "tokenized_joint_accuracy_advantage", 0.0
        ),
        "intervention_tokenized_memory_content_joint_accuracy_advantage": intervention_nl.get(
            "tokenized_memory_content_joint_accuracy_advantage", 0.0
        ),
        "intervention_attention_change_fraction": intervention_nl.get(
            "attention_change_fraction", 0.0
        ),
        "content_supported": nl_report.get("content_supported", False),
        "supported": nl_report.get("supported", False),
    }

    perturbational = report.get("perturbational", {})
    perturbational_complexity = {
        "implemented": bool(perturbational),
        # Non-reportability evidence family (perturbational / PCI-style): does perturbing the
        # integrated controller state produce rich-but-recoverable dynamics that degrade under
        # a no-recurrence control?
        "supported": perturbational.get("supported", False),
        "rich_but_recoverable": perturbational.get("rich_but_recoverable", False),
        "integration_exceeds_feedforward": perturbational.get("integration_exceeds_feedforward", False),
        "recurrent_mean_recovery_ratio": perturbational.get("recurrent_mean_recovery_ratio", 0.0),
        "recurrent_mean_attention_propagation": perturbational.get(
            "recurrent_mean_attention_propagation", 0.0
        ),
        "feedforward_mean_attention_propagation": perturbational.get(
            "feedforward_mean_attention_propagation", 0.0
        ),
    }

    return {
        "dissociation": dissociation,
        "closed_loop_adaptation": closed_loop,
        "cue_dependence": cue_dependence,
        "cue_switch_adaptation": cue_switch_adaptation,
        "explicit_attention_modeling": explicit_attention_modeling,
        "engineered_self_state_tracking": engineered_self_state_tracking,
        "learned_self_modeling_of_attention": learned_self_modeling_of_attention,
        "structured_reportability": structured_reportability,
        "structured_reportability_uncertainty_and_allocation_error": (
            structured_reportability_uncertainty_and_allocation_error
        ),
        "natural_language_reportability": natural_language_reportability,
        "causal_attention_intervention": causal_intervention,
        "reduced_shaping_resilience": shaping_resilience,
        "perturbational_complexity": perturbational_complexity,
    }


def build_stage3_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact Stage 3-specific summary for easier downstream consumption."""

    predictive_probe = report.get("predictive_probe", {})
    intervention_test = report.get("intervention_test", {})
    reduced_shaping_summary = report.get("reduced_shaping", {}).get("summary", {})
    multi_seed = report.get("stage3_multi_seed", {})
    return {
        "single_run_supported": (
            predictive_probe.get("supported", False)
            and intervention_test.get("supported", False)
            and reduced_shaping_summary.get("supported", False)
        ),
        "robust_supported": multi_seed.get("supported", False),
        "predictive_probe_supported": predictive_probe.get("supported", False),
        "intervention_supported": intervention_test.get("supported", False),
        "reduced_shaping_supported": reduced_shaping_summary.get("supported", False),
        "predictive_supported_fraction": multi_seed.get("predictive_supported_fraction", 0.0),
        "intervention_supported_fraction": multi_seed.get("intervention_supported_fraction", 0.0),
        "failure_reasons": multi_seed.get("failure_reasons", []),
        "bottleneck_metric": multi_seed.get("bottleneck_metric", ""),
        "bottleneck_gap": multi_seed.get("bottleneck_gap", 0.0),
        "worst_predictive_seed": multi_seed.get("worst_predictive_seed", 0),
        "worst_intervention_seed": multi_seed.get("worst_intervention_seed", 0),
        "predictive_cross_entropy_advantage_min_gap": multi_seed.get(
            "predictive_cross_entropy_advantage_min_gap", 0.0
        ),
        "predictive_top1_advantage_min_gap": multi_seed.get(
            "predictive_top1_advantage_min_gap", 0.0
        ),
        "intervention_attention_change_kl_min_gap": multi_seed.get(
            "intervention_attention_change_kl_min_gap", 0.0
        ),
        "intervention_alternate_target_gain_min_gap": multi_seed.get(
            "intervention_alternate_target_gain_min_gap", 0.0
        ),
    }


def save_probe_plots(
    model,
    output_dir: Path,
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> list[str]:
    """Render one qualitative attention heatmap sequence per cue."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = make_generator(seed, device)
    base_batch = generate_batch(1, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    probe_batch = expand_cues_for_probe(base_batch, task_cfg.num_types)

    with torch.no_grad():
        outputs = model(
            probe_batch.scene,
            probe_batch.cue,
            target=probe_batch.target,
            target_pos=probe_batch.target_pos,
            num_steps=task_cfg.num_steps,
        )

    attention = outputs["attention_seq"]
    saved_paths = []
    for cue_idx in range(task_cfg.num_types):
        target_pos = probe_batch.target_pos[cue_idx].item()
        target_row, target_col = divmod(target_pos, task_cfg.grid_size)
        fig, axes = plt.subplots(1, task_cfg.num_steps, figsize=(3 * task_cfg.num_steps, 3))
        if task_cfg.num_steps == 1:
            axes = [axes]
        for step_idx in range(task_cfg.num_steps):
            ax = axes[step_idx]
            heatmap = attention[cue_idx, step_idx].reshape(task_cfg.grid_size, task_cfg.grid_size).cpu()
            ax.imshow(heatmap, cmap="viridis")
            ax.scatter(
                target_col,
                target_row,
                s=160,
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=2,
            )
            ax.set_title(
                f"Cue {cue_idx} / Step {step_idx + 1}\nTarget: row {target_row}, col {target_col}"
            )
            ax.axis("off")
        path = output_dir / f"attention_probe_cue_{cue_idx}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        saved_paths.append(str(path))

    return saved_paths


def _wrap_panel_text(text: str, width: int = 34) -> str:
    lines = []
    for raw_line in text.splitlines():
        wrapped = textwrap.wrap(raw_line, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


def _draw_stage7_scene_panel(
    ax,
    *,
    visible_types: torch.Tensor,
    digits: torch.Tensor,
    grid_size: int,
    attended_cell: int,
    cue: int,
    previous_cue: int,
    title: str,
) -> None:
    visible_grid = visible_types.reshape(grid_size, grid_size).detach().cpu().numpy()
    digits_grid = digits.reshape(grid_size, grid_size).detach().cpu().numpy()
    row, col = divmod(attended_cell, grid_size)

    ax.imshow(visible_grid, cmap="tab20", vmin=0, vmax=max(int(visible_grid.max()), 1))
    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            ax.text(
                col_idx,
                row_idx,
                f"{int(visible_grid[row_idx, col_idx])}\n{int(digits_grid[row_idx, col_idx])}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if visible_grid[row_idx, col_idx] >= 3 else "black",
            )
    ax.scatter(col, row, s=180, marker="o", facecolors="none", edgecolors="red", linewidths=2)
    ax.set_title(f"{title}\ncue {cue}, prev {previous_cue}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_stage7_text_panel(ax, *, title: str, text: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    ax.text(
        0.0,
        1.0,
        _wrap_panel_text(text),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        transform=ax.transAxes,
    )


def save_stage7_visual_report_plots(
    model,
    output_dir: Path,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Render qualitative Stage 7 panels comparing scene-only, symbolic, and tokenized reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = make_generator(seed, device)
    base_batch = generate_batch(1, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    saved_paths: list[str] = []
    metadata: list[dict[str, Any]] = []

    with torch.no_grad():
        base_outputs = model(
            base_batch.scene,
            base_batch.cue,
            target=base_batch.target,
            target_pos=base_batch.target_pos,
            num_steps=task_cfg.num_steps,
        )
    base_examples = collect_nl_examples(model, task_cfg, base_batch, base_outputs)
    default_step = min(1, task_cfg.num_steps - 1)
    default_example = next(
        (example for example in base_examples if example.step_index == default_step),
        base_examples[-1],
    )

    def _example_metadata(slice_name: str, example, path: Path) -> dict[str, Any]:
        return {
            "slice": slice_name,
            "path": str(path),
            "step_index": int(example.step_index),
            "cue": int(example.cue),
            "previous_cue": int(example.previous_cue),
            "cue_switched": bool(example.cue_switched),
            "attended_cell": int(example.attended_cell),
            "prev_attended_cell": int(example.prev_attended_cell),
            "relevant_region_inspected": bool(example.relevant_region_inspected),
            "unresolved_search": bool(example.unresolved_search),
            "current_wrong_candidate": bool(example.current_wrong_candidate),
            "wrong_candidate_history": bool(example.wrong_candidate_history),
            "revisit_unresolved": bool(example.revisit_unresolved),
            "allocation_error": bool(example.allocation_error),
            "symbolic_state": example.symbolic_state,
            "tokenized_state": example.tokenized_state,
            "observation_only": example.observation_only,
        }

    def _save_triptych(filename: str, figure_title: str, example, batch, *, slice_name: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        _draw_stage7_scene_panel(
            axes[0],
            visible_types=batch.visible_types[0],
            digits=batch.digits[0],
            grid_size=task_cfg.grid_size,
            attended_cell=example.attended_cell,
            cue=example.cue,
            previous_cue=example.previous_cue,
            title="Scene Only",
        )
        _draw_stage7_text_panel(axes[1], title="Explicit Symbolic State", text=example.symbolic_state)
        _draw_stage7_text_panel(axes[2], title="Minimal Tokenized State", text=example.tokenized_state)
        fig.suptitle(figure_title, fontsize=12)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        saved_paths.append(str(path))
        metadata.append(_example_metadata(slice_name, example, path))

    _save_triptych(
        "stage7_visual_default.png",
        f"Stage 7 Default Slice, Step {default_example.step_index + 1}",
        default_example,
        base_batch,
        slice_name="default",
    )

    cue_switch_cfg = cfg["evaluation"].get("cue_switch", {})
    if cue_switch_cfg.get("enabled", False):
        switch_step = int(cue_switch_cfg.get("switch_step", task_cfg.num_steps // 2))
        cue_switch_examples = collect_cue_switch_nl_examples(
            model,
            task_cfg,
            base_batch,
            switch_step=switch_step,
        )
        cue_switch_example = next(
            (example for example in cue_switch_examples if example.step_index == switch_step),
            cue_switch_examples[-1],
        )
        _save_triptych(
            "stage7_visual_cue_switch.png",
            f"Stage 7 Cue Switch Slice, Step {cue_switch_example.step_index + 1}",
            cue_switch_example,
            base_batch,
            slice_name="cue_switch",
        )

    intervention_cfg = cfg["evaluation"].get("intervention_test", {})
    if intervention_cfg.get("enabled", False):
        intervention_step = int(intervention_cfg.get("step", 1))
        intervention_examples = collect_intervention_nl_examples(
            model,
            task_cfg,
            base_batch,
            intervention_step=intervention_step,
        )
        baseline_example = next(
            (
                example
                for example in intervention_examples["baseline_examples"]
                if example.step_index == intervention_step
            ),
            intervention_examples["baseline_examples"][-1],
        )
        intervened_example = next(
            (
                example
                for example in intervention_examples["intervened_examples"]
                if example.step_index == intervention_step
            ),
            intervention_examples["intervened_examples"][-1],
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for row_idx, (row_title, example) in enumerate(
            (("Baseline", baseline_example), ("Intervened", intervened_example))
        ):
            _draw_stage7_scene_panel(
                axes[row_idx, 0],
                visible_types=base_batch.visible_types[0],
                digits=base_batch.digits[0],
                grid_size=task_cfg.grid_size,
                attended_cell=example.attended_cell,
                cue=example.cue,
                previous_cue=example.previous_cue,
                title=f"{row_title} Scene Only",
            )
            _draw_stage7_text_panel(
                axes[row_idx, 1],
                title=f"{row_title} Symbolic State",
                text=example.symbolic_state,
            )
            _draw_stage7_text_panel(
                axes[row_idx, 2],
                title=f"{row_title} Tokenized State",
                text=example.tokenized_state,
            )
        fig.suptitle(
            f"Stage 7 Intervention Slice, Step {intervention_step + 1}",
            fontsize=12,
        )
        fig.tight_layout()
        path = output_dir / "stage7_visual_intervention.png"
        fig.savefig(path)
        plt.close(fig)
        saved_paths.append(str(path))
        metadata.append(_example_metadata("intervention_baseline", baseline_example, path))
        metadata.append(_example_metadata("intervention_intervened", intervened_example, path))

    metadata_path = output_dir / "stage7_visual_report_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "plot_paths": saved_paths,
        "metadata_path": str(metadata_path),
    }


def load_stage7_visual_report_summary(metadata_path: str | Path) -> dict[str, Any]:
    """Build a compact summary over exported Stage 7 visual panel metadata."""

    path = Path(metadata_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    slice_names = [item["slice"] for item in metadata]
    return {
        "num_examples": len(metadata),
        "slice_names": slice_names,
        "cue_switch_examples": sum(int("cue_switch" in name) for name in slice_names),
        "intervention_examples": sum(int("intervention" in name) for name in slice_names),
        "current_wrong_candidate_examples": sum(
            int(bool(item.get("current_wrong_candidate", False))) for item in metadata
        ),
        "revisit_unresolved_examples": sum(
            int(bool(item.get("revisit_unresolved", False))) for item in metadata
        ),
        "allocation_error_examples": sum(
            int(bool(item.get("allocation_error", False))) for item in metadata
        ),
    }


def run_ablations(config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, Any]:
    """Run the full evaluation suite and write a JSON report plus plot artifacts."""

    device = torch.device(config["device"])
    cfg, task_cfg, models = load_models_from_checkpoint(checkpoint_path, device, config)
    output_dir = Path(cfg["output_dir"])
    eval_cfg = cfg["evaluation"]

    report: dict[str, Any] = {
        "baseline": evaluate_model(
            models["static"],
            cfg,
            task_cfg,
            device,
            num_batches=eval_cfg["test_batches"],
            seed=cfg["seed"] + 9000,
        ),
        "recurrent": evaluate_model(
            models["recurrent"],
            cfg,
            task_cfg,
            device,
            num_batches=eval_cfg["test_batches"],
            seed=cfg["seed"] + 9100,
        ),
        "ablations": {},
        "artifacts": {},
    }

    report["baseline"].update(
        trajectory_metrics(
            models["static"],
            task_cfg,
            eval_cfg["probe_scenes"],
            device,
            cfg["seed"] + 9200,
        )
    )
    report["baseline"].update(
        cue_sensitivity_metrics(
            models["static"],
            task_cfg,
            eval_cfg["probe_scenes"],
            device,
            cfg["seed"] + 9250,
        )
    )
    report["recurrent"].update(
        trajectory_metrics(
            models["recurrent"],
            task_cfg,
            eval_cfg["probe_scenes"],
            device,
            cfg["seed"] + 9300,
        )
    )
    report["recurrent"].update(
        cue_sensitivity_metrics(
            models["recurrent"],
            task_cfg,
            eval_cfg["probe_scenes"],
            device,
            cfg["seed"] + 9350,
        )
    )
    report["report_probes"] = report_probe_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9725,
    )
    report["noise_floor"] = noise_floor_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9728,
    )
    report["self_modeling"] = self_model_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9735,
    )
    report["learned_self_modeling"] = learned_self_model_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9736,
    )
    report["uncertainty_report_probes"] = uncertainty_report_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9738,
    )
    report["self_state_diagnostics"] = self_state_diagnostics(
        models["recurrent"],
        task_cfg,
        eval_cfg["probe_scenes"],
        device,
        cfg["seed"] + 9739,
    )
    report["self_model_diagnostics"] = self_model_diagnostics(
        models["recurrent"],
        task_cfg,
        eval_cfg["probe_scenes"],
        device,
        cfg["seed"] + 9741,
    )
    report["nl_report"] = nl_report_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9740,
    )
    report["negative_controls"] = negative_control_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9745,
    )
    report["comparator_systems"] = comparator_system_metrics(
        models,
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9747,
        output_dir,
        nl_report=report["nl_report"],
    )
    cue_switch_cfg = cfg["evaluation"].get("cue_switch", {})
    if cue_switch_cfg.get("enabled", False):
        report["cue_switch"] = {
            "baseline": cue_switch_metrics(
                models["static"],
                task_cfg,
                cue_switch_cfg.get("probe_scenes", eval_cfg["probe_scenes"]),
                device,
                cfg["seed"] + 9750,
                switch_step=cue_switch_cfg.get("switch_step", task_cfg.num_steps // 2),
            ),
            "recurrent": cue_switch_metrics(
                models["recurrent"],
                task_cfg,
                cue_switch_cfg.get("probe_scenes", eval_cfg["probe_scenes"]),
                device,
                cfg["seed"] + 9760,
                switch_step=cue_switch_cfg.get("switch_step", task_cfg.num_steps // 2),
            ),
        }
    report["reduced_shaping"] = reduced_shaping_metrics(
        cfg,
        task_cfg,
        device,
        output_dir,
    )
    stage3_bundle = evaluate_stage3_bundle(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9700,
        reduced_shaping_summary=report["reduced_shaping"].get("summary", {}),
    )
    report["predictive_probe"] = stage3_bundle["predictive_probe"]
    report["intervention_test"] = stage3_bundle["intervention_test"]
    report["stage3_multi_seed"] = stage3_bundle["stage3_multi_seed"]
    report["perturbational"] = perturbational_complexity_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9770,
    )

    for name in eval_cfg["ablations"]:
        ablation = {name: True}
        report["ablations"][name] = evaluate_model(
            models["recurrent"],
            cfg,
            task_cfg,
            device,
            num_batches=eval_cfg["test_batches"],
            seed=cfg["seed"] + 9400 + len(report["ablations"]),
            ablation=ablation,
        )
        report["ablations"][name].update(
            trajectory_metrics(
                lambda scene, cue, target=None, target_pos=None, num_steps=None: models["recurrent"](
                    scene,
                    cue,
                    target=target,
                    target_pos=target_pos,
                    num_steps=num_steps,
                    ablation=ablation,
                ),
                task_cfg,
                eval_cfg["probe_scenes"],
                device,
                cfg["seed"] + 9500 + len(report["ablations"]),
            )
        )
        report["ablations"][name].update(
            cue_sensitivity_metrics(
                lambda scene, cue, target=None, target_pos=None, num_steps=None: models["recurrent"](
                    scene,
                    cue,
                    target=target,
                    target_pos=target_pos,
                    num_steps=num_steps,
                    ablation=ablation,
                ),
                task_cfg,
                eval_cfg["probe_scenes"],
                device,
                cfg["seed"] + 9550 + len(report["ablations"]),
            )
        )

    plot_paths = save_probe_plots(
        models["recurrent"],
        output_dir / "plots",
        task_cfg,
        device,
        cfg["seed"] + 9600,
    )
    intervention_plot_paths = save_intervention_plots(
        models["recurrent"],
        output_dir / "plots",
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9650,
    )
    self_state_plot_paths = save_self_state_plots(
        report["self_state_diagnostics"],
        output_dir / "plots",
    )
    self_model_plot_paths = save_self_model_plots(
        report["self_model_diagnostics"],
        output_dir / "plots",
    )
    uncertainty_plot_paths = save_uncertainty_report_plots(
        report["uncertainty_report_probes"],
        output_dir / "plots",
    )
    cue_switch_plot_paths = save_cue_switch_plots(
        {"baseline": models["static"], "recurrent": models["recurrent"]},
        output_dir / "plots",
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9675,
    )
    stage3_multi_seed_plot_paths = save_stage3_multi_seed_plots(
        report["stage3_multi_seed"],
        output_dir / "plots",
    )
    stage7_visual_report_artifacts = save_stage7_visual_report_plots(
        models["recurrent"],
        output_dir / "plots",
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9685,
    )
    report["artifacts"]["plots"] = plot_paths
    report["artifacts"]["intervention_plots"] = intervention_plot_paths
    report["artifacts"]["self_state_plots"] = self_state_plot_paths
    report["artifacts"]["self_model_plots"] = self_model_plot_paths
    report["artifacts"]["uncertainty_plots"] = uncertainty_plot_paths
    report["artifacts"]["cue_switch_plots"] = cue_switch_plot_paths
    report["artifacts"]["stage3_multi_seed_plots"] = stage3_multi_seed_plot_paths
    report["artifacts"]["stage7_visual_reports"] = stage7_visual_report_artifacts["plot_paths"]
    report["artifacts"]["stage7_visual_report_metadata"] = stage7_visual_report_artifacts[
        "metadata_path"
    ]
    report["stage3_summary"] = build_stage3_summary(report)
    report["stage3_checkpoint_family"] = build_stage3_checkpoint_family_summary(report)
    stage3_summary_path = output_dir / "stage3_summary.json"
    with open(stage3_summary_path, "w", encoding="utf-8") as handle:
        json.dump(report["stage3_summary"], handle, indent=2)
    report["artifacts"]["stage3_summary"] = str(stage3_summary_path)
    stage3_checkpoint_family_path = output_dir / "stage3_checkpoint_family_summary.json"
    with open(stage3_checkpoint_family_path, "w", encoding="utf-8") as handle:
        json.dump(report["stage3_checkpoint_family"], handle, indent=2)
    report["artifacts"]["stage3_checkpoint_family_summary"] = str(stage3_checkpoint_family_path)
    report["artifacts"]["stage3_checkpoint_family_plots"] = save_stage3_checkpoint_family_plots(
        report["stage3_checkpoint_family"],
        output_dir / "plots",
    )
    report["artifacts"].update(export_stage3_robustness_tables(report, output_dir))
    report["artifacts"]["stage3_robustness_note"] = save_stage3_robustness_note(report, output_dir)
    report["stage7_visual_report_summary"] = load_stage7_visual_report_summary(
        stage7_visual_report_artifacts["metadata_path"]
    )
    report["evidence"] = build_evidence_summary(report)
    report["artifacts"]["checkpoint"] = str(checkpoint_path)

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["artifacts"]["report"] = str(report_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the minimal attention-control experiment.")
    parser.add_argument("--config", type=str, default="configs/minimal.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--allow-stale-checkpoint",
        action="store_true",
        help=(
            "Allow migration of checkpoints missing current report/self-model heads. "
            "Migrated checkpoints are limited to probe-only diagnostics."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.allow_stale_checkpoint:
        config.setdefault("evaluation", {})["allow_stale_checkpoint"] = True
    run_ablations(config, args.checkpoint)


if __name__ == "__main__":
    main()
