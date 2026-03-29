from __future__ import annotations

"""Evaluation, ablation, and reporting utilities for the attention-control demo."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from torch import nn
import torch

from .data import TaskConfig, expand_cues_for_probe, generate_batch
from .models import ModelConfig, RecurrentAttentionController, StaticAttentionBaseline
from .nl_report import OpenAI, collect_nl_examples, load_dotenv, run_nl_report_mode
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
        model.load_state_dict(payload["models"][name])
        model.eval()
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
                num_steps=task_cfg.num_steps,
            )

        inspection_true = outputs["inspection_seq"]
        inspection_pred = outputs["self_model_seq"]
        prev_observation = torch.zeros_like(outputs["observation_seq"])
        prev_observation[:, 1:] = outputs["observation_seq"][:, :-1]

        target_pos = batch.target_pos[:, None, None].expand(-1, task_cfg.num_steps, 1)
        target_true = inspection_true.gather(2, target_pos).squeeze(-1)
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
        train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
        test_pred = (torch.sigmoid(test_logits) >= 0.5).float()
        train_positive = train_labels == 1.0
        test_positive = test_labels == 1.0
        return {
            "train_bce": torch.nn.functional.binary_cross_entropy_with_logits(train_logits, train_labels).item(),
            "test_bce": torch.nn.functional.binary_cross_entropy_with_logits(test_logits, test_labels).item(),
            "train_accuracy": (train_pred == train_labels).float().mean().item(),
            "test_accuracy": (test_pred == test_labels).float().mean().item(),
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
        },
    )
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

    return {
        "controller_state_probe": controller_probe,
        "observation_only_probe": observation_probe,
        "controller_advantage_cross_entropy": (
            observation_probe["test_cross_entropy"] - controller_probe["test_cross_entropy"]
        ),
        "controller_advantage_mse": observation_probe["test_mse"] - controller_probe["test_mse"],
        "controller_advantage_top1_match": (
            controller_probe["test_top1_match"] - observation_probe["test_top1_match"]
        ),
        "supported": (
            controller_probe["test_cross_entropy"] < observation_probe["test_cross_entropy"]
            and controller_probe["test_mse"] < observation_probe["test_mse"]
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
    found_obs = _train_binary_probe(
        train["observation_features"],
        train["found_target_labels"],
        test["observation_features"],
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
            "controller_accuracy_advantage": cue_state["test_accuracy"] - cue_obs["test_accuracy"],
        },
        "current_attended_cell": {
            "controller_state_probe": attended_state,
            "observation_only_probe": attended_obs,
            "controller_accuracy_advantage": attended_state["test_accuracy"] - attended_obs["test_accuracy"],
        },
        "target_found_in_glimpse": {
            "controller_state_probe": found_state,
            "observation_only_probe": found_obs,
            "controller_accuracy_advantage": found_state["test_accuracy"] - found_obs["test_accuracy"],
            "controller_positive_recall_advantage": (
                found_state["test_positive_recall"] - found_obs["test_positive_recall"]
            ),
        },
        "supported": (
            cue_state["test_accuracy"] > cue_obs["test_accuracy"]
            and attended_state["test_accuracy"] > attended_obs["test_accuracy"]
            and found_state["test_positive_recall"] > found_obs["test_positive_recall"]
        ),
    }


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


def nl_report_metrics(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Evaluate natural-language reporting from tokenized internal state."""

    nl_cfg = cfg["evaluation"].get(
        "nl_report",
        {
            "enabled": False,
            "model": "gpt-5-mini",
            "dotenv_path": ".env",
            "calibration_examples": 4,
            "evaluation_examples": 4,
            "probe_scenes": 2,
            "max_output_tokens": 240,
        },
    )
    if not nl_cfg.get("enabled", False):
        return {}

    load_dotenv(nl_cfg.get("dotenv_path", ".env"))
    if OpenAI is None:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "openai dependency is not installed",
        }
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "enabled": True,
            "skipped": True,
            "reason": "OPENAI_API_KEY is not set",
        }

    batch_size = nl_cfg.get("probe_scenes", cfg["evaluation"]["probe_scenes"])
    generator = make_generator(seed, device)
    batch = generate_batch(batch_size, task_cfg.num_steps, task_cfg, generator=generator, device=device)
    with torch.no_grad():
        outputs = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            num_steps=task_cfg.num_steps,
        )

    examples = collect_nl_examples(model, task_cfg, batch, outputs)
    calibration_count = int(nl_cfg.get("calibration_examples", 4))
    evaluation_count = int(nl_cfg.get("evaluation_examples", 4))
    required_examples = calibration_count + evaluation_count
    if len(examples) < required_examples:
        return {
            "enabled": True,
            "skipped": True,
            "reason": (
                f"not enough examples for nl_report: need {required_examples}, have {len(examples)}"
            ),
        }

    calibration_examples = examples[:calibration_count]
    evaluation_examples = examples[calibration_count:required_examples]
    model_name = nl_cfg.get("model", "gpt-5-mini")
    max_output_tokens = int(nl_cfg.get("max_output_tokens", 240))
    modes = ("tokenized_state", "symbolic_state", "observation_only")
    try:
        results = {
            mode: run_nl_report_mode(
                mode=mode,
                model_name=model_name,
                calibration_examples=calibration_examples,
                evaluation_examples=evaluation_examples,
                grid_size=task_cfg.grid_size,
                max_output_tokens=max_output_tokens,
            )
            for mode in modes
        }
    except Exception as exc:  # pragma: no cover - network/runtime behavior
        return {
            "enabled": True,
            "skipped": True,
            "reason": f"nl_report request failed: {type(exc).__name__}: {exc}",
            "model": model_name,
        }

    tokenized = results["tokenized_state"]
    symbolic = results["symbolic_state"]
    observation = results["observation_only"]
    return {
        "enabled": True,
        "skipped": False,
        "model": model_name,
        "calibration_examples": calibration_count,
        "evaluation_examples": evaluation_count,
        "tokenized_state": tokenized,
        "symbolic_state": symbolic,
        "observation_only": observation,
        "tokenized_joint_accuracy_advantage": (
            tokenized["joint_accuracy"] - observation["joint_accuracy"]
        ),
        "tokenized_unresolved_accuracy_advantage": (
            tokenized["unresolved_cells_accuracy"] - observation["unresolved_cells_accuracy"]
        ),
        "symbolic_joint_accuracy_advantage": (
            symbolic["joint_accuracy"] - observation["joint_accuracy"]
        ),
        "supported": (
            tokenized["joint_accuracy"] > observation["joint_accuracy"]
            and tokenized["unresolved_cells_accuracy"] >= observation["unresolved_cells_accuracy"]
        ),
    }


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

    return {
        "step": step,
        "attention_change_kl": symmetric_kl(base_step_attention, intervened_step_attention).mean().item(),
        "original_target_attention_drop": base_target_attention - intervened_target_attention,
        "alternate_target_attention_gain": intervened_alt_target_attention - base_alt_target_attention,
        "baseline_target_attention": base_target_attention,
        "intervened_target_attention": intervened_target_attention,
        "baseline_alternate_target_attention": base_alt_target_attention,
        "intervened_alternate_target_attention": intervened_alt_target_attention,
        "supported": (
            intervened_alt_target_attention > base_alt_target_attention
            and base_target_attention > intervened_target_attention
            and symmetric_kl(base_step_attention, intervened_step_attention).mean().item() > 0.0
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

    shaping_dir = output_dir / "reduced_shaping"
    shaping_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for weight_idx, weight in enumerate(weights):
        variant_cfg = deep_update(
            cfg,
            {
                "seed": cfg["seed"] + 20000 + weight_idx * 100,
                "output_dir": str(shaping_dir / f"attn_weight_{str(weight).replace('.', '_')}"),
                "training": {
                    "attention_target_weight": float(weight),
                },
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
        results[str(weight)] = variant_report

    baseline_weight_key = str(min(weights))
    baseline_variant = results[baseline_weight_key]
    results["summary"] = {
        "lowest_weight": float(baseline_weight_key),
        "lowest_weight_accuracy": baseline_variant["accuracy"],
        "lowest_weight_temporal_reallocation": baseline_variant["temporal_reallocation"],
        "lowest_weight_target_attention_gain": baseline_variant["target_attention_gain"],
        "supported": (
            baseline_variant["temporal_reallocation"] > 0.0
            and baseline_variant["target_attention_gain"] > 0.0
            and baseline_variant["accuracy"] > 0.1
        ),
    }
    return results


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
    explicit_attention_modeling = {
        "controller_advantage_cross_entropy": predictive_probe.get(
            "controller_advantage_cross_entropy", 0.0
        ),
        "controller_advantage_mse": predictive_probe.get("controller_advantage_mse", 0.0),
        "controller_advantage_top1_match": predictive_probe.get(
            "controller_advantage_top1_match", 0.0
        ),
        "supported": predictive_probe.get("supported", False),
    }

    report_probes = report.get("report_probes", {})
    self_model = report.get("self_modeling", {})
    reportable_internal_content = {
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
        "supported": report_probes.get("supported", False),
    }
    reportable_internal_content["supported"] = (
        report_probes.get("current_search_type", {}).get("controller_accuracy_advantage", 0.0) > 0.0
        and report_probes.get("current_attended_cell", {}).get("controller_accuracy_advantage", 0.0) > 0.0
        and self_model.get("native_cell_report", {}).get("cell_accuracy_advantage", 0.0) > 0.0
    )

    self_modeling_of_attention = {
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

    reduced_shaping = report.get("reduced_shaping", {})
    reduced_shaping_summary = reduced_shaping.get("summary", {})
    shaping_resilience = {
        "lowest_weight": reduced_shaping_summary.get("lowest_weight", 0.0),
        "lowest_weight_accuracy": reduced_shaping_summary.get("lowest_weight_accuracy", 0.0),
        "lowest_weight_temporal_reallocation": reduced_shaping_summary.get(
            "lowest_weight_temporal_reallocation", 0.0
        ),
        "lowest_weight_target_attention_gain": reduced_shaping_summary.get(
            "lowest_weight_target_attention_gain", 0.0
        ),
        "supported": reduced_shaping_summary.get("supported", False),
    }

    intervention_test = report.get("intervention_test", {})
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
    natural_language_reportability = {
        "skipped": nl_report.get("skipped", not bool(nl_report)),
        "model": nl_report.get("model", ""),
        "tokenized_joint_accuracy": nl_report.get("tokenized_state", {}).get("joint_accuracy", 0.0),
        "observation_joint_accuracy": nl_report.get("observation_only", {}).get("joint_accuracy", 0.0),
        "tokenized_joint_accuracy_advantage": nl_report.get(
            "tokenized_joint_accuracy_advantage", 0.0
        ),
        "tokenized_unresolved_accuracy_advantage": nl_report.get(
            "tokenized_unresolved_accuracy_advantage", 0.0
        ),
        "supported": nl_report.get("supported", False),
    }

    return {
        "dissociation": dissociation,
        "closed_loop_adaptation": closed_loop,
        "cue_dependence": cue_dependence,
        "cue_switch_adaptation": cue_switch_adaptation,
        "explicit_attention_modeling": explicit_attention_modeling,
        "self_modeling_of_attention": self_modeling_of_attention,
        "reportable_internal_content": reportable_internal_content,
        "natural_language_reportability": natural_language_reportability,
        "causal_attention_intervention": causal_intervention,
        "reduced_shaping_resilience": shaping_resilience,
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
    report["predictive_probe"] = predictive_probe_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9700,
    )
    report["report_probes"] = report_probe_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9725,
    )
    report["self_modeling"] = self_model_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9735,
    )
    report["nl_report"] = nl_report_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9740,
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
    report["intervention_test"] = intervention_test_metrics(
        models["recurrent"],
        cfg,
        task_cfg,
        device,
        cfg["seed"] + 9800,
    )
    report["reduced_shaping"] = reduced_shaping_metrics(
        cfg,
        task_cfg,
        device,
        output_dir,
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
                lambda scene, cue, target=None, num_steps=None: models["recurrent"](
                    scene,
                    cue,
                    target=target,
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
                lambda scene, cue, target=None, num_steps=None: models["recurrent"](
                    scene,
                    cue,
                    target=target,
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
    report["evidence"] = build_evidence_summary(report)
    report["artifacts"]["plots"] = plot_paths
    report["artifacts"]["intervention_plots"] = intervention_plot_paths
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
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    run_ablations(config, args.checkpoint)


if __name__ == "__main__":
    main()
