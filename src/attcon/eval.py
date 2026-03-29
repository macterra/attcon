from __future__ import annotations

"""Evaluation, ablation, and reporting utilities for the attention-control demo."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from torch import nn
import torch

from .data import TaskConfig, expand_cues_for_probe, generate_batch
from .models import ModelConfig, RecurrentAttentionController, StaticAttentionBaseline
from .train import deep_update, evaluate_model, load_config, make_generator


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

    return {
        "dissociation": dissociation,
        "closed_loop_adaptation": closed_loop,
        "cue_dependence": cue_dependence,
        "explicit_attention_modeling": explicit_attention_modeling,
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
    report["evidence"] = build_evidence_summary(report)
    report["artifacts"]["plots"] = plot_paths
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
