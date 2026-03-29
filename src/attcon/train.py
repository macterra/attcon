from __future__ import annotations

"""Training entrypoint and optimization utilities for the benchmark."""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from .data import TaskConfig, generate_batch
from .models import ModelConfig, RecurrentAttentionController, StaticAttentionBaseline


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "output_dir": "outputs/minimal",
    "device": "cpu",
    "task": TaskConfig().to_dict(),
    "model": {
        "hidden_size": 32,
        "cue_embedding_dim": 8,
        "scene_embedding_dim": 16,
        "temperature": 0.25,
    },
    "training": {
        "batch_size": 128,
        "train_steps": 500,
        "val_batches": 20,
        "val_interval": 100,
        "log_interval": 25,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "aux_loss_weight": 0.15,
        "attention_target_weight": 1.0,
        "attention_entropy_weight": 0.0,
    },
    "evaluation": {
        "test_batches": 40,
        "probe_scenes": 4,
        "predictive_probe": {
            "train_batches": 12,
            "test_batches": 6,
            "epochs": 60,
            "learning_rate": 0.05,
        },
        "intervention_test": {
            "enabled": True,
            "probe_scenes": 4,
            "step": 2,
        },
        "reduced_shaping": {
            "enabled": True,
            "weights": [0.25, 0.0],
        },
        "ablations": [
            "freeze_recurrence",
            "feedforward_summary",
            "zero_prev_glimpse",
            "zero_prev_loss",
            "zero_prev_confidence",
            "zero_prev_attention",
            "shuffle_cue",
        ],
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a partial config onto the default config."""

    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it with defaults."""

    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return deep_update(config, loaded)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def compute_batch_metrics(outputs: dict[str, torch.Tensor], batch) -> dict[str, float]:
    """Compute the headline metrics used during train/validation logging."""

    logits = outputs["logits"]
    prediction = logits.argmax(dim=-1)
    accuracy = (prediction == batch.target).float().mean().item()

    attention = outputs["attention_seq"]
    target_attention = attention.gather(
        2, batch.target_pos[:, None, None].expand(-1, attention.shape[1], 1)
    ).mean().item()
    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1).mean().item()

    return {
        "accuracy": accuracy,
        "target_attention": target_attention,
        "attention_entropy": entropy,
    }


def compute_attention_target_loss(
    attention_seq: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    """Encourage the final timestep to place mass on the true target cell."""

    final_attention = attention_seq[:, -1:]
    target_attention = final_attention.gather(
        2,
        target_pos[:, None, None].expand(-1, 1, 1),
    ).squeeze(-1)
    return -target_attention.clamp_min(1e-8).log().mean()


def evaluate_model(
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    *,
    num_batches: int,
    seed: int,
    ablation: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Evaluate one model over freshly generated held-out batches."""

    model.eval()
    generator = make_generator(seed, device)
    total_loss = 0.0
    total_accuracy = 0.0
    total_target_attention = 0.0
    total_attention_entropy = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            batch = generate_batch(
                cfg["training"]["batch_size"],
                task_cfg.num_steps,
                task_cfg,
                generator=generator,
                device=device,
            )
            outputs = model(
                batch.scene,
                batch.cue,
                target=batch.target,
                num_steps=task_cfg.num_steps,
                ablation=ablation,
            )
            loss = F.cross_entropy(outputs["logits"], batch.target)
            metrics = compute_batch_metrics(outputs, batch)
            total_loss += loss.item()
            total_accuracy += metrics["accuracy"]
            total_target_attention += metrics["target_attention"]
            total_attention_entropy += metrics["attention_entropy"]

    scale = 1.0 / num_batches
    return {
        "loss": total_loss * scale,
        "accuracy": total_accuracy * scale,
        "target_attention": total_target_attention * scale,
        "attention_entropy": total_attention_entropy * scale,
    }


def train_single_model(
    name: str,
    model,
    cfg: dict[str, Any],
    task_cfg: TaskConfig,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Train one model variant and keep the best validation checkpoint in memory."""

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    generator = make_generator(cfg["seed"] + (0 if name == "static" else 1000), device)
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, Any] | None = None

    for step in range(1, train_cfg["train_steps"] + 1):
        model.train()
        batch = generate_batch(
            train_cfg["batch_size"],
            task_cfg.num_steps,
            task_cfg,
            generator=generator,
            device=device,
        )
        outputs = model(batch.scene, batch.cue, target=batch.target, num_steps=task_cfg.num_steps)
        final_loss = F.cross_entropy(outputs["logits"], batch.target)
        aux_loss = F.cross_entropy(
            outputs["logits_seq"][:, :-1].reshape(-1, task_cfg.digit_vocab_size),
            batch.target.repeat_interleave(task_cfg.num_steps - 1),
        )
        attention = outputs["attention_seq"]
        attention_target_loss = compute_attention_target_loss(attention, batch.target_pos)
        attention_entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1).mean()
        # The main task loss is paired with a small final-fixation objective so the
        # controller gets a direct signal about where useful evidence should end up.
        loss = (
            final_loss
            + train_cfg["aux_loss_weight"] * aux_loss
            + train_cfg["attention_target_weight"] * attention_target_loss
            + train_cfg["attention_entropy_weight"] * attention_entropy
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % train_cfg["log_interval"] == 0 or step == 1:
            metrics = compute_batch_metrics(outputs, batch)
            print(
                f"[{name}] step={step} loss={loss.item():.4f} "
                f"acc={metrics['accuracy']:.3f} target_attn={metrics['target_attention']:.3f} "
                f"entropy={metrics['attention_entropy']:.3f}"
            )

        if step % train_cfg["val_interval"] == 0 or step == train_cfg["train_steps"]:
            metrics = evaluate_model(
                model,
                cfg,
                task_cfg,
                device,
                num_batches=train_cfg["val_batches"],
                seed=cfg["seed"] + step + (0 if name == "static" else 5000),
            )
            print(
                f"[{name}] val step={step} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
                f"target_attn={metrics['target_attention']:.3f}"
            )
            if best_metrics is None or metrics["accuracy"] >= best_metrics["accuracy"]:
                best_metrics = metrics
                best_state = {
                    "step": step,
                    "model_state_dict": deepcopy(model.state_dict()),
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                }

    assert best_metrics is not None and best_state is not None
    torch.save(best_state, output_dir / f"{name}_best.pt")
    return best_metrics, best_state


def train_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train both baseline and recurrent models and save a joint experiment checkpoint."""

    cfg = deep_update(DEFAULT_CONFIG, config or {})
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = TaskConfig.from_dict(cfg["task"])
    model_cfg = ModelConfig.from_dict(cfg["model"])
    models = {
        "static": StaticAttentionBaseline(task_cfg, model_cfg).to(device),
        "recurrent": RecurrentAttentionController(task_cfg, model_cfg).to(device),
    }

    metrics = {}
    states = {}
    for name, model in models.items():
        model_metrics, model_state = train_single_model(name, model, cfg, task_cfg, device, output_dir)
        metrics[name] = model_metrics
        states[name] = model_state["model_state_dict"]

    checkpoint_path = output_dir / "experiment.pt"
    torch.save(
        {
            "config": cfg,
            "task": task_cfg.to_dict(),
            "models": states,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    return {"checkpoint_path": str(checkpoint_path), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the minimal attention-control experiment.")
    parser.add_argument("--config", type=str, default="configs/minimal.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    train_experiment(config)


if __name__ == "__main__":
    main()
