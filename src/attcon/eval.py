from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from .data import TaskConfig, expand_cues_for_probe, generate_batch
from .models import ModelConfig, RecurrentAttentionController, StaticAttentionBaseline
from .train import deep_update, evaluate_model, load_config, make_generator


def load_models_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    config: dict[str, Any] | None = None,
):
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


def trajectory_divergence(
    model,
    task_cfg: TaskConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> float:
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

    attention = outputs["attention_seq"].view(batch_size, task_cfg.num_types, task_cfg.num_steps, -1)
    divergences = []
    for cue_a in range(task_cfg.num_types):
        for cue_b in range(cue_a + 1, task_cfg.num_types):
            divergences.append(
                symmetric_kl(attention[:, cue_a], attention[:, cue_b]).mean().item()
            )
    return sum(divergences) / max(len(divergences), 1)


def save_probe_plots(
    model,
    output_dir: Path,
    task_cfg: TaskConfig,
    device: torch.device,
    seed: int,
) -> list[str]:
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
        fig, axes = plt.subplots(1, task_cfg.num_steps, figsize=(3 * task_cfg.num_steps, 3))
        if task_cfg.num_steps == 1:
            axes = [axes]
        for step_idx in range(task_cfg.num_steps):
            ax = axes[step_idx]
            heatmap = attention[cue_idx, step_idx].reshape(task_cfg.grid_size, task_cfg.grid_size).cpu()
            ax.imshow(heatmap, cmap="viridis")
            ax.set_title(f"Cue {cue_idx} / Step {step_idx + 1}")
            ax.axis("off")
        path = output_dir / f"attention_probe_cue_{cue_idx}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        saved_paths.append(str(path))

    return saved_paths


def run_ablations(config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, Any]:
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

    report["baseline"]["trajectory_divergence"] = trajectory_divergence(
        models["static"],
        task_cfg,
        eval_cfg["probe_scenes"],
        device,
        cfg["seed"] + 9200,
    )
    report["recurrent"]["trajectory_divergence"] = trajectory_divergence(
        models["recurrent"],
        task_cfg,
        eval_cfg["probe_scenes"],
        device,
        cfg["seed"] + 9300,
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
        report["ablations"][name]["trajectory_divergence"] = trajectory_divergence(
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

    plot_paths = save_probe_plots(
        models["recurrent"],
        output_dir / "plots",
        task_cfg,
        device,
        cfg["seed"] + 9600,
    )
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
