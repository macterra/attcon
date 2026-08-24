"""Stage 7 latent-only permuted-label noise floor.

The latent-only decoder originally used a bare directional gate: ``current_adv > 0 and
memory_adv > 0 and content_adv > 0``. On a 24-example evaluation slice, a joint-accuracy
advantage of ``+1/24`` (``0.0417``) was enough to flip it to ``True``. The shared reporter now
uses this permuted-label floor as its primary support gate. This standalone audit remains the
powered runner for published artifacts: it refits the latent decoder many times with fit-time
labels permuted (features held fixed), so the feature-to-label association is destroyed and no
real state-to-content signal can survive. The p95 of the permuted advantage is a data-driven
significance floor; a real advantage that does not clear it is indistinguishable from probe-fit
luck.

Usage:
    .venv/bin/python scripts/stage7_latent_noise_floor.py \
        --config configs/stage7_content_memory_v3.yaml \
        --checkpoint outputs/stage7_content_memory_v3/experiment.pt \
        --state-key content_memory_state_seq \
        --out audits/stage7_latent_noise_floor_content_memory_v3.json
"""
from __future__ import annotations

import argparse
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
    run_latent_content_noise_floor,
    run_observation_only_heuristic_report_mode,
)

INTERFACES = ((8, 4), (16, 8), (32, 8), (48, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permuted-label noise floor for the Stage 7 latent decoder.")
    parser.add_argument("--config", default="configs/stage7_content_memory_v3.yaml")
    parser.add_argument("--checkpoint", default="outputs/stage7_content_memory_v3/experiment.pt")
    parser.add_argument("--state-key", default="content_memory_state_seq")
    parser.add_argument("--out", default="audits/stage7_latent_noise_floor_content_memory_v3.json")
    parser.add_argument("--probe-scenes", type=int, default=32)
    parser.add_argument("--calibration-examples", type=int, default=8)
    parser.add_argument("--evaluation-examples", type=int, default=24)
    parser.add_argument("--translator-train-examples", type=int, default=64)
    parser.add_argument("--seed-offset", type=int, default=8811)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--permutation-seed", type=int, default=20260705)
    parser.add_argument(
        "--interfaces",
        default="all",
        help=(
            "Comma-separated interface widths (e.g. '32x8,48x8'), or 'all'. Only the widths the "
            "pilot reports as directionally positive need a floor; narrower widths are already "
            "negative under the bare >0 gate."
        ),
    )
    return parser.parse_args()


def _selected_interfaces(spec: str) -> tuple[tuple[int, int], ...]:
    if spec.strip().lower() == "all":
        return INTERFACES
    chosen = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        chunks, levels = token.lower().split("x")
        chosen.append((int(chunks), int(levels)))
    return tuple(chosen)


def _build_fit_eval(
    examples: list[Any],
    *,
    grid_size: int,
    calibration_examples: int,
    evaluation_examples: int,
    translator_train_examples: int,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Reproduce the fit/evaluation/observation split used by stage7_latent_followup._score_slice."""
    calibration, evaluation = _select_diverse_nl_examples(
        examples,
        grid_size=grid_size,
        calibration_count=calibration_examples,
        evaluation_count=evaluation_examples,
    )
    held_out = {id(example) for example in calibration + evaluation}
    translator_pool = [example for example in examples if id(example) not in held_out]
    translator = _select_translator_examples(
        translator_pool,
        grid_size=grid_size,
        target_count=translator_train_examples,
    )
    if not translator:
        translator = calibration
    fit = translator + calibration
    observation = run_observation_only_heuristic_report_mode(
        evaluation_examples=evaluation,
        grid_size=grid_size,
    )
    return fit, evaluation, observation


def _score_interface(
    *,
    fit: list[Any],
    evaluation: list[Any],
    observation: dict[str, Any],
    grid_size: int,
    num_chunks: int,
    num_levels: int,
    permutations: int,
    permutation_seed: int,
    init_seed: int,
) -> dict[str, Any]:
    return run_latent_content_noise_floor(
        fit_examples=fit,
        evaluation_examples=evaluation,
        observation_metrics=observation,
        grid_size=grid_size,
        num_chunks=num_chunks,
        num_levels=num_levels,
        permutations=permutations,
        percentile=95.0,
        permutation_seed=permutation_seed,
        probe_init_seed=init_seed,
    )["noise_floor"]


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    cfg = load_config(args.config)
    _, task_cfg, models = load_models_from_checkpoint(args.checkpoint, device, cfg)
    model = models["recurrent"]
    model.eval()

    generator = make_generator(int(cfg["seed"]) + args.seed_offset, device)
    batch = generate_batch(
        args.probe_scenes,
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

    default_examples = collect_nl_examples(model, task_cfg, batch, outputs, state_key=args.state_key)
    cue_switch_examples = collect_cue_switch_nl_examples(
        model,
        task_cfg,
        batch,
        switch_step=int(cfg["evaluation"]["cue_switch"].get("switch_step", 3)),
        state_key=args.state_key,
    )
    intervention = collect_intervention_nl_examples(
        model,
        task_cfg,
        batch,
        intervention_step=int(cfg["evaluation"]["intervention_test"].get("step", 5)),
        state_key=args.state_key,
    )

    slice_examples = {
        "default": default_examples,
        "cue_switch": cue_switch_examples,
        "intervention_baseline": intervention["baseline_examples"],
        "intervention_intervened": intervention["intervened_examples"],
    }

    slices: dict[str, Any] = {}
    for slice_index, (slice_name, examples) in enumerate(slice_examples.items()):
        fit, evaluation, observation = _build_fit_eval(
            examples,
            grid_size=task_cfg.grid_size,
            calibration_examples=args.calibration_examples,
            evaluation_examples=args.evaluation_examples,
            translator_train_examples=args.translator_train_examples,
        )
        interfaces: dict[str, Any] = {}
        for interface_index, (num_chunks, num_levels) in enumerate(
            _selected_interfaces(args.interfaces)
        ):
            # Give each cell a deterministic independent permutation stream.
            cell_seed = args.permutation_seed + slice_index * 1000 + interface_index
            interfaces[f"{num_chunks}x{num_levels}"] = _score_interface(
                fit=fit,
                evaluation=evaluation,
                observation=observation,
                grid_size=task_cfg.grid_size,
                num_chunks=num_chunks,
                num_levels=num_levels,
                permutations=args.permutations,
                permutation_seed=cell_seed,
                init_seed=args.permutation_seed,
            )
        slices[slice_name] = {
            "fit_examples": len(fit),
            "evaluation_examples": len(evaluation),
            "quantized_opaque_levels": interfaces,
        }

    result = {
        "note": (
            "Permuted-label noise floor for the Stage 7 latent-only decoder. For each slice and "
            "interface width, the decoder is refit `permutations` times with fit-time labels "
            "shuffled (features fixed), giving a null distribution of the content advantage. A "
            "content claim is credible only if the observed advantage clears the permuted-label "
            "p95 floor. `content_supported` and `content_supported_vs_floor` are the primary "
            "significance-aware gates; `content_supported_directional` is retained only for "
            "comparison with older pilots."
        ),
        "config_base": args.config,
        "checkpoint": args.checkpoint,
        "state_key": args.state_key,
        "permutations": args.permutations,
        "permutation_seed": args.permutation_seed,
        "evaluation_examples": args.evaluation_examples,
        "slices": slices,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
