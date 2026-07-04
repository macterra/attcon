"""Tiny external-LLM Stage 7 audit for the latent-only reporter route.

This is intentionally small because it makes live API calls. It compares an external LLM
on the stricter latent-only interface against observation-only on the same held-out examples.
If the API/model/quota is unavailable, the script writes a blocked audit artifact instead of
failing silently.

Usage:
    .venv/bin/python scripts/stage7_external_llm_audit.py
"""
from __future__ import annotations

import argparse
import json
import os
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
    load_dotenv,
    run_nl_report_mode,
    run_observation_only_heuristic_report_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny external-LLM Stage 7 audit.")
    parser.add_argument("--config", default="configs/tune_prob_035.yaml")
    parser.add_argument("--checkpoint", default="outputs/tune_prob_035/experiment.pt")
    parser.add_argument("--out", default="audits/stage7_external_llm_tiny_tune_prob_035.json")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--probe-scenes", type=int, default=8)
    parser.add_argument("--calibration-examples", type=int, default=4)
    parser.add_argument("--evaluation-examples", type=int, default=2)
    parser.add_argument("--translator-train-examples", type=int, default=8)
    parser.add_argument("--latent-num-chunks", type=int, default=16)
    parser.add_argument("--latent-num-levels", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=9901)
    parser.add_argument(
        "--slices",
        nargs="+",
        default=["default"],
        choices=["default", "cue_switch", "intervention_baseline", "intervention_intervened"],
    )
    return parser.parse_args()


def _summarize_against_observation(scored: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_content_joint_accuracy": scored["current_content_joint_accuracy"],
        "memory_content_joint_accuracy": scored["memory_content_joint_accuracy"],
        "content_only_joint_accuracy": scored["content_only_joint_accuracy"],
        "joint_accuracy": scored["joint_accuracy"],
        "current_content_joint_accuracy_advantage": (
            scored["current_content_joint_accuracy"] - observation["current_content_joint_accuracy"]
        ),
        "memory_content_joint_accuracy_advantage": (
            scored["memory_content_joint_accuracy"] - observation["memory_content_joint_accuracy"]
        ),
        "content_only_joint_accuracy_advantage": (
            scored["content_only_joint_accuracy"] - observation["content_only_joint_accuracy"]
        ),
        "content_supported": (
            scored["current_content_joint_accuracy"] > observation["current_content_joint_accuracy"]
            and scored["memory_content_joint_accuracy"] > observation["memory_content_joint_accuracy"]
            and scored["content_only_joint_accuracy"] > observation["content_only_joint_accuracy"]
        ),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")


def _score_slice(
    *,
    args: argparse.Namespace,
    examples: list[Any],
    grid_size: int,
) -> dict[str, Any]:
    examples = [example for example in examples if example.step_index > 0]
    required = args.calibration_examples + args.evaluation_examples
    if len(examples) < required:
        return {
            "status": "blocked",
            "reason": f"not enough examples: need {required}, have {len(examples)}",
        }

    calibration, evaluation = _select_diverse_nl_examples(
        examples,
        grid_size=grid_size,
        calibration_count=args.calibration_examples,
        evaluation_count=args.evaluation_examples,
    )
    held_out = {id(example) for example in calibration + evaluation}
    translator_pool = [example for example in examples if id(example) not in held_out]
    teaching = _select_translator_examples(
        translator_pool,
        grid_size=grid_size,
        target_count=args.translator_train_examples,
    )
    if not teaching:
        teaching = calibration

    observation = run_observation_only_heuristic_report_mode(
        evaluation_examples=evaluation,
        grid_size=grid_size,
    )
    latent_llm = run_nl_report_mode(
        mode="latent_only_state",
        model_name=args.model,
        calibration_examples=calibration,
        evaluation_examples=evaluation,
        grid_size=grid_size,
        max_output_tokens=args.max_output_tokens,
        request_retries=args.request_retries,
        teaching_examples=teaching,
        latent_num_chunks=args.latent_num_chunks,
        latent_num_levels=args.latent_num_levels,
    )
    observation_llm = run_nl_report_mode(
        mode="observation_only",
        model_name=args.model,
        calibration_examples=calibration,
        evaluation_examples=evaluation,
        grid_size=grid_size,
        max_output_tokens=args.max_output_tokens,
        request_retries=args.request_retries,
        teaching_examples=teaching,
    )
    return {
        "status": "complete",
        "calibration_examples": len(calibration),
        "evaluation_examples": len(evaluation),
        "translator_train_examples": len(teaching),
        "local_observation_baseline": {
            "current_content_joint_accuracy": observation["current_content_joint_accuracy"],
            "memory_content_joint_accuracy": observation["memory_content_joint_accuracy"],
            "content_only_joint_accuracy": observation["content_only_joint_accuracy"],
        },
        "latent_only_llm_summary": _summarize_against_observation(latent_llm, observation),
        "observation_only_llm_summary": _summarize_against_observation(observation_llm, observation),
        "latent_only_llm": latent_llm,
        "observation_only_llm": observation_llm,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    load_dotenv(".env")
    if not os.environ.get("OPENAI_API_KEY"):
        _write(
            out_path,
            {
                "status": "blocked",
                "reason": "OPENAI_API_KEY is not set",
                "config_base": args.config,
                "checkpoint": args.checkpoint,
                "model": args.model,
            },
        )
        return

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
    slice_examples = {
        "default": collect_nl_examples(model, task_cfg, batch, outputs),
    }
    if any(name == "cue_switch" for name in args.slices):
        slice_examples["cue_switch"] = collect_cue_switch_nl_examples(
            model,
            task_cfg,
            batch,
            switch_step=int(cfg["evaluation"].get("cue_switch", {}).get("switch_step", 3)),
        )
    if any(name.startswith("intervention_") for name in args.slices):
        intervention = collect_intervention_nl_examples(
            model,
            task_cfg,
            batch,
            intervention_step=int(cfg["evaluation"].get("intervention_test", {}).get("step", 5)),
        )
        slice_examples["intervention_baseline"] = intervention["baseline_examples"]
        slice_examples["intervention_intervened"] = intervention["intervened_examples"]

    try:
        slices = {
            name: _score_slice(
                args=args,
                examples=slice_examples[name],
                grid_size=task_cfg.grid_size,
            )
            for name in args.slices
        }
    except Exception as exc:
        _write(
            out_path,
            {
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
                "config_base": args.config,
                "checkpoint": args.checkpoint,
                "model": args.model,
                "probe_scenes": args.probe_scenes,
                "calibration_examples": args.calibration_examples,
                "evaluation_examples": args.evaluation_examples,
                "translator_train_examples": args.translator_train_examples,
                "slices": args.slices,
                "latent_interface": {
                    "num_chunks": args.latent_num_chunks,
                    "num_levels": args.latent_num_levels,
                },
            },
        )
        return

    _write(
        out_path,
        {
            "status": "complete",
            "note": (
                "Tiny live external-LLM Stage 7 audit. This is a plumbing and smoke result, "
                "not enough sample size for a support claim."
            ),
            "config_base": args.config,
            "checkpoint": args.checkpoint,
            "model": args.model,
            "probe_scenes": args.probe_scenes,
            "calibration_examples": args.calibration_examples,
            "evaluation_examples": args.evaluation_examples,
            "translator_train_examples": args.translator_train_examples,
            "slices_requested": args.slices,
            "latent_interface": {
                "num_chunks": args.latent_num_chunks,
                "num_levels": args.latent_num_levels,
            },
            "slices": slices,
        },
    )


if __name__ == "__main__":
    main()
