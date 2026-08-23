"""External-VLM Stage 7 audit over visual internal-state renderings.

The same vision-capable model sees three aligned conditions: a label-free controller-state
heatmap, an observation-only panel, and an explicit symbolic-state panel. The symbolic panel
is an upper-bound/OCR control; support requires a valid upper bound plus a significant paired
heatmap advantage over the observation-only condition.

Usage:
    .venv/bin/python scripts/stage7_external_vlm_audit.py
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
    load_models_from_checkpoint,
    make_generator,
)
from attcon.nl_report import load_dotenv, run_nl_report_mode
from attcon.train import load_config
from attcon.vlm_report import VLMImageRenderer
try:
    from scripts.stage7_external_llm_audit import (
        _paired_llm_comparison,
        _terminal_api_blocker,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from stage7_external_llm_audit import (
        _paired_llm_comparison,
        _terminal_api_blocker,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an external-VLM Stage 7 audit.")
    parser.add_argument("--config", default="configs/stage7_content_memory_v3.yaml")
    parser.add_argument("--checkpoint", default="outputs/stage7_content_memory_v3/experiment.pt")
    parser.add_argument("--out", default="audits/stage7_external_vlm_smoke_content_memory_v3.json")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--probe-scenes", type=int, default=16)
    parser.add_argument("--calibration-examples", type=int, default=4)
    parser.add_argument("--evaluation-examples", type=int, default=2)
    parser.add_argument("--translator-train-examples", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=9911)
    parser.add_argument("--state-key", default="content_memory_state_seq")
    parser.add_argument("--min-symbolic-control-accuracy", type=float, default=0.75)
    parser.add_argument(
        "--slices",
        nargs="+",
        default=["default"],
        choices=["default", "cue_switch", "intervention_baseline", "intervention_intervened"],
    )
    return parser.parse_args()


def _summary(scored: dict[str, Any]) -> dict[str, float]:
    return {
        "current_content_joint_accuracy": scored["current_content_joint_accuracy"],
        "memory_content_joint_accuracy": scored["memory_content_joint_accuracy"],
        "content_only_joint_accuracy": scored["content_only_joint_accuracy"],
        "joint_accuracy": scored["joint_accuracy"],
    }


def _write(path: Path, payload: dict[str, Any], *, verbose: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    if verbose:
        print(json.dumps(payload, indent=2))
        print(f"\nwrote {path}")


def _score_condition(
    *,
    args: argparse.Namespace,
    mode: str,
    calibration: list[Any],
    evaluation: list[Any],
    teaching: list[Any],
    grid_size: int,
) -> dict[str, Any]:
    renderer = VLMImageRenderer(mode)
    return run_nl_report_mode(
        mode=mode,
        model_name=args.model,
        calibration_examples=calibration,
        evaluation_examples=evaluation,
        grid_size=grid_size,
        max_output_tokens=args.max_output_tokens,
        request_retries=args.request_retries,
        teaching_examples=teaching,
        input_content_builder=renderer.content,
        input_summary_builder=renderer.summary,
    )


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
    teaching = _select_translator_examples(
        [example for example in examples if id(example) not in held_out],
        grid_size=grid_size,
        target_count=args.translator_train_examples,
    )
    if not teaching:
        teaching = calibration

    conditions = {
        mode: _score_condition(
            args=args,
            mode=mode,
            calibration=calibration,
            evaluation=evaluation,
            teaching=teaching,
            grid_size=grid_size,
        )
        for mode in (
            "visual_latent_state",
            "visual_observation_only",
            "visual_symbolic_state",
        )
    }
    latent = conditions["visual_latent_state"]
    observation = conditions["visual_observation_only"]
    symbolic = conditions["visual_symbolic_state"]
    symbolic_summary = _summary(symbolic)
    control_valid = all(
        symbolic_summary[key] >= args.min_symbolic_control_accuracy
        for key in (
            "current_content_joint_accuracy",
            "memory_content_joint_accuracy",
            "content_only_joint_accuracy",
        )
    )
    paired = _paired_llm_comparison(latent, observation)
    return {
        "status": "complete",
        "calibration_examples": len(calibration),
        "evaluation_examples": len(evaluation),
        "translator_train_examples": len(teaching),
        "condition_summaries": {
            name: _summary(scored) for name, scored in conditions.items()
        },
        "symbolic_upper_bound": {
            "minimum_required_accuracy": args.min_symbolic_control_accuracy,
            "valid": control_valid,
        },
        "latent_vs_observation_vlm": paired,
        "content_supported": bool(control_valid and paired["content_supported"]),
        "conditions": conditions,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    load_dotenv(".env")
    if not os.environ.get("OPENAI_API_KEY"):
        _write(out_path, {"status": "blocked", "reason": "OPENAI_API_KEY is not set"})
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
        "default": collect_nl_examples(
            model,
            task_cfg,
            batch,
            outputs,
            state_key=args.state_key,
        )
    }
    if "cue_switch" in args.slices:
        slice_examples["cue_switch"] = collect_cue_switch_nl_examples(
            model,
            task_cfg,
            batch,
            switch_step=int(cfg["evaluation"].get("cue_switch", {}).get("switch_step", 3)),
            state_key=args.state_key,
        )
    if any(name.startswith("intervention_") for name in args.slices):
        intervention = collect_intervention_nl_examples(
            model,
            task_cfg,
            batch,
            intervention_step=int(cfg["evaluation"].get("intervention_test", {}).get("step", 5)),
            state_key=args.state_key,
        )
        slice_examples["intervention_baseline"] = intervention["baseline_examples"]
        slice_examples["intervention_intervened"] = intervention["intervened_examples"]

    payload: dict[str, Any] = {
        "status": "running",
        "note": (
            "Powered paired Stage 7 VLM audit."
            if args.evaluation_examples >= 8
            else "Small live Stage 7 VLM route/control audit; not powered for support."
        ),
        "config_base": args.config,
        "checkpoint": args.checkpoint,
        "model": args.model,
        "probe_scenes": args.probe_scenes,
        "calibration_examples": args.calibration_examples,
        "evaluation_examples": args.evaluation_examples,
        "translator_train_examples": args.translator_train_examples,
        "state_key": args.state_key,
        "slices_requested": args.slices,
        "estimated_api_requests": len(args.slices) * args.evaluation_examples * 3,
        "slices": {},
    }
    _write(out_path, payload, verbose=False)
    terminal_blocker = None
    for slice_index, name in enumerate(args.slices):
        try:
            payload["slices"][name] = _score_slice(
                args=args,
                examples=slice_examples[name],
                grid_size=task_cfg.grid_size,
            )
        except Exception as exc:
            payload["slices"][name] = {
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        _write(out_path, payload, verbose=False)
        print(f"completed slice {name}: {payload['slices'][name]['status']}", flush=True)
        reason = payload["slices"][name].get("reason", "")
        if payload["slices"][name]["status"] == "blocked" and _terminal_api_blocker(reason):
            terminal_blocker = reason
            for remaining in args.slices[slice_index + 1 :]:
                payload["slices"][remaining] = {
                    "status": "not_attempted",
                    "reason": "terminal API account blocker encountered on an earlier slice",
                }
            break

    completed = sum(item.get("status") == "complete" for item in payload["slices"].values())
    payload["completed_slices"] = completed
    payload["terminal_api_blocker"] = terminal_blocker
    payload["status"] = "complete" if completed == len(args.slices) else "partial"
    payload["content_supported"] = bool(
        completed == len(args.slices)
        and all(item.get("content_supported", False) for item in payload["slices"].values())
    )
    _write(out_path, payload)


if __name__ == "__main__":
    main()
