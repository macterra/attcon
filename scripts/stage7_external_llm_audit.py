"""External-LLM Stage 7 audit for the latent-only reporter route.

It compares an external LLM on the stricter latent-only interface against the same model on
observation-only prompts for aligned held-out examples. The powered route uses an exact paired
sign test over per-example joint-content correctness. If the API/model/quota is unavailable,
the script writes a blocked or partial audit artifact instead of failing silently.

Usage:
    .venv/bin/python scripts/stage7_external_llm_audit.py
"""
from __future__ import annotations

import argparse
import json
import math
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
    parser = argparse.ArgumentParser(description="Run an external-LLM Stage 7 audit.")
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
    parser.add_argument("--state-key", default="controller_state_seq")
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


def _write(path: Path, payload: dict[str, Any], *, verbose: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    if verbose:
        print(json.dumps(payload, indent=2))
        print(f"\nwrote {path}")


def _joint_correctness(scored: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """Recover the scorer's per-example Stage 7 joint-content decisions."""

    correctness: dict[str, dict[str, bool]] = {}
    for item in scored.get("examples", []):
        response = item["response"]
        expected = item["expected"]
        current = bool(
            response["attended_visible_type"] == expected["attended_visible_type"]
            and response["attended_digit"] == expected["attended_digit"]
            and response["glimpse_digit"] == expected["glimpse_digit"]
            and bool(response["glimpse_target_match"]) == expected["glimpse_target_match"]
        )
        memory = bool(
            response["previous_attended_visible_type"]
            == expected["previous_attended_visible_type"]
            and response["previous_attended_digit"] == expected["previous_attended_digit"]
            and response["previous_glimpse_digit"] == expected["previous_glimpse_digit"]
        )
        content_only = bool(
            response["previous_search_type"] == expected["previous_search_type"]
            and bool(response["cue_switched"]) == expected["cue_switched"]
            and bool(response["previous_found_target"]) == expected["previous_found_target"]
            and response["inspected_count"] == expected["inspected_count"]
            and response["previous_inspected_count"] == expected["previous_inspected_count"]
            and bool(response["attended_cell_previously_inspected"])
            == expected["attended_cell_previously_inspected"]
            and response["attended_visible_type"] == expected["attended_visible_type"]
            and response["attended_digit"] == expected["attended_digit"]
            and response["glimpse_digit"] == expected["glimpse_digit"]
            and response["previous_attended_visible_type"]
            == expected["previous_attended_visible_type"]
            and response["previous_attended_digit"] == expected["previous_attended_digit"]
            and response["previous_glimpse_digit"] == expected["previous_glimpse_digit"]
            and bool(response["glimpse_target_match"]) == expected["glimpse_target_match"]
            and bool(response["found_target"]) == expected["found_target"]
            and bool(response["relevant_region_inspected"])
            == expected["relevant_region_inspected"]
            and bool(response["unresolved_search"]) == expected["unresolved_search"]
            and bool(response["current_wrong_candidate"])
            == expected["current_wrong_candidate"]
            and bool(response["wrong_candidate_history"])
            == expected["wrong_candidate_history"]
            and bool(response["revisit_unresolved"]) == expected["revisit_unresolved"]
            and bool(response["allocation_error"]) == expected["allocation_error"]
        )
        correctness[item["example_id"]] = {
            "current": current,
            "memory": memory,
            "content_only": content_only,
        }
    return correctness


def _one_sided_exact_sign_p_value(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, value) for value in range(wins, discordant + 1)) / (
        2**discordant
    )


def _paired_llm_comparison(
    latent: dict[str, Any],
    observation: dict[str, Any],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    latent_correct = _joint_correctness(latent)
    observation_correct = _joint_correctness(observation)
    if latent_correct.keys() != observation_correct.keys():
        raise ValueError("latent and observation LLM results are not example-aligned")

    metrics: dict[str, Any] = {}
    for name, aggregate_key in (
        ("current", "current_content_joint_accuracy"),
        ("memory", "memory_content_joint_accuracy"),
        ("content_only", "content_only_joint_accuracy"),
    ):
        wins = sum(
            latent_correct[example_id][name] and not observation_correct[example_id][name]
            for example_id in latent_correct
        )
        losses = sum(
            observation_correct[example_id][name] and not latent_correct[example_id][name]
            for example_id in latent_correct
        )
        advantage = float(latent[aggregate_key] - observation[aggregate_key])
        p_value = _one_sided_exact_sign_p_value(wins, losses)
        metrics[name] = {
            "latent_accuracy": latent[aggregate_key],
            "observation_accuracy": observation[aggregate_key],
            "accuracy_advantage": advantage,
            "latent_only_wins": wins,
            "observation_only_wins": losses,
            "ties": len(latent_correct) - wins - losses,
            "one_sided_exact_p_value": p_value,
            "significant_advantage": bool(advantage > 0.0 and p_value < alpha),
        }
    directional = all(metrics[name]["accuracy_advantage"] > 0.0 for name in metrics)
    significant = all(metrics[name]["significant_advantage"] for name in metrics)
    return {
        "evaluation_examples": len(latent_correct),
        "alpha": alpha,
        "metrics": metrics,
        "content_supported_directional": directional,
        "content_supported_paired_significance": significant,
        "content_supported": significant,
    }


def _terminal_api_blocker(reason: str) -> bool:
    lowered = reason.lower()
    return any(
        marker in lowered
        for marker in (
            "insufficient_quota",
            "credit_balance_exhausted",
            "no credits remaining",
            "invalid_api_key",
        )
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
        "latent_vs_observation_llm": _paired_llm_comparison(latent_llm, observation_llm),
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
        "default": collect_nl_examples(
            model,
            task_cfg,
            batch,
            outputs,
            state_key=args.state_key,
        ),
    }
    if any(name == "cue_switch" for name in args.slices):
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
            "Powered live external-LLM Stage 7 audit with an exact paired latent-vs-observation "
            "sign test. A support claim requires all three joint-content families to show a "
            "positive, p<0.05 paired advantage."
            if args.evaluation_examples >= 8
            else "Small live external-LLM Stage 7 plumbing audit; not powered for support."
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
        "estimated_api_requests": len(args.slices) * args.evaluation_examples * 2,
        "latent_interface": {
            "num_chunks": args.latent_num_chunks,
            "num_levels": args.latent_num_levels,
        },
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
            for remaining_name in args.slices[slice_index + 1 :]:
                payload["slices"][remaining_name] = {
                    "status": "not_attempted",
                    "reason": "terminal API account blocker encountered on an earlier slice",
                }
            break

    completed = sum(
        slice_result.get("status") == "complete"
        for slice_result in payload["slices"].values()
    )
    payload["completed_slices"] = completed
    payload["terminal_api_blocker"] = terminal_blocker
    payload["status"] = "complete" if completed == len(args.slices) else "partial"
    payload["content_supported_slices"] = [
        name
        for name, slice_result in payload["slices"].items()
        if slice_result.get("latent_vs_observation_llm", {}).get("content_supported", False)
    ]
    payload["content_supported"] = bool(
        completed == len(args.slices)
        and payload["content_supported_slices"]
        and len(payload["content_supported_slices"]) == len(args.slices)
    )
    _write(out_path, payload)


if __name__ == "__main__":
    main()
