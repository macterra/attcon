from __future__ import annotations

"""Natural-language reportability helpers for Stage 7 experiments."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency missing at runtime.
    OpenAI = None


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class NLExample:
    """One step-level reportability example."""

    example_id: str
    step_index: int
    cue: int
    attended_cell: int
    attended_visible_type: int
    attended_digit: int
    glimpse_digit: int
    glimpse_target_match: bool
    found_target: bool
    unresolved_cells: list[int]
    symbolic_state: str
    tokenized_state: str
    observation_only: str


def _cell_name(cell_idx: int, grid_size: int) -> str:
    row, col = divmod(cell_idx, grid_size)
    return f"r{row}c{col}"


def _render_symbolic_state(example: NLExample, grid_size: int) -> str:
    unresolved = ", ".join(_cell_name(cell, grid_size) for cell in example.unresolved_cells) or "none"
    attended = _cell_name(example.attended_cell, grid_size)
    return (
        f"search_type={example.cue}\n"
        f"attended_cell={attended}\n"
        f"attended_visible_type={example.attended_visible_type}\n"
        f"attended_digit={example.attended_digit}\n"
        f"glimpse_digit={example.glimpse_digit}\n"
        f"glimpse_target_match={str(example.glimpse_target_match).lower()}\n"
        f"found_target={str(example.found_target).lower()}\n"
        f"unresolved_cells={unresolved}"
    )


def _render_tokenized_state(
    grid_size: int,
    cue: int,
    attended_cell: int,
    attended_visible_type: int,
    attended_digit: int,
    glimpse_digit: int,
    glimpse_target_match: bool,
    found_target: bool,
    unresolved_cells: list[int],
) -> str:
    attended_row, attended_col = divmod(attended_cell, grid_size)
    unresolved_set = set(unresolved_cells)
    tokens = [
        "x900",
        f"x{100 + cue}",
        "x901",
        f"x{200 + attended_row}",
        f"x{210 + attended_col}",
        f"x{220 + attended_visible_type}",
        f"x{230 + attended_digit}",
        f"x{240 + glimpse_digit}",
        f"x{250 + int(glimpse_target_match)}",
        f"x{260 + int(found_target)}",
        "x902",
    ]
    for cell in range(grid_size * grid_size):
        state_bit = 0 if cell in unresolved_set else 1
        tokens.append(f"x{1000 + cell * 2 + state_bit}")
    return " ".join(tokens)


def _render_observation_only(
    visible_types: torch.Tensor,
    observation: torch.Tensor,
    cue: int,
    attended_cell: int,
    attended_visible_type: int,
    grid_size: int,
) -> str:
    visible_rows = []
    for row_idx in range(grid_size):
        row = visible_types[row_idx * grid_size : (row_idx + 1) * grid_size].tolist()
        visible_rows.append(" ".join(str(int(value)) for value in row))

    target_match = float(observation[0].item())
    digit_idx = int(observation[1:].argmax().item())
    attended = _cell_name(attended_cell, grid_size)
    return (
        f"cue={cue}\n"
        f"visible_grid=\n" + "\n".join(visible_rows) + "\n"
        f"attended_cell={attended}\n"
        f"attended_visible_type={attended_visible_type}\n"
        f"current_glimpse_target_match={target_match:.3f}\n"
        f"current_glimpse_digit_argmax={digit_idx}"
    )


def collect_nl_examples(
    model,
    task_cfg,
    batch,
    outputs: dict[str, torch.Tensor],
) -> list[NLExample]:
    """Extract per-step examples from one batch of recurrent outputs."""

    attention = outputs["attention_seq"]
    found_state = outputs["found_state_seq"][..., 0]
    inspection = outputs["inspection_seq"]
    observation = outputs["observation_seq"]
    visible_types = batch.visible_types

    examples: list[NLExample] = []
    for batch_idx in range(batch.scene.shape[0]):
        for step_idx in range(task_cfg.num_steps):
            attended_cell = int(attention[batch_idx, step_idx].argmax().item())
            attended_visible_type = int(visible_types[batch_idx, attended_cell].item())
            attended_digit = int(batch.digits[batch_idx, attended_cell].item())
            glimpse_digit = int(observation[batch_idx, step_idx, 1:].argmax().item())
            glimpse_target_match = bool(observation[batch_idx, step_idx, 0].item() >= 0.5)
            unresolved = torch.nonzero(
                inspection[batch_idx, step_idx] < 0.5, as_tuple=False
            ).flatten().tolist()
            example = NLExample(
                example_id=f"scene{batch_idx}_step{step_idx}",
                step_index=step_idx,
                cue=int(batch.cue[batch_idx].item()),
                attended_cell=attended_cell,
                attended_visible_type=attended_visible_type,
                attended_digit=attended_digit,
                glimpse_digit=glimpse_digit,
                glimpse_target_match=glimpse_target_match,
                found_target=bool(found_state[batch_idx, step_idx].item() >= 0.5),
                unresolved_cells=[int(cell) for cell in unresolved],
                symbolic_state="",
                tokenized_state="",
                observation_only="",
            )
            example.symbolic_state = _render_symbolic_state(example, task_cfg.grid_size)
            example.tokenized_state = _render_tokenized_state(
                task_cfg.grid_size,
                example.cue,
                example.attended_cell,
                example.attended_visible_type,
                example.attended_digit,
                example.glimpse_digit,
                example.glimpse_target_match,
                example.found_target,
                example.unresolved_cells,
            )
            example.observation_only = _render_observation_only(
                visible_types[batch_idx],
                observation[batch_idx, step_idx],
                example.cue,
                example.attended_cell,
                example.attended_visible_type,
                task_cfg.grid_size,
            )
            examples.append(example)
    return examples


def _schema() -> dict[str, Any]:
    return {
        "name": "nl_report",
        "schema": {
            "type": "object",
            "properties": {
                "natural_language_report": {"type": "string"},
                "search_type": {"type": "integer"},
                "attended_cell": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "attended_visible_type": {"type": "integer"},
                "attended_digit": {"type": "integer"},
                "glimpse_digit": {"type": "integer"},
                "glimpse_target_match": {"type": "boolean"},
                "found_target": {"type": "boolean"},
                "unresolved_cells": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            },
            "required": [
                "natural_language_report",
                "search_type",
                "attended_cell",
                "attended_visible_type",
                "attended_digit",
                "glimpse_digit",
                "glimpse_target_match",
                "found_target",
                "unresolved_cells",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _make_messages(
    mode: str,
    calibration_examples: list[NLExample],
    eval_example: NLExample,
    grid_size: int,
) -> list[dict[str, Any]]:
    mode_instructions = {
        "tokenized_state": (
            "You receive opaque state tokens. Infer their meanings from the examples, "
            "then answer using the same internal-state semantics."
        ),
        "symbolic_state": (
            "You receive a direct symbolic dump of internal state. Answer faithfully."
        ),
        "observation_only": (
            "You receive only cue, visible grid, and current glimpse information. "
            "Answer as best you can from observation alone."
        ),
    }
    state_attr = {
        "tokenized_state": "tokenized_state",
        "symbolic_state": "symbolic_state",
        "observation_only": "observation_only",
    }[mode]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are reporting a controller's internal state on a "
                        f"{grid_size}x{grid_size} grid. "
                        + mode_instructions[mode]
                    ),
                }
            ],
        }
    ]
    for example in calibration_examples:
        payload = getattr(example, state_attr)
        answer = {
            "natural_language_report": (
                f"I am searching for type {example.cue}, attending {_cell_name(example.attended_cell, grid_size)}, "
                f"seeing visible type {example.attended_visible_type} and digit {example.attended_digit}, "
                f"with glimpse digit {example.glimpse_digit} and glimpse_target_match "
                f"{str(example.glimpse_target_match).lower()}, "
                f"found_target is {str(example.found_target).lower()}, and unresolved cells are "
                + ", ".join(_cell_name(cell, grid_size) for cell in example.unresolved_cells)
            ),
            "search_type": example.cue,
            "attended_cell": list(divmod(example.attended_cell, grid_size)),
            "attended_visible_type": example.attended_visible_type,
            "attended_digit": example.attended_digit,
            "glimpse_digit": example.glimpse_digit,
            "glimpse_target_match": example.glimpse_target_match,
            "found_target": example.found_target,
            "unresolved_cells": [list(divmod(cell, grid_size)) for cell in example.unresolved_cells],
        }
        messages.extend(
            [
                {"role": "user", "content": [{"type": "input_text", "text": payload}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(answer)}]},
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": getattr(eval_example, state_attr)}],
        }
    )
    return messages


def run_nl_report_mode(
    *,
    mode: str,
    model_name: str,
    calibration_examples: list[NLExample],
    evaluation_examples: list[NLExample],
    grid_size: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Query an OpenAI model for one reporting mode and score structured faithfulness."""

    if OpenAI is None:
        raise RuntimeError("openai dependency is not installed")

    client = OpenAI(max_retries=0, timeout=20.0)
    results = []
    exact_search = 0
    exact_attended = 0
    exact_visible_type = 0
    exact_attended_digit = 0
    exact_glimpse_digit = 0
    exact_glimpse_match = 0
    exact_found = 0
    exact_unresolved = 0

    for example in evaluation_examples:
        parsed = None
        last_error = None
        for _ in range(3):
            try:
                response = client.responses.create(
                    model=model_name,
                    input=_make_messages(mode, calibration_examples, example, grid_size),
                    max_output_tokens=max_output_tokens,
                    reasoning={"effort": "low"},
                    text={
                        "verbosity": "low",
                        "format": {"type": "json_schema", **_schema()},
                    },
                )
                output_text = getattr(response, "output_text", "") or ""
                if not output_text:
                    for item in getattr(response, "output", []) or []:
                        for content in getattr(item, "content", None) or []:
                            text_value = getattr(content, "text", None)
                            if text_value:
                                output_text = text_value
                                break
                        if output_text:
                            break
                if not output_text:
                    raise RuntimeError(
                        f"empty structured response with status={getattr(response, 'status', None)}"
                    )
                parsed = json.loads(output_text)
                break
            except (json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
        if parsed is None:
            raise RuntimeError(str(last_error) if last_error is not None else "nl_report parsing failed")
        target_unresolved = [list(divmod(cell, grid_size)) for cell in example.unresolved_cells]
        expected_attended = list(divmod(example.attended_cell, grid_size))

        exact_search += int(parsed["search_type"] == example.cue)
        exact_attended += int(parsed["attended_cell"] == expected_attended)
        exact_visible_type += int(parsed["attended_visible_type"] == example.attended_visible_type)
        exact_attended_digit += int(parsed["attended_digit"] == example.attended_digit)
        exact_glimpse_digit += int(parsed["glimpse_digit"] == example.glimpse_digit)
        exact_glimpse_match += int(bool(parsed["glimpse_target_match"]) == example.glimpse_target_match)
        exact_found += int(bool(parsed["found_target"]) == example.found_target)
        exact_unresolved += int(parsed["unresolved_cells"] == target_unresolved)

        results.append(
            {
                "example_id": example.example_id,
                "mode": mode,
                "input": getattr(example, mode),
                "response": parsed,
                "expected": {
                    "search_type": example.cue,
                    "attended_cell": expected_attended,
                    "attended_visible_type": example.attended_visible_type,
                    "attended_digit": example.attended_digit,
                    "glimpse_digit": example.glimpse_digit,
                    "glimpse_target_match": example.glimpse_target_match,
                    "found_target": example.found_target,
                    "unresolved_cells": target_unresolved,
                },
            }
        )

    denom = max(len(evaluation_examples), 1)
    return {
        "mode": mode,
        "search_type_accuracy": exact_search / denom,
        "attended_cell_accuracy": exact_attended / denom,
        "attended_visible_type_accuracy": exact_visible_type / denom,
        "attended_digit_accuracy": exact_attended_digit / denom,
        "glimpse_digit_accuracy": exact_glimpse_digit / denom,
        "glimpse_target_match_accuracy": exact_glimpse_match / denom,
        "found_target_accuracy": exact_found / denom,
        "unresolved_cells_accuracy": exact_unresolved / denom,
        "joint_accuracy": (
            sum(
                int(
                    item["response"]["search_type"] == item["expected"]["search_type"]
                    and item["response"]["attended_cell"] == item["expected"]["attended_cell"]
                    and item["response"]["attended_visible_type"] == item["expected"]["attended_visible_type"]
                    and item["response"]["attended_digit"] == item["expected"]["attended_digit"]
                    and item["response"]["glimpse_digit"] == item["expected"]["glimpse_digit"]
                    and bool(item["response"]["glimpse_target_match"])
                    == item["expected"]["glimpse_target_match"]
                    and bool(item["response"]["found_target"]) == item["expected"]["found_target"]
                    and item["response"]["unresolved_cells"] == item["expected"]["unresolved_cells"]
                )
                for item in results
            )
            / denom
        ),
        "examples": results,
    }
