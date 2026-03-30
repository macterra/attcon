from __future__ import annotations

"""Natural-language reportability helpers for Stage 7 experiments."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

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
    prev_attended_cell: int
    prev_attended_visible_type: int
    prev_attended_digit: int
    prev_glimpse_digit: int
    glimpse_target_match: bool
    found_target: bool
    unresolved_cells: list[int]
    unresolved_rows: list[int]
    unresolved_cols: list[int]
    unresolved_count: int
    controller_state: torch.Tensor
    prev_controller_state: torch.Tensor
    attention_state: torch.Tensor
    prev_attention_state: torch.Tensor
    memory_state: torch.Tensor
    symbolic_state: str
    tokenized_state: str
    observation_only: str


def _cell_name(cell_idx: int, grid_size: int) -> str:
    row, col = divmod(cell_idx, grid_size)
    return f"r{row}c{col}"


def _render_symbolic_state(example: NLExample, grid_size: int) -> str:
    unresolved = ", ".join(_cell_name(cell, grid_size) for cell in example.unresolved_cells) or "none"
    unresolved_rows = " ".join(str(row) for row in example.unresolved_rows) or "none"
    unresolved_cols = " ".join(str(col) for col in example.unresolved_cols) or "none"
    attended = _cell_name(example.attended_cell, grid_size)
    return (
        f"search_type={example.cue}\n"
        f"attended_cell={attended}\n"
        f"attended_visible_type={example.attended_visible_type}\n"
        f"attended_digit={example.attended_digit}\n"
        f"glimpse_digit={example.glimpse_digit}\n"
        f"previous_attended_cell={_cell_name(example.prev_attended_cell, grid_size)}\n"
        f"previous_attended_visible_type={example.prev_attended_visible_type}\n"
        f"previous_attended_digit={example.prev_attended_digit}\n"
        f"previous_glimpse_digit={example.prev_glimpse_digit}\n"
        f"glimpse_target_match={str(example.glimpse_target_match).lower()}\n"
        f"found_target={str(example.found_target).lower()}\n"
        f"unresolved_rows={unresolved_rows}\n"
        f"unresolved_cols={unresolved_cols}\n"
        f"unresolved_count={example.unresolved_count}\n"
        f"unresolved_cells={unresolved}"
    )


def _chunk_bits(vector: torch.Tensor, num_bits: int = 4) -> list[int]:
    """Produce a few coarse binary bits from chunked latent activations."""

    chunks = torch.chunk(vector, num_bits)
    return [int(chunk.mean().item() >= 0.0) for chunk in chunks]


def _fit_multiclass_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    epochs: int = 200,
    learning_rate: float = 0.05,
) -> nn.Linear:
    """Fit a small linear classifier and return the trained head."""

    head = nn.Linear(features.shape[-1], num_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    for _ in range(epochs):
        logits = head(features)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head


def _fit_binary_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 200,
    learning_rate: float = 0.05,
) -> nn.Linear:
    """Fit a small linear binary probe and return the trained head."""

    head = nn.Linear(features.shape[-1], labels.shape[-1])
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    for _ in range(epochs):
        logits = head(features)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head


def _feature_matrix(examples: list[NLExample]) -> torch.Tensor:
    return torch.stack(
        [
            torch.cat(
                [
                    example.controller_state,
                    example.prev_controller_state,
                    example.attention_state,
                    example.prev_attention_state,
                    example.memory_state,
                ],
                dim=0,
            )
            for example in examples
        ],
        dim=0,
    )


def _render_tokenized_examples(
    calibration_examples: list[NLExample],
    target_examples: list[NLExample],
) -> None:
    """Build learned opaque token streams using a translator fit on calibration examples only."""

    train_features = _feature_matrix(calibration_examples)
    target_features = _feature_matrix(target_examples)
    current_cell_head = _fit_multiclass_probe(
        train_features, torch.tensor([example.attended_cell for example in calibration_examples]), num_classes=25
    )
    current_visible_type_head = _fit_multiclass_probe(
        train_features,
        torch.tensor([example.attended_visible_type for example in calibration_examples]),
        num_classes=8,
    )
    current_digit_head = _fit_multiclass_probe(
        train_features, torch.tensor([example.attended_digit for example in calibration_examples]), num_classes=10
    )
    glimpse_digit_head = _fit_multiclass_probe(
        train_features, torch.tensor([example.glimpse_digit for example in calibration_examples]), num_classes=10
    )
    prev_cell_head = _fit_multiclass_probe(
        train_features, torch.tensor([example.prev_attended_cell for example in calibration_examples]), num_classes=25
    )
    prev_visible_type_head = _fit_multiclass_probe(
        train_features,
        torch.tensor([example.prev_attended_visible_type for example in calibration_examples]),
        num_classes=8,
    )
    prev_digit_head = _fit_multiclass_probe(
        train_features, torch.tensor([example.prev_attended_digit for example in calibration_examples]), num_classes=10
    )
    prev_glimpse_digit_head = _fit_multiclass_probe(
        train_features, torch.tensor([example.prev_glimpse_digit for example in calibration_examples]), num_classes=10
    )
    unresolved_row_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[int(row in example.unresolved_rows) for row in range(5)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    unresolved_col_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[int(col in example.unresolved_cols) for col in range(5)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    unresolved_count_head = _fit_multiclass_probe(
        train_features,
        torch.tensor([example.unresolved_count for example in calibration_examples]),
        num_classes=26,
    )

    with torch.no_grad():
        current_cell_pred = current_cell_head(target_features).argmax(dim=-1)
        current_visible_type_pred = current_visible_type_head(target_features).argmax(dim=-1)
        current_digit_pred = current_digit_head(target_features).argmax(dim=-1)
        glimpse_digit_pred = glimpse_digit_head(target_features).argmax(dim=-1)
        prev_cell_pred = prev_cell_head(target_features).argmax(dim=-1)
        prev_visible_type_pred = prev_visible_type_head(target_features).argmax(dim=-1)
        prev_digit_pred = prev_digit_head(target_features).argmax(dim=-1)
        prev_glimpse_digit_pred = prev_glimpse_digit_head(target_features).argmax(dim=-1)
        unresolved_row_pred = (torch.sigmoid(unresolved_row_head(target_features)) >= 0.5).long()
        unresolved_col_pred = (torch.sigmoid(unresolved_col_head(target_features)) >= 0.5).long()
        unresolved_count_pred = unresolved_count_head(target_features).argmax(dim=-1)

    for idx, example in enumerate(target_examples):
        current_bits = _chunk_bits(example.controller_state)
        prev_bits = _chunk_bits(example.prev_controller_state)
        attention_bits = _chunk_bits(example.attention_state)
        prev_attention_bits = _chunk_bits(example.prev_attention_state)
        memory_bits = _chunk_bits(example.memory_state)
        tokens = [
            "x900",
            f"x{100 + example.cue}",
            "x901",
            f"x{1000 + int(current_cell_pred[idx].item())}",
            f"x{1100 + int(current_visible_type_pred[idx].item())}",
            f"x{1110 + int(current_digit_pred[idx].item())}",
            f"x{1120 + int(glimpse_digit_pred[idx].item())}",
            *[f"x{1020 + bit_idx * 2 + bit}" for bit_idx, bit in enumerate(current_bits)],
            *[f"x{1040 + bit_idx * 2 + bit}" for bit_idx, bit in enumerate(attention_bits)],
            "x910",
            f"x{1200 + int(prev_cell_pred[idx].item())}",
            f"x{1300 + int(prev_visible_type_pred[idx].item())}",
            f"x{1310 + int(prev_digit_pred[idx].item())}",
            f"x{1320 + int(prev_glimpse_digit_pred[idx].item())}",
            *[f"x{1120 + bit_idx * 2 + bit}" for bit_idx, bit in enumerate(prev_bits)],
            *[f"x{1140 + bit_idx * 2 + bit}" for bit_idx, bit in enumerate(prev_attention_bits)],
            "x920",
            f"x{1400 + int(unresolved_count_pred[idx].item())}",
            *[f"x{1210 + bit_idx * 2 + bit}" for bit_idx, bit in enumerate(memory_bits)],
        ]
        for row in range(5):
            tokens.append(f"x{1500 + row * 2 + int(unresolved_row_pred[idx, row].item())}")
        tokens.append("x930")
        for col in range(5):
            tokens.append(f"x{1520 + col * 2 + int(unresolved_col_pred[idx, col].item())}")
        example.tokenized_state = " ".join(tokens)


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
        f"current_glimpse_digit_argmax={digit_idx}\n"
        f"memory_about_previous_attention=not_available\n"
        f"unresolved_information=not_available"
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
    controller_states = outputs["controller_state_seq"]
    visible_types = batch.visible_types

    examples: list[NLExample] = []
    for batch_idx in range(batch.scene.shape[0]):
        for step_idx in range(task_cfg.num_steps):
            attended_cell = int(attention[batch_idx, step_idx].argmax().item())
            attended_visible_type = int(visible_types[batch_idx, attended_cell].item())
            attended_digit = int(batch.digits[batch_idx, attended_cell].item())
            glimpse_digit = int(observation[batch_idx, step_idx, 1:].argmax().item())
            prev_step_idx = max(step_idx - 1, 0)
            prev_attended_cell = int(attention[batch_idx, prev_step_idx].argmax().item())
            prev_attended_visible_type = int(visible_types[batch_idx, prev_attended_cell].item())
            prev_attended_digit = int(batch.digits[batch_idx, prev_attended_cell].item())
            prev_glimpse_digit = int(observation[batch_idx, prev_step_idx, 1:].argmax().item())
            glimpse_target_match = bool(observation[batch_idx, step_idx, 0].item() >= 0.5)
            unresolved = torch.nonzero(
                inspection[batch_idx, step_idx] < 0.5, as_tuple=False
            ).flatten().tolist()
            unresolved = [int(cell) for cell in unresolved]
            example = NLExample(
                example_id=f"scene{batch_idx}_step{step_idx}",
                step_index=step_idx,
                cue=int(batch.cue[batch_idx].item()),
                attended_cell=attended_cell,
                attended_visible_type=attended_visible_type,
                attended_digit=attended_digit,
                glimpse_digit=glimpse_digit,
                prev_attended_cell=prev_attended_cell,
                prev_attended_visible_type=prev_attended_visible_type,
                prev_attended_digit=prev_attended_digit,
                prev_glimpse_digit=prev_glimpse_digit,
                glimpse_target_match=glimpse_target_match,
                found_target=bool(found_state[batch_idx, step_idx].item() >= 0.5),
                unresolved_cells=unresolved,
                unresolved_rows=sorted({cell // task_cfg.grid_size for cell in unresolved}),
                unresolved_cols=sorted({cell % task_cfg.grid_size for cell in unresolved}),
                unresolved_count=len(unresolved),
                controller_state=controller_states[batch_idx, step_idx].detach().cpu(),
                prev_controller_state=controller_states[batch_idx, prev_step_idx].detach().cpu(),
                attention_state=attention[batch_idx, step_idx].detach().cpu(),
                prev_attention_state=attention[batch_idx, prev_step_idx].detach().cpu(),
                memory_state=torch.cat(
                    [
                        inspection[batch_idx, step_idx].detach().cpu(),
                        found_state[batch_idx, step_idx : step_idx + 1].detach().cpu(),
                    ],
                    dim=0,
                ),
                symbolic_state="",
                tokenized_state="",
                observation_only="",
            )
            example.symbolic_state = _render_symbolic_state(example, task_cfg.grid_size)
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
                "previous_attended_cell": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "previous_attended_visible_type": {"type": "integer"},
                "previous_attended_digit": {"type": "integer"},
                "previous_glimpse_digit": {"type": "integer"},
                "glimpse_target_match": {"type": "boolean"},
                "found_target": {"type": "boolean"},
                "unresolved_rows": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "unresolved_cols": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "unresolved_count": {"type": "integer"},
            },
            "required": [
                "natural_language_report",
                "search_type",
                "attended_cell",
                "attended_visible_type",
                "attended_digit",
                "glimpse_digit",
                "previous_attended_cell",
                "previous_attended_visible_type",
                "previous_attended_digit",
                "previous_glimpse_digit",
                "glimpse_target_match",
                "found_target",
                "unresolved_rows",
                "unresolved_cols",
                "unresolved_count",
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
            "You receive opaque state tokens. Infer their meanings from the examples. "
            "The token stream is organized into repeated latent sections for current attention, "
            "previous attention, and unresolved-state summaries, but the tokens are not pre-labeled. "
            "Use the examples to decode the same hidden structure in the evaluation case."
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
                f"search type {example.cue}; attend {_cell_name(example.attended_cell, grid_size)}; "
                f"visible type {example.attended_visible_type}; attended digit {example.attended_digit}; "
                f"glimpse digit {example.glimpse_digit}; previous attend {_cell_name(example.prev_attended_cell, grid_size)}; "
                f"previous visible type {example.prev_attended_visible_type}; previous attended digit {example.prev_attended_digit}; "
                f"previous glimpse digit {example.prev_glimpse_digit}; match {str(example.glimpse_target_match).lower()}; "
                f"found {str(example.found_target).lower()}; rows {example.unresolved_rows}; cols {example.unresolved_cols}; unresolved {example.unresolved_count}"
            ),
            "search_type": example.cue,
            "attended_cell": list(divmod(example.attended_cell, grid_size)),
            "attended_visible_type": example.attended_visible_type,
            "attended_digit": example.attended_digit,
            "glimpse_digit": example.glimpse_digit,
            "previous_attended_cell": list(divmod(example.prev_attended_cell, grid_size)),
            "previous_attended_visible_type": example.prev_attended_visible_type,
            "previous_attended_digit": example.prev_attended_digit,
            "previous_glimpse_digit": example.prev_glimpse_digit,
            "glimpse_target_match": example.glimpse_target_match,
            "found_target": example.found_target,
            "unresolved_rows": example.unresolved_rows,
            "unresolved_cols": example.unresolved_cols,
            "unresolved_count": example.unresolved_count,
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

    client = OpenAI(max_retries=0, timeout=45.0)
    results = []
    exact_search = 0
    exact_attended = 0
    exact_visible_type = 0
    exact_attended_digit = 0
    exact_glimpse_digit = 0
    exact_prev_attended = 0
    exact_prev_visible_type = 0
    exact_prev_attended_digit = 0
    exact_prev_glimpse_digit = 0
    exact_glimpse_match = 0
    exact_found = 0
    exact_unresolved_rows = 0
    exact_unresolved_cols = 0
    exact_unresolved_count = 0
    for example in evaluation_examples:
        parsed = None
        last_error = None
        for _ in range(6):
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
        expected_attended = list(divmod(example.attended_cell, grid_size))
        expected_prev_attended = list(divmod(example.prev_attended_cell, grid_size))

        exact_search += int(parsed["search_type"] == example.cue)
        exact_attended += int(parsed["attended_cell"] == expected_attended)
        exact_visible_type += int(parsed["attended_visible_type"] == example.attended_visible_type)
        exact_attended_digit += int(parsed["attended_digit"] == example.attended_digit)
        exact_glimpse_digit += int(parsed["glimpse_digit"] == example.glimpse_digit)
        exact_prev_attended += int(parsed["previous_attended_cell"] == expected_prev_attended)
        exact_prev_visible_type += int(
            parsed["previous_attended_visible_type"] == example.prev_attended_visible_type
        )
        exact_prev_attended_digit += int(parsed["previous_attended_digit"] == example.prev_attended_digit)
        exact_prev_glimpse_digit += int(parsed["previous_glimpse_digit"] == example.prev_glimpse_digit)
        exact_glimpse_match += int(bool(parsed["glimpse_target_match"]) == example.glimpse_target_match)
        exact_found += int(bool(parsed["found_target"]) == example.found_target)
        exact_unresolved_rows += int(parsed["unresolved_rows"] == example.unresolved_rows)
        exact_unresolved_cols += int(parsed["unresolved_cols"] == example.unresolved_cols)
        exact_unresolved_count += int(parsed["unresolved_count"] == example.unresolved_count)

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
                    "previous_attended_cell": expected_prev_attended,
                    "previous_attended_visible_type": example.prev_attended_visible_type,
                    "previous_attended_digit": example.prev_attended_digit,
                    "previous_glimpse_digit": example.prev_glimpse_digit,
                    "glimpse_target_match": example.glimpse_target_match,
                    "found_target": example.found_target,
                    "unresolved_rows": example.unresolved_rows,
                    "unresolved_cols": example.unresolved_cols,
                    "unresolved_count": example.unresolved_count,
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
        "previous_attended_cell_accuracy": exact_prev_attended / denom,
        "previous_attended_visible_type_accuracy": exact_prev_visible_type / denom,
        "previous_attended_digit_accuracy": exact_prev_attended_digit / denom,
        "previous_glimpse_digit_accuracy": exact_prev_glimpse_digit / denom,
        "glimpse_target_match_accuracy": exact_glimpse_match / denom,
        "found_target_accuracy": exact_found / denom,
        "unresolved_rows_accuracy": exact_unresolved_rows / denom,
        "unresolved_cols_accuracy": exact_unresolved_cols / denom,
        "unresolved_count_accuracy": exact_unresolved_count / denom,
        "current_content_joint_accuracy": (
            sum(
                int(
                    item["response"]["attended_visible_type"] == item["expected"]["attended_visible_type"]
                    and item["response"]["attended_digit"] == item["expected"]["attended_digit"]
                    and item["response"]["glimpse_digit"] == item["expected"]["glimpse_digit"]
                    and bool(item["response"]["glimpse_target_match"])
                    == item["expected"]["glimpse_target_match"]
                )
                for item in results
            )
            / denom
        ),
        "memory_content_joint_accuracy": (
            sum(
                int(
                    item["response"]["previous_attended_visible_type"]
                    == item["expected"]["previous_attended_visible_type"]
                    and item["response"]["previous_attended_digit"]
                    == item["expected"]["previous_attended_digit"]
                    and item["response"]["previous_glimpse_digit"]
                    == item["expected"]["previous_glimpse_digit"]
                )
                for item in results
            )
            / denom
        ),
        "content_only_joint_accuracy": (
            sum(
                int(
                    item["response"]["attended_visible_type"] == item["expected"]["attended_visible_type"]
                    and item["response"]["attended_digit"] == item["expected"]["attended_digit"]
                    and item["response"]["glimpse_digit"] == item["expected"]["glimpse_digit"]
                    and item["response"]["previous_attended_visible_type"]
                    == item["expected"]["previous_attended_visible_type"]
                    and item["response"]["previous_attended_digit"]
                    == item["expected"]["previous_attended_digit"]
                    and item["response"]["previous_glimpse_digit"]
                    == item["expected"]["previous_glimpse_digit"]
                    and bool(item["response"]["glimpse_target_match"])
                    == item["expected"]["glimpse_target_match"]
                    and bool(item["response"]["found_target"]) == item["expected"]["found_target"]
                )
                for item in results
            )
            / denom
        ),
        "joint_accuracy": (
            sum(
                int(
                    item["response"]["search_type"] == item["expected"]["search_type"]
                    and item["response"]["attended_cell"] == item["expected"]["attended_cell"]
                    and item["response"]["attended_visible_type"] == item["expected"]["attended_visible_type"]
                    and item["response"]["attended_digit"] == item["expected"]["attended_digit"]
                    and item["response"]["glimpse_digit"] == item["expected"]["glimpse_digit"]
                    and item["response"]["previous_attended_cell"]
                    == item["expected"]["previous_attended_cell"]
                    and item["response"]["previous_attended_visible_type"]
                    == item["expected"]["previous_attended_visible_type"]
                    and item["response"]["previous_attended_digit"]
                    == item["expected"]["previous_attended_digit"]
                    and item["response"]["previous_glimpse_digit"]
                    == item["expected"]["previous_glimpse_digit"]
                    and bool(item["response"]["glimpse_target_match"])
                    == item["expected"]["glimpse_target_match"]
                    and bool(item["response"]["found_target"]) == item["expected"]["found_target"]
                    and item["response"]["unresolved_rows"] == item["expected"]["unresolved_rows"]
                    and item["response"]["unresolved_cols"] == item["expected"]["unresolved_cols"]
                    and item["response"]["unresolved_count"] == item["expected"]["unresolved_count"]
                )
                for item in results
            )
            / denom
        ),
        "examples": results,
    }
