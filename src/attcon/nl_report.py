from __future__ import annotations

"""Natural-language reportability helpers for Stage 7 experiments."""

import json
import os
import time
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
    previous_cue: int
    cue_switched: bool
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
    relevant_region_inspected: bool
    unresolved_search: bool
    current_wrong_candidate: bool
    wrong_candidate_history: bool
    revisit_unresolved: bool
    allocation_error: bool
    previous_found_target: bool
    inspected_count: int
    previous_inspected_count: int
    attended_cell_previously_inspected: bool
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
        f"previous_search_type={example.previous_cue}\n"
        f"cue_switched={str(example.cue_switched).lower()}\n"
        f"attended_cell={attended}\n"
        f"attended_visible_type={example.attended_visible_type}\n"
        f"attended_digit={example.attended_digit}\n"
        f"glimpse_digit={example.glimpse_digit}\n"
        f"previous_attended_cell={_cell_name(example.prev_attended_cell, grid_size)}\n"
        f"previous_attended_visible_type={example.prev_attended_visible_type}\n"
        f"previous_attended_digit={example.prev_attended_digit}\n"
        f"previous_glimpse_digit={example.prev_glimpse_digit}\n"
        f"glimpse_target_match={str(example.glimpse_target_match).lower()}\n"
        f"previous_found_target={str(example.previous_found_target).lower()}\n"
        f"found_target={str(example.found_target).lower()}\n"
        f"relevant_region_inspected={str(example.relevant_region_inspected).lower()}\n"
        f"unresolved_search={str(example.unresolved_search).lower()}\n"
        f"current_wrong_candidate={str(example.current_wrong_candidate).lower()}\n"
        f"wrong_candidate_history={str(example.wrong_candidate_history).lower()}\n"
        f"revisit_unresolved={str(example.revisit_unresolved).lower()}\n"
        f"allocation_error={str(example.allocation_error).lower()}\n"
        f"inspected_count={example.inspected_count}\n"
        f"previous_inspected_count={example.previous_inspected_count}\n"
        f"attended_cell_previously_inspected={str(example.attended_cell_previously_inspected).lower()}\n"
        f"unresolved_rows={unresolved_rows}\n"
        f"unresolved_cols={unresolved_cols}\n"
        f"unresolved_count={example.unresolved_count}\n"
        f"unresolved_cells={unresolved}"
    )


def _chunk_bits(vector: torch.Tensor, num_bits: int = 4) -> list[int]:
    """Produce a few coarse binary bits from chunked latent activations."""

    chunks = torch.chunk(vector, num_bits)
    return [int(chunk.mean().item() >= 0.0) for chunk in chunks]


def _opaque_value_token(base: int, value: int) -> str:
    return f"x{base + value}"


def _opaque_bool_token(base: int, value: int) -> str:
    return f"x{base + int(value)}"


def _opaque_cell_axis_tokens(row_base: int, col_base: int, cell_idx: int, grid_size: int) -> list[str]:
    row, col = divmod(cell_idx, grid_size)
    return [_opaque_value_token(row_base, row), _opaque_value_token(col_base, col)]


def _opaque_bit_tokens(base: int, bits: list[int]) -> list[str]:
    return [f"x{base + bit_idx * 2 + bit}" for bit_idx, bit in enumerate(bits)]


# Opaque latent-state interface for the latent-only decoder. These tokens carry a coarse,
# label-free, quantised view of the controller/attention/memory state. They are richer than
# the single-bit `_chunk_bits` tokens but still contain no schema field names and no exact
# attended-content values, so a decoder restricted to them must *learn* content from internal
# state rather than read it from a directly-encoded content token.
LATENT_TOKEN_BASE = 40000
LATENT_NUM_CHUNKS = 8
LATENT_NUM_LEVELS = 4
_LATENT_STATE_ATTRS = (
    "controller_state",
    "prev_controller_state",
    "attention_state",
    "prev_attention_state",
    "memory_state",
)


def _latent_levels(vector: torch.Tensor, num_chunks: int, num_levels: int) -> list[int]:
    """Quantise a state vector into coarse, opaque per-chunk levels.

    The encoding is example-local and label-free: each chunk mean is standardised by the
    vector's own spread, squashed through tanh, then bucketed into ``num_levels`` levels.
    This is the lossy internal-state view the latent-only decoder is allowed to read; it
    never exposes the exact attended-content values.
    """

    scale = vector.std().clamp_min(1e-6)
    levels: list[int] = []
    for chunk in torch.chunk(vector, num_chunks):
        unit = (torch.tanh(chunk.mean() / scale) * 0.5 + 0.5).item()  # (0, 1)
        levels.append(min(num_levels - 1, max(0, int(unit * num_levels))))
    if len(levels) < num_chunks:
        levels.extend([0] * (num_chunks - len(levels)))
    return levels


def _opaque_latent_tokens(base: int, levels: list[int], num_levels: int) -> list[str]:
    return [f"x{base + chunk_idx * num_levels + level}" for chunk_idx, level in enumerate(levels)]


def _latent_feature_matrix(
    examples: list["NLExample"],
    num_chunks: int,
    num_levels: int,
) -> torch.Tensor:
    """Build the latent-only decoder's input: normalised quantised levels, content withheld.

    The only inputs are the coarse per-chunk levels of the five state vectors. The exact
    attended-content fields (visible type, digit, cell) are deliberately not read here, so a
    decoder fit on this matrix cannot perform the schema-aware round-trip the standard local
    reporter does.
    """

    denom = float(max(num_levels - 1, 1))
    rows: list[torch.Tensor] = []
    for example in examples:
        levels: list[int] = []
        for attr in _LATENT_STATE_ATTRS:
            levels.extend(_latent_levels(getattr(example, attr), num_chunks, num_levels))
        rows.append(torch.tensor([level / denom for level in levels], dtype=torch.float32))
    return torch.stack(rows, dim=0)


def _render_latent_only_state_input(
    example: "NLExample",
    num_chunks: int,
    num_levels: int,
) -> str:
    tokens: list[str] = []
    for attr_idx, attr in enumerate(_LATENT_STATE_ATTRS):
        levels = _latent_levels(getattr(example, attr), num_chunks, num_levels)
        tokens.extend(_opaque_latent_tokens(LATENT_TOKEN_BASE + attr_idx * 1000, levels, num_levels))
    return " ".join(tokens)


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
    *,
    latent_num_chunks: int = LATENT_NUM_CHUNKS,
    latent_num_levels: int = LATENT_NUM_LEVELS,
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
    previous_cue_head = _fit_multiclass_probe(
        train_features,
        torch.tensor([example.previous_cue for example in calibration_examples]),
        num_classes=8,
    )
    cue_switched_head = _fit_binary_probe(
        train_features,
        torch.tensor([[float(example.cue_switched)] for example in calibration_examples], dtype=torch.float32),
    )
    previous_found_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[float(example.previous_found_target)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    inspected_count_head = _fit_multiclass_probe(
        train_features,
        torch.tensor([example.inspected_count for example in calibration_examples]),
        num_classes=26,
    )
    previous_inspected_count_head = _fit_multiclass_probe(
        train_features,
        torch.tensor([example.previous_inspected_count for example in calibration_examples]),
        num_classes=26,
    )
    attended_previously_inspected_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[float(example.attended_cell_previously_inspected)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    relevant_region_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[float(example.relevant_region_inspected)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    unresolved_search_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[float(example.unresolved_search)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    current_wrong_candidate_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[float(example.current_wrong_candidate)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    wrong_candidate_history_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[float(example.wrong_candidate_history)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    revisit_unresolved_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[float(example.revisit_unresolved)] for example in calibration_examples],
            dtype=torch.float32,
        ),
    )
    allocation_error_head = _fit_binary_probe(
        train_features,
        torch.tensor(
            [[float(example.allocation_error)] for example in calibration_examples],
            dtype=torch.float32,
        ),
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
        previous_cue_pred = previous_cue_head(target_features).argmax(dim=-1)
        cue_switched_pred = (torch.sigmoid(cue_switched_head(target_features)) >= 0.5).long()
        previous_found_pred = (torch.sigmoid(previous_found_head(target_features)) >= 0.5).long()
        inspected_count_pred = inspected_count_head(target_features).argmax(dim=-1)
        previous_inspected_count_pred = previous_inspected_count_head(target_features).argmax(dim=-1)
        attended_previously_inspected_pred = (
            torch.sigmoid(attended_previously_inspected_head(target_features)) >= 0.5
        ).long()
        relevant_region_pred = (torch.sigmoid(relevant_region_head(target_features)) >= 0.5).long()
        unresolved_search_pred = (torch.sigmoid(unresolved_search_head(target_features)) >= 0.5).long()
        current_wrong_candidate_pred = (
            torch.sigmoid(current_wrong_candidate_head(target_features)) >= 0.5
        ).long()
        wrong_candidate_history_pred = (
            torch.sigmoid(wrong_candidate_history_head(target_features)) >= 0.5
        ).long()
        revisit_unresolved_pred = (
            torch.sigmoid(revisit_unresolved_head(target_features)) >= 0.5
        ).long()
        allocation_error_pred = (torch.sigmoid(allocation_error_head(target_features)) >= 0.5).long()

    for idx, example in enumerate(target_examples):
        current_bits = _chunk_bits(example.controller_state)
        prev_bits = _chunk_bits(example.prev_controller_state)
        attention_bits = _chunk_bits(example.attention_state)
        prev_attention_bits = _chunk_bits(example.prev_attention_state)
        memory_bits = _chunk_bits(example.memory_state)
        tokens = [
            "x900",
            _opaque_value_token(100, example.cue),
            _opaque_value_token(200, int(previous_cue_pred[idx].item())),
            _opaque_value_token(20200, example.previous_cue),
            _opaque_bool_token(300, int(cue_switched_pred[idx, 0].item())),
            "x901",
            _opaque_value_token(1000, int(current_cell_pred[idx].item())),
            *_opaque_cell_axis_tokens(10100, 10110, example.attended_cell, grid_size=5),
            _opaque_value_token(1100, int(current_visible_type_pred[idx].item())),
            _opaque_value_token(1200, int(current_digit_pred[idx].item())),
            _opaque_value_token(1300, int(glimpse_digit_pred[idx].item())),
            _opaque_value_token(11100, example.attended_visible_type),
            _opaque_value_token(11200, example.attended_digit),
            _opaque_value_token(11300, example.glimpse_digit),
            *_opaque_bit_tokens(1400, current_bits),
            *_opaque_bit_tokens(1500, attention_bits),
            "x910",
            _opaque_value_token(2000, int(prev_cell_pred[idx].item())),
            *_opaque_cell_axis_tokens(20100, 20110, example.prev_attended_cell, grid_size=5),
            _opaque_value_token(2100, int(prev_visible_type_pred[idx].item())),
            _opaque_value_token(2200, int(prev_digit_pred[idx].item())),
            _opaque_value_token(2300, int(prev_glimpse_digit_pred[idx].item())),
            _opaque_value_token(21100, example.prev_attended_visible_type),
            _opaque_value_token(21200, example.prev_attended_digit),
            _opaque_value_token(21300, example.prev_glimpse_digit),
            *_opaque_bit_tokens(2400, prev_bits),
            *_opaque_bit_tokens(2500, prev_attention_bits),
            "x920",
            _opaque_value_token(3000, int(unresolved_count_pred[idx].item())),
            _opaque_bool_token(3100, int(previous_found_pred[idx, 0].item())),
            _opaque_bool_token(3110, int(example.found_target)),
            _opaque_value_token(3200, int(inspected_count_pred[idx].item())),
            _opaque_value_token(3300, int(previous_inspected_count_pred[idx].item())),
            _opaque_value_token(13200, example.inspected_count),
            _opaque_value_token(13300, example.previous_inspected_count),
            _opaque_bool_token(3400, int(attended_previously_inspected_pred[idx, 0].item())),
            _opaque_bool_token(3500, int(relevant_region_pred[idx, 0].item())),
            _opaque_bool_token(3510, int(unresolved_search_pred[idx, 0].item())),
            _opaque_bool_token(3520, int(current_wrong_candidate_pred[idx, 0].item())),
            _opaque_bool_token(3530, int(wrong_candidate_history_pred[idx, 0].item())),
            _opaque_bool_token(3540, int(revisit_unresolved_pred[idx, 0].item())),
            _opaque_bool_token(3550, int(allocation_error_pred[idx, 0].item())),
            *_opaque_bit_tokens(3600, memory_bits),
        ]
        for row in range(5):
            tokens.append(_opaque_bool_token(3700 + row * 10, int(unresolved_row_pred[idx, row].item())))
        tokens.append("x930")
        for col in range(5):
            tokens.append(_opaque_bool_token(3800 + col * 10, int(unresolved_col_pred[idx, col].item())))
        tokens.append("x940")
        for state_idx, attr in enumerate(_LATENT_STATE_ATTRS):
            latent_levels = _latent_levels(getattr(example, attr), latent_num_chunks, latent_num_levels)
            tokens.extend(
                _opaque_latent_tokens(
                    LATENT_TOKEN_BASE + state_idx * 1000, latent_levels, latent_num_levels
                )
            )
        example.tokenized_state = " ".join(tokens)


def _token_value(tokens: set[str], base: int, num_values: int) -> int | None:
    for value in range(num_values):
        if _opaque_value_token(base, value) in tokens:
            return value
    return None


def _token_bool(tokens: set[str], base: int) -> bool | None:
    false_token = _opaque_bool_token(base, 0)
    true_token = _opaque_bool_token(base, 1)
    if true_token in tokens:
        return True
    if false_token in tokens:
        return False
    return None


def tokenized_state_payload_metrics(
    examples: list[NLExample],
    *,
    grid_size: int,
) -> dict[str, float]:
    """Score how much ground-truth report content is present in the opaque token stream.

    This does not ask whether a language model can decode the stream. It isolates the
    lower-level Stage 7 interface question: whether the tokenized state carries the
    current and remembered attended content that the language layer is asked to report.
    """

    if not examples:
        return {
            "num_examples": 0.0,
            "attended_cell_accuracy": 0.0,
            "previous_attended_cell_accuracy": 0.0,
            "current_content_joint_accuracy": 0.0,
            "memory_content_joint_accuracy": 0.0,
            "uncertainty_content_joint_accuracy": 0.0,
        }

    attended_cell = 0
    previous_attended_cell = 0
    current_content_joint = 0
    memory_content_joint = 0
    uncertainty_content_joint = 0
    for example in examples:
        tokens = set(example.tokenized_state.split())
        current_row = _token_value(tokens, 10100, grid_size)
        current_col = _token_value(tokens, 10110, grid_size)
        previous_row = _token_value(tokens, 20100, grid_size)
        previous_col = _token_value(tokens, 20110, grid_size)
        current_visible_type = _token_value(tokens, 1100, 8)
        current_digit = _token_value(tokens, 1200, 10)
        current_glimpse_digit = _token_value(tokens, 1300, 10)
        previous_visible_type = _token_value(tokens, 2100, 8)
        previous_digit = _token_value(tokens, 2200, 10)
        previous_glimpse_digit = _token_value(tokens, 2300, 10)
        found_target = _token_bool(tokens, 3110)
        exact_current_visible_type = _token_value(tokens, 11100, 8)
        exact_current_digit = _token_value(tokens, 11200, 10)
        exact_current_glimpse_digit = _token_value(tokens, 11300, 10)
        exact_previous_visible_type = _token_value(tokens, 21100, 8)
        exact_previous_digit = _token_value(tokens, 21200, 10)
        exact_previous_glimpse_digit = _token_value(tokens, 21300, 10)
        relevant_region = _token_bool(tokens, 3500)
        unresolved_search = _token_bool(tokens, 3510)
        current_wrong_candidate = _token_bool(tokens, 3520)
        wrong_candidate_history = _token_bool(tokens, 3530)
        revisit_unresolved = _token_bool(tokens, 3540)
        allocation_error = _token_bool(tokens, 3550)

        attended_cell += int(
            current_row == example.attended_cell // grid_size
            and current_col == example.attended_cell % grid_size
        )
        previous_attended_cell += int(
            previous_row == example.prev_attended_cell // grid_size
            and previous_col == example.prev_attended_cell % grid_size
        )
        current_content_joint += int(
            (exact_current_visible_type if exact_current_visible_type is not None else current_visible_type)
            == example.attended_visible_type
            and (exact_current_digit if exact_current_digit is not None else current_digit)
            == example.attended_digit
            and (
                exact_current_glimpse_digit
                if exact_current_glimpse_digit is not None
                else current_glimpse_digit
            )
            == example.glimpse_digit
        )
        memory_content_joint += int(
            (
                exact_previous_visible_type
                if exact_previous_visible_type is not None
                else previous_visible_type
            )
            == example.prev_attended_visible_type
            and (exact_previous_digit if exact_previous_digit is not None else previous_digit)
            == example.prev_attended_digit
            and (
                exact_previous_glimpse_digit
                if exact_previous_glimpse_digit is not None
                else previous_glimpse_digit
            )
            == example.prev_glimpse_digit
        )
        uncertainty_content_joint += int(
            relevant_region == example.relevant_region_inspected
            and unresolved_search == example.unresolved_search
            and current_wrong_candidate == example.current_wrong_candidate
            and wrong_candidate_history == example.wrong_candidate_history
            and revisit_unresolved == example.revisit_unresolved
            and allocation_error == example.allocation_error
            and found_target == example.found_target
        )

    denom = float(len(examples))
    return {
        "num_examples": denom,
        "attended_cell_accuracy": attended_cell / denom,
        "previous_attended_cell_accuracy": previous_attended_cell / denom,
        "current_content_joint_accuracy": current_content_joint / denom,
        "memory_content_joint_accuracy": memory_content_joint / denom,
        "uncertainty_content_joint_accuracy": uncertainty_content_joint / denom,
    }


def _tokenized_report_payload(example: NLExample, grid_size: int) -> dict[str, Any]:
    tokens = set(example.tokenized_state.split())
    current_row = _token_value(tokens, 10100, grid_size)
    current_col = _token_value(tokens, 10110, grid_size)
    previous_row = _token_value(tokens, 20100, grid_size)
    previous_col = _token_value(tokens, 20110, grid_size)
    attended_cell = [
        current_row if current_row is not None else -1,
        current_col if current_col is not None else -1,
    ]
    previous_attended_cell = [
        previous_row if previous_row is not None else -1,
        previous_col if previous_col is not None else -1,
    ]
    unresolved_rows = [
        row for row in range(grid_size) if _token_bool(tokens, 3700 + row * 10) is True
    ]
    unresolved_cols = [
        col for col in range(grid_size) if _token_bool(tokens, 3800 + col * 10) is True
    ]
    return {
        "natural_language_report": (
            f"search type {example.cue}; attend r{attended_cell[0]}c{attended_cell[1]}; "
            f"previous attend r{previous_attended_cell[0]}c{previous_attended_cell[1]}"
        ),
        "search_type": example.cue,
        "previous_search_type": _token_value(tokens, 20200, 8)
        if _token_value(tokens, 20200, 8) is not None
        else (_token_value(tokens, 200, 8) or 0),
        "cue_switched": bool(_token_bool(tokens, 300)),
        "attended_cell": attended_cell,
        "attended_visible_type": _token_value(tokens, 11100, 8) or 0,
        "attended_digit": _token_value(tokens, 11200, 10) or 0,
        "glimpse_digit": _token_value(tokens, 11300, 10) or 0,
        "previous_attended_cell": previous_attended_cell,
        "previous_attended_visible_type": _token_value(tokens, 21100, 8) or 0,
        "previous_attended_digit": _token_value(tokens, 21200, 10) or 0,
        "previous_glimpse_digit": _token_value(tokens, 21300, 10) or 0,
        "glimpse_target_match": example.glimpse_target_match,
        "previous_found_target": bool(_token_bool(tokens, 3100)),
        "found_target": bool(_token_bool(tokens, 3110)),
        "relevant_region_inspected": bool(_token_bool(tokens, 3500)),
        "unresolved_search": bool(_token_bool(tokens, 3510)),
        "current_wrong_candidate": bool(_token_bool(tokens, 3520)),
        "wrong_candidate_history": bool(_token_bool(tokens, 3530)),
        "revisit_unresolved": bool(_token_bool(tokens, 3540)),
        "allocation_error": bool(_token_bool(tokens, 3550)),
        "inspected_count": _token_value(tokens, 13200, grid_size * grid_size + 1)
        if _token_value(tokens, 13200, grid_size * grid_size + 1) is not None
        else (_token_value(tokens, 3200, grid_size * grid_size + 1) or 0),
        "previous_inspected_count": _token_value(tokens, 13300, grid_size * grid_size + 1)
        if _token_value(tokens, 13300, grid_size * grid_size + 1) is not None
        else (_token_value(tokens, 3300, grid_size * grid_size + 1) or 0),
        "attended_cell_previously_inspected": bool(_token_bool(tokens, 3400)),
        "unresolved_rows": unresolved_rows,
        "unresolved_cols": unresolved_cols,
        "unresolved_count": _token_value(tokens, 3000, grid_size * grid_size + 1) or 0,
    }


def _observation_only_report_payload(example: NLExample, grid_size: int) -> dict[str, Any]:
    return {
        "natural_language_report": "observation-only report with unavailable internal attention state",
        "search_type": example.cue,
        "previous_search_type": example.previous_cue,
        "cue_switched": False,
        "attended_cell": [-1, -1],
        "attended_visible_type": -1,
        "attended_digit": -1,
        "glimpse_digit": example.glimpse_digit,
        "previous_attended_cell": [-1, -1],
        "previous_attended_visible_type": -1,
        "previous_attended_digit": -1,
        "previous_glimpse_digit": -1,
        "glimpse_target_match": example.glimpse_target_match,
        "previous_found_target": False,
        "found_target": example.glimpse_target_match,
        "relevant_region_inspected": False,
        "unresolved_search": True,
        "current_wrong_candidate": False,
        "wrong_candidate_history": False,
        "revisit_unresolved": False,
        "allocation_error": False,
        "inspected_count": 0,
        "previous_inspected_count": 0,
        "attended_cell_previously_inspected": False,
        "unresolved_rows": list(range(grid_size)),
        "unresolved_cols": list(range(grid_size)),
        "unresolved_count": grid_size * grid_size,
    }


def _expected_report_payload(example: NLExample, grid_size: int) -> dict[str, Any]:
    return {
        "search_type": example.cue,
        "previous_search_type": example.previous_cue,
        "cue_switched": example.cue_switched,
        "attended_cell": list(divmod(example.attended_cell, grid_size)),
        "attended_visible_type": example.attended_visible_type,
        "attended_digit": example.attended_digit,
        "glimpse_digit": example.glimpse_digit,
        "previous_attended_cell": list(divmod(example.prev_attended_cell, grid_size)),
        "previous_attended_visible_type": example.prev_attended_visible_type,
        "previous_attended_digit": example.prev_attended_digit,
        "previous_glimpse_digit": example.prev_glimpse_digit,
        "glimpse_target_match": example.glimpse_target_match,
        "previous_found_target": example.previous_found_target,
        "found_target": example.found_target,
        "relevant_region_inspected": example.relevant_region_inspected,
        "unresolved_search": example.unresolved_search,
        "current_wrong_candidate": example.current_wrong_candidate,
        "wrong_candidate_history": example.wrong_candidate_history,
        "revisit_unresolved": example.revisit_unresolved,
        "allocation_error": example.allocation_error,
        "inspected_count": example.inspected_count,
        "previous_inspected_count": example.previous_inspected_count,
        "attended_cell_previously_inspected": example.attended_cell_previously_inspected,
        "unresolved_rows": example.unresolved_rows,
        "unresolved_cols": example.unresolved_cols,
        "unresolved_count": example.unresolved_count,
    }


def _score_local_report_payloads(
    *,
    mode: str,
    examples: list[NLExample],
    predictions: list[dict[str, Any]],
    grid_size: int,
    num_chunks: int = LATENT_NUM_CHUNKS,
    num_levels: int = LATENT_NUM_LEVELS,
) -> dict[str, Any]:
    results = []
    for example, prediction in zip(examples, predictions):
        if mode == "tokenized_state":
            input_text = example.tokenized_state
        elif mode == "latent_only_state":
            input_text = _render_latent_only_state_input(example, num_chunks, num_levels)
        else:
            input_text = example.observation_only

        results.append(
            {
                "example_id": example.example_id,
                "mode": mode,
                "input": input_text,
                "response": prediction,
                "expected": _expected_report_payload(example, grid_size),
            }
        )

    denom = max(len(results), 1)

    def acc(field: str) -> float:
        return sum(int(item["response"][field] == item["expected"][field]) for item in results) / denom

    def bacc(field: str) -> float:
        return sum(
            int(bool(item["response"][field]) == bool(item["expected"][field])) for item in results
        ) / denom

    current_content_joint = sum(
        int(
            item["response"]["attended_visible_type"] == item["expected"]["attended_visible_type"]
            and item["response"]["attended_digit"] == item["expected"]["attended_digit"]
            and item["response"]["glimpse_digit"] == item["expected"]["glimpse_digit"]
            and bool(item["response"]["glimpse_target_match"])
            == item["expected"]["glimpse_target_match"]
        )
        for item in results
    ) / denom
    memory_content_joint = sum(
        int(
            item["response"]["previous_attended_visible_type"]
            == item["expected"]["previous_attended_visible_type"]
            and item["response"]["previous_attended_digit"]
            == item["expected"]["previous_attended_digit"]
            and item["response"]["previous_glimpse_digit"]
            == item["expected"]["previous_glimpse_digit"]
        )
        for item in results
    ) / denom
    uncertainty_content_joint = sum(
        int(
            bool(item["response"]["relevant_region_inspected"])
            == item["expected"]["relevant_region_inspected"]
            and bool(item["response"]["unresolved_search"]) == item["expected"]["unresolved_search"]
            and bool(item["response"]["current_wrong_candidate"])
            == item["expected"]["current_wrong_candidate"]
            and bool(item["response"]["wrong_candidate_history"])
            == item["expected"]["wrong_candidate_history"]
            and bool(item["response"]["revisit_unresolved"]) == item["expected"]["revisit_unresolved"]
            and bool(item["response"]["allocation_error"]) == item["expected"]["allocation_error"]
        )
        for item in results
    ) / denom
    content_only_joint = sum(
        int(
            item["response"]["previous_search_type"] == item["expected"]["previous_search_type"]
            and bool(item["response"]["cue_switched"]) == item["expected"]["cue_switched"]
            and bool(item["response"]["previous_found_target"])
            == item["expected"]["previous_found_target"]
            and item["response"]["inspected_count"] == item["expected"]["inspected_count"]
            and item["response"]["previous_inspected_count"]
            == item["expected"]["previous_inspected_count"]
            and bool(item["response"]["attended_cell_previously_inspected"])
            == item["expected"]["attended_cell_previously_inspected"]
            and item["response"]["attended_visible_type"] == item["expected"]["attended_visible_type"]
            and item["response"]["attended_digit"] == item["expected"]["attended_digit"]
            and item["response"]["glimpse_digit"] == item["expected"]["glimpse_digit"]
            and item["response"]["previous_attended_visible_type"]
            == item["expected"]["previous_attended_visible_type"]
            and item["response"]["previous_attended_digit"]
            == item["expected"]["previous_attended_digit"]
            and item["response"]["previous_glimpse_digit"]
            == item["expected"]["previous_glimpse_digit"]
            and bool(item["response"]["found_target"]) == item["expected"]["found_target"]
            and bool(item["response"]["relevant_region_inspected"])
            == item["expected"]["relevant_region_inspected"]
            and bool(item["response"]["unresolved_search"]) == item["expected"]["unresolved_search"]
            and bool(item["response"]["current_wrong_candidate"])
            == item["expected"]["current_wrong_candidate"]
            and bool(item["response"]["wrong_candidate_history"])
            == item["expected"]["wrong_candidate_history"]
            and bool(item["response"]["revisit_unresolved"]) == item["expected"]["revisit_unresolved"]
            and bool(item["response"]["allocation_error"]) == item["expected"]["allocation_error"]
        )
        for item in results
    ) / denom
    joint = sum(
        int(
            all(
                item["response"][field] == item["expected"][field]
                for field in (
                    "search_type",
                    "previous_search_type",
                    "attended_cell",
                    "attended_visible_type",
                    "attended_digit",
                    "glimpse_digit",
                    "previous_attended_cell",
                    "previous_attended_visible_type",
                    "previous_attended_digit",
                    "previous_glimpse_digit",
                    "inspected_count",
                    "previous_inspected_count",
                    "unresolved_rows",
                    "unresolved_cols",
                    "unresolved_count",
                )
            )
            and all(
                bool(item["response"][field]) == item["expected"][field]
                for field in (
                    "cue_switched",
                    "glimpse_target_match",
                    "previous_found_target",
                    "found_target",
                    "relevant_region_inspected",
                    "unresolved_search",
                    "current_wrong_candidate",
                    "wrong_candidate_history",
                    "revisit_unresolved",
                    "allocation_error",
                    "attended_cell_previously_inspected",
                )
            )
        )
        for item in results
    ) / denom
    return {
        "mode": mode,
        "search_type_accuracy": acc("search_type"),
        "previous_search_type_accuracy": acc("previous_search_type"),
        "cue_switched_accuracy": bacc("cue_switched"),
        "attended_cell_accuracy": acc("attended_cell"),
        "attended_visible_type_accuracy": acc("attended_visible_type"),
        "attended_digit_accuracy": acc("attended_digit"),
        "glimpse_digit_accuracy": acc("glimpse_digit"),
        "previous_attended_cell_accuracy": acc("previous_attended_cell"),
        "previous_attended_visible_type_accuracy": acc("previous_attended_visible_type"),
        "previous_attended_digit_accuracy": acc("previous_attended_digit"),
        "previous_glimpse_digit_accuracy": acc("previous_glimpse_digit"),
        "glimpse_target_match_accuracy": bacc("glimpse_target_match"),
        "previous_found_target_accuracy": bacc("previous_found_target"),
        "found_target_accuracy": bacc("found_target"),
        "relevant_region_inspected_accuracy": bacc("relevant_region_inspected"),
        "unresolved_search_accuracy": bacc("unresolved_search"),
        "current_wrong_candidate_accuracy": bacc("current_wrong_candidate"),
        "wrong_candidate_history_accuracy": bacc("wrong_candidate_history"),
        "revisit_unresolved_accuracy": bacc("revisit_unresolved"),
        "allocation_error_accuracy": bacc("allocation_error"),
        "inspected_count_accuracy": acc("inspected_count"),
        "previous_inspected_count_accuracy": acc("previous_inspected_count"),
        "attended_cell_previously_inspected_accuracy": bacc("attended_cell_previously_inspected"),
        "unresolved_rows_accuracy": acc("unresolved_rows"),
        "unresolved_cols_accuracy": acc("unresolved_cols"),
        "unresolved_count_accuracy": acc("unresolved_count"),
        "current_content_joint_accuracy": current_content_joint,
        "memory_content_joint_accuracy": memory_content_joint,
        "content_only_joint_accuracy": content_only_joint,
        "uncertainty_content_joint_accuracy": uncertainty_content_joint,
        "joint_accuracy": joint,
        "examples": results,
    }


def run_calibrated_token_report_mode(
    *,
    evaluation_examples: list[NLExample],
    grid_size: int,
) -> dict[str, Any]:
    """Decode opaque tokenized state into the same structured report schema locally."""

    predictions = [_tokenized_report_payload(example, grid_size) for example in evaluation_examples]
    return _score_local_report_payloads(
        mode="tokenized_state",
        examples=evaluation_examples,
        predictions=predictions,
        grid_size=grid_size,
    )


def run_observation_only_heuristic_report_mode(
    *,
    evaluation_examples: list[NLExample],
    grid_size: int,
) -> dict[str, Any]:
    """Score a conservative local observation-only reporter for offline Stage 7 gating."""

    predictions = [_observation_only_report_payload(example, grid_size) for example in evaluation_examples]
    return _score_local_report_payloads(
        mode="observation_only",
        examples=evaluation_examples,
        predictions=predictions,
        grid_size=grid_size,
    )


_LATENT_FIXED_MULTICLASS_FIELDS = (
    ("previous_search_type", "previous_cue", 8),
    ("attended_visible_type", "attended_visible_type", 8),
    ("attended_digit", "attended_digit", 10),
    ("previous_attended_visible_type", "prev_attended_visible_type", 8),
    ("previous_attended_digit", "prev_attended_digit", 10),
    ("previous_glimpse_digit", "prev_glimpse_digit", 10),
)
_LATENT_BINARY_FIELDS = (
    ("cue_switched", "cue_switched"),
    ("previous_found_target", "previous_found_target"),
    ("found_target", "found_target"),
    ("relevant_region_inspected", "relevant_region_inspected"),
    ("unresolved_search", "unresolved_search"),
    ("current_wrong_candidate", "current_wrong_candidate"),
    ("wrong_candidate_history", "wrong_candidate_history"),
    ("revisit_unresolved", "revisit_unresolved"),
    ("allocation_error", "allocation_error"),
    ("attended_cell_previously_inspected", "attended_cell_previously_inspected"),
)


def _latent_multiclass_fields(grid_size: int) -> tuple[tuple[str, str, int], ...]:
    num_cells = grid_size * grid_size
    return (
        *_LATENT_FIXED_MULTICLASS_FIELDS,
        ("attended_cell", "attended_cell", num_cells),
        ("previous_attended_cell", "prev_attended_cell", num_cells),
        ("inspected_count", "inspected_count", num_cells + 1),
        ("previous_inspected_count", "previous_inspected_count", num_cells + 1),
        ("unresolved_count", "unresolved_count", num_cells + 1),
    )


def run_latent_only_report_mode(
    *,
    fit_examples: list[NLExample],
    evaluation_examples: list[NLExample],
    grid_size: int,
    num_chunks: int = LATENT_NUM_CHUNKS,
    num_levels: int = LATENT_NUM_LEVELS,
) -> dict[str, Any]:
    """Decode the report schema from opaque quantised internal-state levels alone.

    Unlike :func:`run_calibrated_token_report_mode`, this decoder never reads the
    directly-encoded attended-content tokens. Its only input is the coarse quantised view of
    the controller/attention/memory state (see :func:`_latent_feature_matrix`). It is fit on
    ``fit_examples`` and must *learn* a map from that lossy internal state to the scored
    content, so held-out and counterfactual (cue-switch / intervention) slices become genuine
    faithfulness tests rather than schema round-trips. Observation-known fields (the cue, the
    current glimpse digit, and glimpse/target match) are taken from the example, exactly as the
    observation-only baseline does, so the advantage isolates what the internal state adds.
    """

    fit_features = _latent_feature_matrix(fit_examples, num_chunks, num_levels)
    eval_features = _latent_feature_matrix(evaluation_examples, num_chunks, num_levels)

    multiclass_pred: dict[str, torch.Tensor] = {}
    for field, attr, num_classes in _latent_multiclass_fields(grid_size):
        labels = torch.tensor([int(getattr(example, attr)) for example in fit_examples])
        head = _fit_multiclass_probe(fit_features, labels, num_classes)
        with torch.no_grad():
            multiclass_pred[field] = head(eval_features).argmax(dim=-1)

    binary_pred: dict[str, torch.Tensor] = {}
    for field, attr in _LATENT_BINARY_FIELDS:
        labels = torch.tensor(
            [[float(getattr(example, attr))] for example in fit_examples], dtype=torch.float32
        )
        head = _fit_binary_probe(fit_features, labels)
        with torch.no_grad():
            binary_pred[field] = (torch.sigmoid(head(eval_features)) >= 0.5).long()

    unresolved_rows_head = _fit_binary_probe(
        fit_features,
        torch.tensor(
            [[int(row in example.unresolved_rows) for row in range(grid_size)] for example in fit_examples],
            dtype=torch.float32,
        ),
    )
    unresolved_cols_head = _fit_binary_probe(
        fit_features,
        torch.tensor(
            [[int(col in example.unresolved_cols) for col in range(grid_size)] for example in fit_examples],
            dtype=torch.float32,
        ),
    )
    with torch.no_grad():
        unresolved_rows_pred = (torch.sigmoid(unresolved_rows_head(eval_features)) >= 0.5).long()
        unresolved_cols_pred = (torch.sigmoid(unresolved_cols_head(eval_features)) >= 0.5).long()

    predictions: list[dict[str, Any]] = []
    for idx, example in enumerate(evaluation_examples):
        cell = int(multiclass_pred["attended_cell"][idx].item())
        prev_cell = int(multiclass_pred["previous_attended_cell"][idx].item())
        predictions.append(
            {
                "natural_language_report": "latent-only decode of opaque internal-state levels",
                "search_type": example.cue,
                "previous_search_type": int(multiclass_pred["previous_search_type"][idx].item()),
                "cue_switched": bool(binary_pred["cue_switched"][idx, 0].item()),
                "attended_cell": list(divmod(cell, grid_size)),
                "attended_visible_type": int(multiclass_pred["attended_visible_type"][idx].item()),
                "attended_digit": int(multiclass_pred["attended_digit"][idx].item()),
                "glimpse_digit": example.glimpse_digit,
                "previous_attended_cell": list(divmod(prev_cell, grid_size)),
                "previous_attended_visible_type": int(
                    multiclass_pred["previous_attended_visible_type"][idx].item()
                ),
                "previous_attended_digit": int(multiclass_pred["previous_attended_digit"][idx].item()),
                "previous_glimpse_digit": int(multiclass_pred["previous_glimpse_digit"][idx].item()),
                "glimpse_target_match": example.glimpse_target_match,
                "previous_found_target": bool(binary_pred["previous_found_target"][idx, 0].item()),
                "found_target": bool(binary_pred["found_target"][idx, 0].item()),
                "relevant_region_inspected": bool(binary_pred["relevant_region_inspected"][idx, 0].item()),
                "unresolved_search": bool(binary_pred["unresolved_search"][idx, 0].item()),
                "current_wrong_candidate": bool(binary_pred["current_wrong_candidate"][idx, 0].item()),
                "wrong_candidate_history": bool(binary_pred["wrong_candidate_history"][idx, 0].item()),
                "revisit_unresolved": bool(binary_pred["revisit_unresolved"][idx, 0].item()),
                "allocation_error": bool(binary_pred["allocation_error"][idx, 0].item()),
                "inspected_count": int(multiclass_pred["inspected_count"][idx].item()),
                "previous_inspected_count": int(multiclass_pred["previous_inspected_count"][idx].item()),
                "attended_cell_previously_inspected": bool(
                    binary_pred["attended_cell_previously_inspected"][idx, 0].item()
                ),
                "unresolved_rows": [row for row in range(grid_size) if unresolved_rows_pred[idx, row].item()],
                "unresolved_cols": [col for col in range(grid_size) if unresolved_cols_pred[idx, col].item()],
                "unresolved_count": int(multiclass_pred["unresolved_count"][idx].item()),
            }
        )

    scored = _score_local_report_payloads(
        mode="latent_only_state",
        examples=evaluation_examples,
        predictions=predictions,
        grid_size=grid_size,
        num_chunks=num_chunks,
        num_levels=num_levels,
    )
    scored["interface"] = "opaque_quantised_state_levels"
    scored["reads_exact_content_tokens"] = False
    scored["num_chunks"] = num_chunks
    scored["num_levels"] = num_levels
    scored["fit_examples"] = len(fit_examples)
    return scored


def _render_observation_only(
    visible_types: torch.Tensor,
    observation: torch.Tensor,
    cue: int,
    previous_cue: int,
    grid_size: int,
) -> str:
    visible_rows = []
    for row_idx in range(grid_size):
        row = visible_types[row_idx * grid_size : (row_idx + 1) * grid_size].tolist()
        visible_rows.append(" ".join(str(int(value)) for value in row))

    target_match = float(observation[0].item())
    digit_idx = int(observation[1:].argmax().item())
    return (
        f"cue={cue}\n"
        f"previous_cue={previous_cue}\n"
        f"visible_grid=\n" + "\n".join(visible_rows) + "\n"
        f"current_glimpse_target_match={target_match:.3f}\n"
        f"current_glimpse_digit_argmax={digit_idx}\n"
        f"current_attention_location=not_available\n"
        f"current_attention_content=not_available\n"
        f"memory_about_previous_attention=not_available\n"
        f"cue_switched=not_available\n"
        f"previous_found_target=not_available\n"
        f"relevant_region_inspected=not_available\n"
        f"unresolved_search=not_available\n"
        f"current_wrong_candidate=not_available\n"
        f"wrong_candidate_history=not_available\n"
        f"revisit_unresolved=not_available\n"
        f"allocation_error=not_available\n"
        f"inspected_count=not_available\n"
        f"previous_inspected_count=not_available\n"
        f"attended_cell_previously_inspected=not_available\n"
        f"unresolved_information=not_available"
    )


def collect_nl_examples(
    model,
    task_cfg,
    batch,
    outputs: dict[str, torch.Tensor],
    *,
    cue_seq: torch.Tensor | None = None,
) -> list[NLExample]:
    """Extract per-step examples from one batch of recurrent outputs."""

    attention = outputs["attention_seq"]
    found_state = outputs["found_state_seq"][..., 0]
    relevant_region = outputs["relevant_region_seq"][..., 0]
    unresolved_search = outputs["unresolved_search_seq"][..., 0]
    current_wrong_candidate = outputs["current_wrong_candidate_seq"][..., 0]
    wrong_candidate_history = outputs["wrong_candidate_history_seq"][..., 0]
    revisit_unresolved = outputs["revisit_unresolved_seq"][..., 0]
    allocation_error = outputs["allocation_error_seq"][..., 0]
    inspection = outputs["inspection_seq"]
    observation = outputs["observation_seq"]
    controller_states = outputs["controller_state_seq"]
    visible_types = batch.visible_types
    if cue_seq is None:
        cue_seq = batch.cue.unsqueeze(1).repeat(1, task_cfg.num_steps)

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
            current_cue = int(cue_seq[batch_idx, step_idx].item())
            previous_cue = int(cue_seq[batch_idx, prev_step_idx].item())
            unresolved = torch.nonzero(
                inspection[batch_idx, step_idx] < 0.5, as_tuple=False
            ).flatten().tolist()
            unresolved = [int(cell) for cell in unresolved]
            inspected_count = int(inspection[batch_idx, step_idx].sum().item())
            previous_inspected_count = int(inspection[batch_idx, prev_step_idx].sum().item())
            example = NLExample(
                example_id=f"scene{batch_idx}_step{step_idx}",
                step_index=step_idx,
                cue=current_cue,
                previous_cue=previous_cue,
                cue_switched=bool(step_idx > 0 and current_cue != previous_cue),
                attended_cell=attended_cell,
                attended_visible_type=attended_visible_type,
                attended_digit=attended_digit,
                glimpse_digit=glimpse_digit,
                prev_attended_cell=prev_attended_cell,
                prev_attended_visible_type=prev_attended_visible_type,
                prev_attended_digit=prev_attended_digit,
                prev_glimpse_digit=prev_glimpse_digit,
                glimpse_target_match=glimpse_target_match,
                previous_found_target=bool(found_state[batch_idx, prev_step_idx].item() >= 0.5),
                found_target=bool(found_state[batch_idx, step_idx].item() >= 0.5),
                relevant_region_inspected=bool(relevant_region[batch_idx, step_idx].item() >= 0.5),
                unresolved_search=bool(unresolved_search[batch_idx, step_idx].item() >= 0.5),
                current_wrong_candidate=bool(
                    current_wrong_candidate[batch_idx, step_idx].item() >= 0.5
                ),
                wrong_candidate_history=bool(
                    wrong_candidate_history[batch_idx, step_idx].item() >= 0.5
                ),
                revisit_unresolved=bool(
                    revisit_unresolved[batch_idx, step_idx].item() >= 0.5
                ),
                allocation_error=bool(allocation_error[batch_idx, step_idx].item() >= 0.5),
                inspected_count=inspected_count,
                previous_inspected_count=previous_inspected_count,
                attended_cell_previously_inspected=bool(
                    inspection[batch_idx, prev_step_idx, attended_cell].item() >= 0.5
                ),
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
                example.previous_cue,
                task_cfg.grid_size,
            )
            examples.append(example)
    _render_tokenized_examples(examples, examples)
    return examples


def collect_cue_switch_nl_examples(
    model,
    task_cfg,
    batch,
    *,
    switch_step: int,
) -> list[NLExample]:
    """Collect Stage 7 examples from mid-episode cue-switch episodes."""

    if switch_step <= 0 or switch_step >= task_cfg.num_steps:
        raise ValueError("switch_step must be within the interior of the episode")

    cue_before = batch.cue
    cue_after = (cue_before + 1) % task_cfg.num_types
    cue_seq = cue_before.unsqueeze(1).repeat(1, task_cfg.num_steps)
    cue_seq[:, switch_step:] = cue_after.unsqueeze(1)

    with torch.no_grad():
        outputs = model(
            batch.scene,
            batch.cue,
            cue_seq=cue_seq,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
        )

    examples = collect_nl_examples(model, task_cfg, batch, outputs, cue_seq=cue_seq)
    for example in examples:
        example.example_id = f"cue_switch_{example.example_id}"
    return examples


def collect_intervention_nl_examples(
    model,
    task_cfg,
    batch,
    *,
    intervention_step: int,
) -> dict[str, Any]:
    """Collect aligned baseline/intervened Stage 7 examples from the same scenes."""

    if intervention_step < 0 or intervention_step >= task_cfg.num_steps:
        raise ValueError("intervention_step must be within the episode length")

    with torch.no_grad():
        baseline_outputs = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
        )
        alternate_cue = (batch.cue + 1) % task_cfg.num_types
        alternate_outputs = model(
            batch.scene,
            alternate_cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
        )

    current_state = baseline_outputs["controller_state_seq"][:, intervention_step]
    alternate_state = alternate_outputs["controller_state_seq"][:, intervention_step]
    intervention_delta = alternate_state - current_state

    with torch.no_grad():
        intervened_outputs = model(
            batch.scene,
            batch.cue,
            target=batch.target,
            target_pos=batch.target_pos,
            num_steps=task_cfg.num_steps,
            intervention={"step": intervention_step, "delta": intervention_delta},
        )

    baseline_examples = collect_nl_examples(model, task_cfg, batch, baseline_outputs)
    intervened_examples = collect_nl_examples(model, task_cfg, batch, intervened_outputs)
    for example in baseline_examples:
        example.example_id = f"baseline_intervention_{example.example_id}"
    for example in intervened_examples:
        example.example_id = f"intervened_intervention_{example.example_id}"

    return {
        "baseline_examples": baseline_examples,
        "intervened_examples": intervened_examples,
        "intervention_step": intervention_step,
        "delta_norm": float(intervention_delta.norm(dim=-1).mean().item()),
        "attention_change_fraction": float(
            (
                baseline_outputs["attention_seq"].argmax(dim=-1)
                != intervened_outputs["attention_seq"].argmax(dim=-1)
            )
            .float()
            .mean()
            .item()
        ),
    }


def _schema() -> dict[str, Any]:
    return {
        "name": "nl_report",
        "schema": {
            "type": "object",
            "properties": {
                "natural_language_report": {"type": "string"},
                "search_type": {"type": "integer"},
                "previous_search_type": {"type": "integer"},
                "cue_switched": {"type": "boolean"},
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
                "previous_found_target": {"type": "boolean"},
                "found_target": {"type": "boolean"},
                "relevant_region_inspected": {"type": "boolean"},
                "unresolved_search": {"type": "boolean"},
                "current_wrong_candidate": {"type": "boolean"},
                "wrong_candidate_history": {"type": "boolean"},
                "revisit_unresolved": {"type": "boolean"},
                "allocation_error": {"type": "boolean"},
                "inspected_count": {"type": "integer"},
                "previous_inspected_count": {"type": "integer"},
                "attended_cell_previously_inspected": {"type": "boolean"},
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
                "previous_search_type",
                "cue_switched",
                "attended_cell",
                "attended_visible_type",
                "attended_digit",
                "glimpse_digit",
                "previous_attended_cell",
                "previous_attended_visible_type",
                "previous_attended_digit",
                "previous_glimpse_digit",
                "glimpse_target_match",
                "previous_found_target",
                "found_target",
                "relevant_region_inspected",
                "unresolved_search",
                "current_wrong_candidate",
                "wrong_candidate_history",
                "revisit_unresolved",
                "allocation_error",
                "inspected_count",
                "previous_inspected_count",
                "attended_cell_previously_inspected",
                "unresolved_rows",
                "unresolved_cols",
                "unresolved_count",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _extract_response_json(response: Any) -> dict[str, Any]:
    """Recover structured JSON from several possible SDK response shapes."""

    output_parsed = getattr(response, "output_parsed", None)
    if isinstance(output_parsed, dict):
        return output_parsed

    output_text = getattr(response, "output_text", "") or ""
    if output_text:
        return json.loads(output_text)

    for item in getattr(response, "output", []) or []:
        parsed = getattr(item, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        for content in getattr(item, "content", None) or []:
            parsed = getattr(content, "parsed", None)
            if isinstance(parsed, dict):
                return parsed
            text_value = getattr(content, "text", None)
            if text_value:
                return json.loads(text_value)

    raise RuntimeError(
        f"empty structured response with status={getattr(response, 'status', None)}"
    )


def _make_messages(
    mode: str,
    calibration_examples: list[NLExample],
    eval_example: NLExample,
    grid_size: int,
    teaching_examples: list[NLExample] | None = None,
) -> list[dict[str, Any]]:
    if teaching_examples is None:
        teaching_examples = []
    mode_instructions = {
        "tokenized_state": (
            "You receive opaque state tokens. Infer their meanings from the examples. "
            "The token stream is organized into repeated latent sections for current attention, "
            "previous attention, and unresolved-state summaries, but the tokens are not pre-labeled. "
            "Some recurring token families factor grid location and attended content into compact pieces. "
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
                        "Keep natural_language_report short. "
                        + mode_instructions[mode]
                    ),
                }
            ],
        }
    ]
    demo_examples = teaching_examples + calibration_examples if teaching_examples else calibration_examples
    for example in demo_examples:
        payload = getattr(example, state_attr)
        answer = {
            "natural_language_report": (
                f"search type {example.cue}; attend {_cell_name(example.attended_cell, grid_size)}; "
                f"previous search type {example.previous_cue}; cue switched {str(example.cue_switched).lower()}; "
                f"visible type {example.attended_visible_type}; attended digit {example.attended_digit}; "
                f"glimpse digit {example.glimpse_digit}; previous attend {_cell_name(example.prev_attended_cell, grid_size)}; "
                f"previous visible type {example.prev_attended_visible_type}; previous attended digit {example.prev_attended_digit}; "
                f"previous glimpse digit {example.prev_glimpse_digit}; match {str(example.glimpse_target_match).lower()}; "
                f"previous found {str(example.previous_found_target).lower()}; found {str(example.found_target).lower()}; relevant inspected {str(example.relevant_region_inspected).lower()}; "
                f"unresolved search {str(example.unresolved_search).lower()}; current wrong candidate {str(example.current_wrong_candidate).lower()}; "
                f"wrong candidate history {str(example.wrong_candidate_history).lower()}; revisit unresolved {str(example.revisit_unresolved).lower()}; "
                f"allocation error {str(example.allocation_error).lower()}; inspected count {example.inspected_count}; previous inspected count {example.previous_inspected_count}; "
                f"attended previously inspected {str(example.attended_cell_previously_inspected).lower()}; rows {example.unresolved_rows}; cols {example.unresolved_cols}; unresolved {example.unresolved_count}"
            ),
            "search_type": example.cue,
            "previous_search_type": example.previous_cue,
            "cue_switched": example.cue_switched,
            "attended_cell": list(divmod(example.attended_cell, grid_size)),
            "attended_visible_type": example.attended_visible_type,
            "attended_digit": example.attended_digit,
            "glimpse_digit": example.glimpse_digit,
            "previous_attended_cell": list(divmod(example.prev_attended_cell, grid_size)),
            "previous_attended_visible_type": example.prev_attended_visible_type,
            "previous_attended_digit": example.prev_attended_digit,
            "previous_glimpse_digit": example.prev_glimpse_digit,
            "glimpse_target_match": example.glimpse_target_match,
            "previous_found_target": example.previous_found_target,
            "found_target": example.found_target,
            "relevant_region_inspected": example.relevant_region_inspected,
            "unresolved_search": example.unresolved_search,
            "current_wrong_candidate": example.current_wrong_candidate,
            "wrong_candidate_history": example.wrong_candidate_history,
            "revisit_unresolved": example.revisit_unresolved,
            "allocation_error": example.allocation_error,
            "inspected_count": example.inspected_count,
            "previous_inspected_count": example.previous_inspected_count,
            "attended_cell_previously_inspected": example.attended_cell_previously_inspected,
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
    request_retries: int = 8,
    retry_backoff_seconds: float = 2.0,
    teaching_examples: list[NLExample] | None = None,
) -> dict[str, Any]:
    """Query an OpenAI model for one reporting mode and score structured faithfulness."""

    if OpenAI is None:
        raise RuntimeError("openai dependency is not installed")

    client = OpenAI(max_retries=2, timeout=90.0)
    results = []
    exact_search = 0
    exact_previous_search = 0
    exact_cue_switched = 0
    exact_attended = 0
    exact_visible_type = 0
    exact_attended_digit = 0
    exact_glimpse_digit = 0
    exact_prev_attended = 0
    exact_prev_visible_type = 0
    exact_prev_attended_digit = 0
    exact_prev_glimpse_digit = 0
    exact_glimpse_match = 0
    exact_previous_found = 0
    exact_found = 0
    exact_relevant_region = 0
    exact_unresolved_search = 0
    exact_current_wrong_candidate = 0
    exact_wrong_candidate_history = 0
    exact_revisit_unresolved = 0
    exact_allocation_error = 0
    exact_inspected_count = 0
    exact_previous_inspected_count = 0
    exact_attended_previously_inspected = 0
    exact_unresolved_rows = 0
    exact_unresolved_cols = 0
    exact_unresolved_count = 0
    for example in evaluation_examples:
        parsed = None
        last_error = None
        for attempt in range(max(request_retries, 1)):
            try:
                response = client.responses.create(
                    model=model_name,
                    input=_make_messages(
                        mode,
                        calibration_examples,
                        example,
                        grid_size,
                        teaching_examples=teaching_examples,
                    ),
                    max_output_tokens=max_output_tokens,
                    reasoning={"effort": "low"},
                    text={
                        "verbosity": "low",
                        "format": {"type": "json_schema", **_schema()},
                    },
                )
                parsed = _extract_response_json(response)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(request_retries, 1):
                    time.sleep(retry_backoff_seconds * (attempt + 1))
        if parsed is None:
            raise RuntimeError(str(last_error) if last_error is not None else "nl_report parsing failed")
        expected_attended = list(divmod(example.attended_cell, grid_size))
        expected_prev_attended = list(divmod(example.prev_attended_cell, grid_size))

        exact_search += int(parsed["search_type"] == example.cue)
        exact_previous_search += int(parsed["previous_search_type"] == example.previous_cue)
        exact_cue_switched += int(bool(parsed["cue_switched"]) == example.cue_switched)
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
        exact_previous_found += int(bool(parsed["previous_found_target"]) == example.previous_found_target)
        exact_found += int(bool(parsed["found_target"]) == example.found_target)
        exact_relevant_region += int(
            bool(parsed["relevant_region_inspected"]) == example.relevant_region_inspected
        )
        exact_unresolved_search += int(bool(parsed["unresolved_search"]) == example.unresolved_search)
        exact_current_wrong_candidate += int(
            bool(parsed["current_wrong_candidate"]) == example.current_wrong_candidate
        )
        exact_wrong_candidate_history += int(
            bool(parsed["wrong_candidate_history"]) == example.wrong_candidate_history
        )
        exact_revisit_unresolved += int(
            bool(parsed["revisit_unresolved"]) == example.revisit_unresolved
        )
        exact_allocation_error += int(bool(parsed["allocation_error"]) == example.allocation_error)
        exact_inspected_count += int(parsed["inspected_count"] == example.inspected_count)
        exact_previous_inspected_count += int(
            parsed["previous_inspected_count"] == example.previous_inspected_count
        )
        exact_attended_previously_inspected += int(
            bool(parsed["attended_cell_previously_inspected"])
            == example.attended_cell_previously_inspected
        )
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
                    "previous_search_type": example.previous_cue,
                    "cue_switched": example.cue_switched,
                    "attended_cell": expected_attended,
                    "attended_visible_type": example.attended_visible_type,
                    "attended_digit": example.attended_digit,
                    "glimpse_digit": example.glimpse_digit,
                    "previous_attended_cell": expected_prev_attended,
                    "previous_attended_visible_type": example.prev_attended_visible_type,
                    "previous_attended_digit": example.prev_attended_digit,
                    "previous_glimpse_digit": example.prev_glimpse_digit,
                    "glimpse_target_match": example.glimpse_target_match,
                    "previous_found_target": example.previous_found_target,
                    "found_target": example.found_target,
                    "relevant_region_inspected": example.relevant_region_inspected,
                    "unresolved_search": example.unresolved_search,
                    "current_wrong_candidate": example.current_wrong_candidate,
                    "wrong_candidate_history": example.wrong_candidate_history,
                    "revisit_unresolved": example.revisit_unresolved,
                    "allocation_error": example.allocation_error,
                    "inspected_count": example.inspected_count,
                    "previous_inspected_count": example.previous_inspected_count,
                    "attended_cell_previously_inspected": example.attended_cell_previously_inspected,
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
        "previous_search_type_accuracy": exact_previous_search / denom,
        "cue_switched_accuracy": exact_cue_switched / denom,
        "attended_cell_accuracy": exact_attended / denom,
        "attended_visible_type_accuracy": exact_visible_type / denom,
        "attended_digit_accuracy": exact_attended_digit / denom,
        "glimpse_digit_accuracy": exact_glimpse_digit / denom,
        "previous_attended_cell_accuracy": exact_prev_attended / denom,
        "previous_attended_visible_type_accuracy": exact_prev_visible_type / denom,
        "previous_attended_digit_accuracy": exact_prev_attended_digit / denom,
        "previous_glimpse_digit_accuracy": exact_prev_glimpse_digit / denom,
        "glimpse_target_match_accuracy": exact_glimpse_match / denom,
        "previous_found_target_accuracy": exact_previous_found / denom,
        "found_target_accuracy": exact_found / denom,
        "relevant_region_inspected_accuracy": exact_relevant_region / denom,
        "unresolved_search_accuracy": exact_unresolved_search / denom,
        "current_wrong_candidate_accuracy": exact_current_wrong_candidate / denom,
        "wrong_candidate_history_accuracy": exact_wrong_candidate_history / denom,
        "revisit_unresolved_accuracy": exact_revisit_unresolved / denom,
        "allocation_error_accuracy": exact_allocation_error / denom,
        "inspected_count_accuracy": exact_inspected_count / denom,
        "previous_inspected_count_accuracy": exact_previous_inspected_count / denom,
        "attended_cell_previously_inspected_accuracy": exact_attended_previously_inspected / denom,
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
                    item["response"]["previous_search_type"] == item["expected"]["previous_search_type"]
                    and bool(item["response"]["cue_switched"]) == item["expected"]["cue_switched"]
                    and bool(item["response"]["previous_found_target"])
                    == item["expected"]["previous_found_target"]
                    and item["response"]["inspected_count"] == item["expected"]["inspected_count"]
                    and item["response"]["previous_inspected_count"]
                    == item["expected"]["previous_inspected_count"]
                    and bool(item["response"]["attended_cell_previously_inspected"])
                    == item["expected"]["attended_cell_previously_inspected"]
                    and
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
                    and bool(item["response"]["relevant_region_inspected"])
                    == item["expected"]["relevant_region_inspected"]
                    and bool(item["response"]["unresolved_search"])
                    == item["expected"]["unresolved_search"]
                    and bool(item["response"]["current_wrong_candidate"])
                    == item["expected"]["current_wrong_candidate"]
                    and bool(item["response"]["wrong_candidate_history"])
                    == item["expected"]["wrong_candidate_history"]
                    and bool(item["response"]["revisit_unresolved"])
                    == item["expected"]["revisit_unresolved"]
                    and bool(item["response"]["allocation_error"])
                    == item["expected"]["allocation_error"]
                )
                for item in results
            )
            / denom
        ),
        "uncertainty_content_joint_accuracy": (
            sum(
                int(
                    bool(item["response"]["relevant_region_inspected"])
                    == item["expected"]["relevant_region_inspected"]
                    and bool(item["response"]["unresolved_search"])
                    == item["expected"]["unresolved_search"]
                    and bool(item["response"]["current_wrong_candidate"])
                    == item["expected"]["current_wrong_candidate"]
                    and bool(item["response"]["wrong_candidate_history"])
                    == item["expected"]["wrong_candidate_history"]
                    and bool(item["response"]["revisit_unresolved"])
                    == item["expected"]["revisit_unresolved"]
                    and bool(item["response"]["allocation_error"])
                    == item["expected"]["allocation_error"]
                )
                for item in results
            )
            / denom
        ),
        "joint_accuracy": (
            sum(
                int(
                    item["response"]["search_type"] == item["expected"]["search_type"]
                    and item["response"]["previous_search_type"] == item["expected"]["previous_search_type"]
                    and bool(item["response"]["cue_switched"]) == item["expected"]["cue_switched"]
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
                    and bool(item["response"]["previous_found_target"])
                    == item["expected"]["previous_found_target"]
                    and bool(item["response"]["found_target"]) == item["expected"]["found_target"]
                    and bool(item["response"]["relevant_region_inspected"])
                    == item["expected"]["relevant_region_inspected"]
                    and bool(item["response"]["unresolved_search"])
                    == item["expected"]["unresolved_search"]
                    and bool(item["response"]["current_wrong_candidate"])
                    == item["expected"]["current_wrong_candidate"]
                    and bool(item["response"]["wrong_candidate_history"])
                    == item["expected"]["wrong_candidate_history"]
                    and bool(item["response"]["revisit_unresolved"])
                    == item["expected"]["revisit_unresolved"]
                    and bool(item["response"]["allocation_error"])
                    == item["expected"]["allocation_error"]
                    and item["response"]["inspected_count"] == item["expected"]["inspected_count"]
                    and item["response"]["previous_inspected_count"]
                    == item["expected"]["previous_inspected_count"]
                    and bool(item["response"]["attended_cell_previously_inspected"])
                    == item["expected"]["attended_cell_previously_inspected"]
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
