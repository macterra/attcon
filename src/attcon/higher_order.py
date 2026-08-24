from __future__ import annotations

"""Branch E benchmark cases separating content from higher-order access state."""

from dataclasses import asdict, dataclass
import random
from typing import Any, Literal


HigherOrderStatus = Literal[
    "fresh_current",
    "fresh_memory",
    "inferred_content",
    "unavailable",
    "stale_access_lure",
    "wrong_access_lure",
]

HIGHER_ORDER_STATUSES = (
    "fresh_current",
    "fresh_memory",
    "inferred_content",
    "unavailable",
    "stale_access_lure",
    "wrong_access_lure",
)


@dataclass(frozen=True)
class HigherOrderConfig:
    key_vocab_size: int = 16
    value_vocab_size: int = 10
    heldout_modulus: int = 5


@dataclass(frozen=True)
class HigherOrderExample:
    example_id: str
    counterbalance_group: str
    content_key: int
    content_value: int
    status: HigherOrderStatus
    current_observation_value: int | None
    memory_value: int | None
    inferred_value: int | None
    access_gate_open: bool
    access_available: bool
    report_source: str
    confidence_band: int
    should_reinspect: bool
    should_correct: bool
    lure_type: str | None
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _heldout_content_status(
    key: int,
    value: int,
    status_index: int,
    modulus: int,
) -> bool:
    if modulus < 2:
        raise ValueError("heldout_modulus must be >= 2")
    mixed = key * 73856093 + value * 19349663 + status_index * 83492791
    return mixed % modulus == 0


def _status_fields(
    status: HigherOrderStatus,
    value: int,
    value_vocab_size: int,
    rng: random.Random,
) -> dict[str, Any]:
    decoy = (value + rng.randrange(1, value_vocab_size)) % value_vocab_size
    if status == "fresh_current":
        return {
            "current_observation_value": value,
            "memory_value": None,
            "inferred_value": None,
            "access_gate_open": True,
            "access_available": True,
            "report_source": "current",
            "confidence_band": 3,
            "should_reinspect": False,
            "should_correct": False,
            "lure_type": None,
        }
    if status == "fresh_memory":
        return {
            "current_observation_value": None,
            "memory_value": value,
            "inferred_value": None,
            "access_gate_open": True,
            "access_available": True,
            "report_source": "memory",
            "confidence_band": 2,
            "should_reinspect": False,
            "should_correct": False,
            "lure_type": None,
        }
    if status == "inferred_content":
        return {
            "current_observation_value": None,
            "memory_value": None,
            "inferred_value": value,
            "access_gate_open": True,
            "access_available": True,
            "report_source": "inference",
            "confidence_band": 1,
            "should_reinspect": True,
            "should_correct": False,
            "lure_type": "inferred_content",
        }
    if status == "unavailable":
        return {
            "current_observation_value": None,
            "memory_value": None,
            "inferred_value": None,
            "access_gate_open": False,
            "access_available": False,
            "report_source": "none",
            "confidence_band": 0,
            "should_reinspect": True,
            "should_correct": False,
            "lure_type": None,
        }
    if status == "stale_access_lure":
        return {
            "current_observation_value": None,
            "memory_value": decoy,
            "inferred_value": None,
            "access_gate_open": False,
            "access_available": False,
            "report_source": "none",
            "confidence_band": 0,
            "should_reinspect": True,
            "should_correct": True,
            "lure_type": "stale_access",
        }
    if status == "wrong_access_lure":
        return {
            # First-order content is present exactly as in fresh_current, but its access
            # relation is independently closed.
            "current_observation_value": value,
            "memory_value": None,
            "inferred_value": None,
            "access_gate_open": False,
            "access_available": False,
            "report_source": "none",
            "confidence_band": 0,
            "should_reinspect": True,
            "should_correct": True,
            "lure_type": "wrong_access",
        }
    raise ValueError(f"unknown higher-order status: {status}")


def generate_higher_order_examples(
    count: int,
    *,
    config: HigherOrderConfig | None = None,
    seed: int = 13,
) -> list[HigherOrderExample]:
    """Generate complete six-way counterbalances for each first-order content pair."""

    if count < len(HIGHER_ORDER_STATUSES):
        raise ValueError("count must cover at least one complete counterbalance group")
    if count % len(HIGHER_ORDER_STATUSES):
        raise ValueError("count must be divisible by the six higher-order statuses")
    cfg = config or HigherOrderConfig()
    if cfg.key_vocab_size < 2 or cfg.value_vocab_size < 2:
        raise ValueError("key and value vocabularies must each contain at least two values")
    rng = random.Random(seed)
    examples = []
    group_count = count // len(HIGHER_ORDER_STATUSES)
    for group in range(group_count):
        key = rng.randrange(cfg.key_vocab_size)
        value = rng.randrange(cfg.value_vocab_size)
        group_id = f"higher_order_{seed}_group_{group}"
        for status_index, status in enumerate(HIGHER_ORDER_STATUSES):
            fields = _status_fields(status, value, cfg.value_vocab_size, rng)
            split = (
                "heldout_content_status"
                if _heldout_content_status(
                    key, value, status_index, cfg.heldout_modulus
                )
                else "train"
            )
            examples.append(
                HigherOrderExample(
                    example_id=f"{group_id}_{status}",
                    counterbalance_group=group_id,
                    content_key=key,
                    content_value=value,
                    status=status,
                    split=split,
                    **fields,
                )
            )
    return examples


def validate_higher_order_example(example: HigherOrderExample) -> list[str]:
    failures = []
    if example.status == "fresh_current":
        if example.current_observation_value != example.content_value:
            failures.append("fresh_current_content_mismatch")
        if not example.access_available or example.report_source != "current":
            failures.append("fresh_current_access_mismatch")
    elif example.status == "fresh_memory":
        if example.memory_value != example.content_value:
            failures.append("fresh_memory_content_mismatch")
        if not example.access_available or example.report_source != "memory":
            failures.append("fresh_memory_access_mismatch")
    elif example.status == "inferred_content":
        if example.inferred_value != example.content_value:
            failures.append("inferred_content_mismatch")
        if example.report_source != "inference" or not example.should_reinspect:
            failures.append("inferred_access_mismatch")
    elif example.status == "unavailable":
        if any(
            value is not None
            for value in (
                example.current_observation_value,
                example.memory_value,
                example.inferred_value,
            )
        ):
            failures.append("unavailable_content_leak")
        if example.access_available or example.report_source != "none":
            failures.append("unavailable_access_mismatch")
    elif example.status == "stale_access_lure":
        if example.memory_value in (None, example.content_value):
            failures.append("stale_lure_not_in_tension")
        if example.access_gate_open or not example.should_correct:
            failures.append("stale_lure_relation_mismatch")
    elif example.status == "wrong_access_lure":
        if example.current_observation_value != example.content_value:
            failures.append("wrong_access_first_order_content_missing")
        if example.access_gate_open or example.access_available:
            failures.append("wrong_access_gate_not_closed")
        if not example.should_correct:
            failures.append("wrong_access_correction_missing")
    else:
        failures.append("unknown_status")
    if not 0 <= example.confidence_band <= 3:
        failures.append("confidence_out_of_range")
    return failures


def validate_counterbalance_group(
    examples: list[HigherOrderExample],
) -> list[str]:
    failures = []
    if {example.status for example in examples} != set(HIGHER_ORDER_STATUSES):
        failures.append("incomplete_status_counterbalance")
    if len({(example.content_key, example.content_value) for example in examples}) != 1:
        failures.append("first_order_content_not_held_constant")
    current = next(
        (example for example in examples if example.status == "fresh_current"), None
    )
    wrong = next(
        (example for example in examples if example.status == "wrong_access_lure"), None
    )
    if (
        current is None
        or wrong is None
        or current.current_observation_value != wrong.current_observation_value
    ):
        failures.append("current_wrong_access_pair_not_first_order_matched")
    return failures
