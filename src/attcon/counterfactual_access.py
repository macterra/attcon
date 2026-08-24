from __future__ import annotations

"""Branch D benchmark cases for access beyond current attention."""

from dataclasses import asdict, dataclass
import random
from typing import Any, Literal


AccessStatus = Literal[
    "unavailable",
    "merely_visible",
    "previously_attended",
    "counterfactually_accessible",
    "current_anchor",
    "distractor",
]

UNKNOWN_ANSWER = -1
TARGET_STATUSES = (
    "unavailable",
    "merely_visible",
    "previously_attended",
    "counterfactually_accessible",
)


@dataclass(frozen=True)
class CounterfactualAccessConfig:
    num_items: int = 8
    key_vocab_size: int = 16
    value_vocab_size: int = 10
    heldout_modulus: int = 5


@dataclass(frozen=True)
class AccessItem:
    slot: int
    key: int
    initial_value: int
    current_observation_value: int | None
    access_cache_value: int | None
    attended_before: bool
    status: AccessStatus


@dataclass(frozen=True)
class CounterfactualAccessExample:
    example_id: str
    items: tuple[AccessItem, ...]
    target_index: int
    target_status: str
    initial_query_key: int
    switched_query_key: int
    current_attention_before: int
    current_attention_after: int
    expected_answer: int
    scene_only_answer: int
    current_glimpse_answer: int
    split: str

    @property
    def target(self) -> AccessItem:
        return self.items[self.target_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "items": [asdict(item) for item in self.items],
            "target": asdict(self.target),
        }


def _heldout_query_pair(key: int, value: int, modulus: int) -> bool:
    if modulus < 2:
        raise ValueError("heldout_modulus must be >= 2")
    return (key * 73856093 + value * 19349663) % modulus == 0


def generate_counterfactual_access_examples(
    count: int,
    *,
    config: CounterfactualAccessConfig | None = None,
    seed: int = 11,
) -> list[CounterfactualAccessExample]:
    """Generate query switches whose current fixation is explicitly held constant."""

    if count < 1:
        raise ValueError("count must be >= 1")
    cfg = config or CounterfactualAccessConfig()
    if cfg.num_items < 5:
        raise ValueError("num_items must be >= 5")
    if cfg.key_vocab_size < cfg.num_items:
        raise ValueError("key_vocab_size must cover unique keys in every episode")
    if cfg.value_vocab_size < 2:
        raise ValueError("value_vocab_size must be >= 2")
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        keys = rng.sample(range(cfg.key_vocab_size), cfg.num_items)
        values = [rng.randrange(cfg.value_vocab_size) for _ in range(cfg.num_items)]
        statuses: list[AccessStatus] = [
            "unavailable",
            "merely_visible",
            "previously_attended",
            "counterfactually_accessible",
            "current_anchor",
        ] + ["distractor"] * (cfg.num_items - 5)
        rng.shuffle(statuses)
        items = []
        for slot, (key, value, status) in enumerate(zip(keys, values, statuses)):
            if status == "unavailable":
                current_value = None
                cache_value = None
                attended = False
            elif status == "merely_visible":
                current_value = value
                cache_value = None
                attended = False
            elif status == "previously_attended":
                current_value = None
                cache_value = value
                attended = True
            elif status == "counterfactually_accessible":
                # The visible alternative conflicts with the task-access cache by construction.
                current_value = (value + rng.randrange(1, cfg.value_vocab_size)) % cfg.value_vocab_size
                cache_value = value
                attended = False
            else:
                current_value = value
                cache_value = value if status == "current_anchor" else None
                attended = status == "current_anchor"
            items.append(
                AccessItem(
                    slot=slot,
                    key=key,
                    initial_value=value,
                    current_observation_value=current_value,
                    access_cache_value=cache_value,
                    attended_before=attended,
                    status=status,
                )
            )
        item_tuple = tuple(items)
        target_status = TARGET_STATUSES[index % len(TARGET_STATUSES)]
        target_index = next(
            item.slot for item in item_tuple if item.status == target_status
        )
        anchor_index = next(
            item.slot for item in item_tuple if item.status == "current_anchor"
        )
        target = item_tuple[target_index]
        anchor = item_tuple[anchor_index]
        expected = (
            UNKNOWN_ANSWER if target_status == "unavailable" else target.initial_value
        )
        scene_answer = (
            UNKNOWN_ANSWER
            if target.current_observation_value is None
            else target.current_observation_value
        )
        split = (
            "heldout_query_value"
            if _heldout_query_pair(target.key, expected, cfg.heldout_modulus)
            else "train"
        )
        examples.append(
            CounterfactualAccessExample(
                example_id=f"counterfactual_access_{seed}_{index}",
                items=item_tuple,
                target_index=target_index,
                target_status=target_status,
                initial_query_key=anchor.key,
                switched_query_key=target.key,
                current_attention_before=anchor_index,
                current_attention_after=anchor_index,
                expected_answer=expected,
                scene_only_answer=scene_answer,
                current_glimpse_answer=anchor.current_observation_value,
                split=split,
            )
        )
    return examples


def validate_counterfactual_access_example(
    example: CounterfactualAccessExample,
) -> list[str]:
    failures = []
    target = example.target
    if len({item.key for item in example.items}) != len(example.items):
        failures.append("duplicate_query_key")
    if example.initial_query_key == example.switched_query_key:
        failures.append("query_did_not_switch")
    if example.current_attention_before != example.current_attention_after:
        failures.append("attention_not_held_fixed")
    if example.current_attention_before == example.target_index:
        failures.append("switched_target_is_currently_attended")
    if target.key != example.switched_query_key:
        failures.append("switched_query_does_not_select_target")
    if target.status != example.target_status:
        failures.append("target_status_mismatch")
    if example.target_status == "unavailable":
        if target.current_observation_value is not None or target.access_cache_value is not None:
            failures.append("unavailable_content_leaked")
        if example.expected_answer != UNKNOWN_ANSWER:
            failures.append("unavailable_answer_not_unknown")
    elif example.target_status == "merely_visible":
        if target.current_observation_value != target.initial_value:
            failures.append("merely_visible_not_scene_recoverable")
        if target.attended_before or target.access_cache_value is not None:
            failures.append("merely_visible_has_memory_access")
    elif example.target_status == "previously_attended":
        if not target.attended_before or target.access_cache_value != target.initial_value:
            failures.append("previously_attended_not_cached")
        if target.current_observation_value is not None:
            failures.append("previously_attended_still_visible")
    elif example.target_status == "counterfactually_accessible":
        if target.attended_before or target.access_cache_value != target.initial_value:
            failures.append("counterfactual_access_signature_invalid")
        if target.current_observation_value in (None, target.initial_value):
            failures.append("counterfactual_tension_missing")
        if example.scene_only_answer == example.expected_answer:
            failures.append("counterfactual_scene_answer_not_in_tension")
    else:
        failures.append("invalid_target_status")
    return failures
