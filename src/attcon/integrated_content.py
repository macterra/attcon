from __future__ import annotations

"""Paired same-content cases for the Stage 8 convergence benchmark.

Each episode presents one initial object scene, then reuses the exact target identity and
feature bundle in four access-status variants.  This gives binding, access, and future
shared-state perturbation analyses a common unit of analysis instead of merely a common
checkpoint.
"""

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

TARGET_STATUSES = (
    "unavailable",
    "merely_visible",
    "previously_attended",
    "counterfactually_accessible",
)
UNKNOWN_ANSWER = -1


@dataclass(frozen=True)
class IntegratedContentConfig:
    grid_size: int = 5
    num_objects: int = 7
    content_id_vocab_size: int = 16
    feature_type_vocab_size: int = 5
    value_vocab_size: int = 10
    heldout_modulus: int = 5

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size


@dataclass(frozen=True)
class ContentObject:
    content_id: int
    location: int
    feature_type: int
    initial_value: int
    current_observation_value: int | None
    access_cache_value: int | None
    attended_before: bool
    status: AccessStatus

    def initial_bundle(self) -> tuple[int, int, int]:
        return (self.location, self.feature_type, self.initial_value)


@dataclass(frozen=True)
class ContentBundle:
    location: int
    feature_type: int
    value: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.location, self.feature_type, self.value)


@dataclass(frozen=True)
class IntegratedContentExample:
    example_id: str
    pair_group_id: str
    objects: tuple[ContentObject, ...]
    target_index: int
    target_status: str
    binding_cue_content_id: int
    initial_query_content_id: int
    switched_query_content_id: int
    current_attention_before: int
    current_attention_after: int
    false_binding_lure: ContentBundle
    expected_access_answer: int
    split: str
    shared_state_perturbation_step: int = 1

    @property
    def target(self) -> ContentObject:
        return self.objects[self.target_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "objects": [asdict(obj) for obj in self.objects],
            "target": asdict(self.target),
            "content_identity": {
                "pair_group_id": self.pair_group_id,
                "content_id": self.target.content_id,
                "initial_bundle": self.target.initial_bundle(),
            },
        }


def _heldout_bundle(
    content_id: int,
    bundle: tuple[int, int, int],
    modulus: int,
) -> bool:
    if modulus < 2:
        raise ValueError("heldout_modulus must be >= 2")
    location, feature_type, value = bundle
    mixed = (
        content_id * 73856093
        + location * 19349663
        + feature_type * 83492791
        + value * 2654435761
    )
    return mixed % modulus == 0


def _false_binding_lure(
    bundles: tuple[tuple[int, int, int], ...],
    rng: random.Random,
) -> ContentBundle:
    present = set(bundles)
    for _ in range(256):
        lure = ContentBundle(
            location=rng.choice(bundles)[0],
            feature_type=rng.choice(bundles)[1],
            value=rng.choice(bundles)[2],
        )
        if lure.as_tuple() not in present:
            return lure
    raise RuntimeError("failed to construct an absent false-binding lure")


def generate_integrated_content_examples(
    episode_count: int,
    *,
    config: IntegratedContentConfig | None = None,
    seed: int = 811,
) -> list[IntegratedContentExample]:
    """Generate four paired access transitions for every initial target content."""

    if episode_count < 1:
        raise ValueError("episode_count must be >= 1")
    cfg = config or IntegratedContentConfig()
    if not (3 <= cfg.num_objects <= cfg.num_cells):
        raise ValueError("num_objects must be between 3 and the number of cells")
    if cfg.content_id_vocab_size < cfg.num_objects:
        raise ValueError("content_id vocabulary must cover unique episode objects")
    if cfg.feature_type_vocab_size < 2 or cfg.value_vocab_size < 2:
        raise ValueError("feature and value vocabularies must each contain at least two values")

    rng = random.Random(seed)
    examples: list[IntegratedContentExample] = []
    for episode_index in range(episode_count):
        group_id = f"integrated_{seed}_{episode_index}"
        content_ids = rng.sample(range(cfg.content_id_vocab_size), cfg.num_objects)
        locations = rng.sample(range(cfg.num_cells), cfg.num_objects)
        feature_types = [rng.randrange(cfg.feature_type_vocab_size) for _ in locations]
        values = [rng.randrange(cfg.value_vocab_size) for _ in locations]
        target_index = rng.randrange(cfg.num_objects)
        anchor_index = rng.choice(
            [index for index in range(cfg.num_objects) if index != target_index]
        )
        bundles = tuple(zip(locations, feature_types, values))
        lure = _false_binding_lure(bundles, rng)
        target_bundle = bundles[target_index]
        split = (
            "heldout_content_bundle"
            if _heldout_bundle(
                content_ids[target_index], target_bundle, cfg.heldout_modulus
            )
            else "train"
        )

        for status in TARGET_STATUSES:
            objects = []
            for index, (content_id, location, feature_type, value) in enumerate(
                zip(content_ids, locations, feature_types, values)
            ):
                if index == target_index:
                    if status == "unavailable":
                        current_value, cache_value, attended = None, None, False
                    elif status == "merely_visible":
                        current_value, cache_value, attended = value, None, False
                    elif status == "previously_attended":
                        current_value, cache_value, attended = None, value, True
                    else:
                        current_value = (value + rng.randrange(1, cfg.value_vocab_size)) % cfg.value_vocab_size
                        cache_value, attended = value, False
                    object_status: AccessStatus = status
                elif index == anchor_index:
                    current_value, cache_value, attended = value, value, True
                    object_status = "current_anchor"
                else:
                    current_value, cache_value, attended = value, None, False
                    object_status = "distractor"
                objects.append(
                    ContentObject(
                        content_id=content_id,
                        location=location,
                        feature_type=feature_type,
                        initial_value=value,
                        current_observation_value=current_value,
                        access_cache_value=cache_value,
                        attended_before=attended,
                        status=object_status,
                    )
                )
            expected = UNKNOWN_ANSWER if status == "unavailable" else values[target_index]
            examples.append(
                IntegratedContentExample(
                    example_id=f"{group_id}_{status}",
                    pair_group_id=group_id,
                    objects=tuple(objects),
                    target_index=target_index,
                    target_status=status,
                    binding_cue_content_id=content_ids[target_index],
                    initial_query_content_id=content_ids[anchor_index],
                    switched_query_content_id=content_ids[target_index],
                    current_attention_before=anchor_index,
                    current_attention_after=anchor_index,
                    false_binding_lure=lure,
                    expected_access_answer=expected,
                    split=split,
                )
            )
    return examples


def validate_integrated_content_example(
    example: IntegratedContentExample,
) -> list[str]:
    failures: list[str] = []
    target = example.target
    bundles = {obj.initial_bundle() for obj in example.objects}
    if len({obj.content_id for obj in example.objects}) != len(example.objects):
        failures.append("duplicate_content_id")
    if len({obj.location for obj in example.objects}) != len(example.objects):
        failures.append("duplicate_location")
    if example.binding_cue_content_id != target.content_id:
        failures.append("binding_cue_target_mismatch")
    if example.switched_query_content_id != target.content_id:
        failures.append("access_query_target_mismatch")
    if example.current_attention_before != example.current_attention_after:
        failures.append("attention_not_held_fixed")
    if example.current_attention_after == example.target_index:
        failures.append("queried_target_is_currently_attended")
    if example.false_binding_lure.as_tuple() in bundles:
        failures.append("false_binding_lure_present")
    for field_index, field_name in enumerate(("location", "feature_type", "value")):
        if example.false_binding_lure.as_tuple()[field_index] not in {
            bundle[field_index] for bundle in bundles
        }:
            failures.append(f"lure_{field_name}_not_individually_present")
    if target.status != example.target_status:
        failures.append("target_status_mismatch")
    if example.target_status == "unavailable":
        if target.current_observation_value is not None or target.access_cache_value is not None:
            failures.append("unavailable_content_leaked")
        if example.expected_access_answer != UNKNOWN_ANSWER:
            failures.append("unavailable_answer_not_unknown")
    elif example.target_status == "merely_visible":
        if target.current_observation_value != target.initial_value or target.access_cache_value is not None:
            failures.append("merely_visible_signature_invalid")
    elif example.target_status == "previously_attended":
        if target.current_observation_value is not None or target.access_cache_value != target.initial_value:
            failures.append("previously_attended_signature_invalid")
    elif example.target_status == "counterfactually_accessible":
        if target.current_observation_value in (None, target.initial_value):
            failures.append("counterfactual_observation_tension_missing")
        if target.access_cache_value != target.initial_value:
            failures.append("counterfactual_cache_invalid")
    else:
        failures.append("invalid_target_status")
    return failures


def validate_paired_content_group(
    examples: list[IntegratedContentExample],
) -> list[str]:
    failures: list[str] = []
    if len(examples) != len(TARGET_STATUSES):
        failures.append("group_does_not_have_four_variants")
        return failures
    if {example.target_status for example in examples} != set(TARGET_STATUSES):
        failures.append("group_status_coverage_incomplete")
    reference = examples[0]
    signature = (
        tuple(obj.initial_bundle() for obj in reference.objects),
        tuple(obj.content_id for obj in reference.objects),
        reference.target_index,
        reference.binding_cue_content_id,
        reference.switched_query_content_id,
        reference.false_binding_lure,
        reference.split,
    )
    for example in examples[1:]:
        candidate = (
            tuple(obj.initial_bundle() for obj in example.objects),
            tuple(obj.content_id for obj in example.objects),
            example.target_index,
            example.binding_cue_content_id,
            example.switched_query_content_id,
            example.false_binding_lure,
            example.split,
        )
        if candidate != signature:
            failures.append("content_identity_changed_across_status_variants")
            break
    return failures
