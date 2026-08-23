from __future__ import annotations

"""Temporal event-stream scaffold for genuinely different Stage 8 replication."""

from dataclasses import asdict, dataclass
import random
from typing import Any, Literal


RelayStatus = Literal[
    "missing",
    "live_buffer",
    "archived",
    "conflict_recoverable",
]
RELAY_STATUSES = ("missing", "live_buffer", "archived", "conflict_recoverable")
UNKNOWN_PAYLOAD = -1


@dataclass(frozen=True)
class TemporalRelayConfig:
    stream_length: int = 10
    entity_vocab_size: int = 12
    operation_vocab_size: int = 4
    payload_vocab_size: int = 10
    heldout_modulus: int = 5


@dataclass(frozen=True)
class RelayEvent:
    event_id: int
    time_index: int
    entity: int
    operation: int
    payload: int

    def content_bundle(self) -> tuple[int, int, int]:
        return (self.time_index, self.operation, self.payload)


@dataclass(frozen=True)
class RelayLure:
    time_index: int
    operation: int
    payload: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.time_index, self.operation, self.payload)


@dataclass(frozen=True)
class TemporalRelayExample:
    example_id: str
    pair_group_id: str
    events: tuple[RelayEvent, ...]
    query_entity: int
    target_event_index: int
    target_event_id: int
    target_status: str
    current_attention_before: int
    current_attention_after: int
    live_buffer_payload: int | None
    archive_payload: int | None
    expected_payload: int
    false_chronology_lure: RelayLure
    split: str

    @property
    def target(self) -> RelayEvent:
        return self.events[self.target_event_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "events": [asdict(event) for event in self.events],
            "target": asdict(self.target),
            "content_identity": {
                "pair_group_id": self.pair_group_id,
                "event_id": self.target_event_id,
                "bundle": self.target.content_bundle(),
            },
        }


def _heldout(event: RelayEvent, modulus: int) -> bool:
    if modulus < 2:
        raise ValueError("heldout_modulus must be >= 2")
    mixed = (
        event.entity * 73856093
        + event.time_index * 19349663
        + event.operation * 83492791
        + event.payload * 2654435761
    )
    return mixed % modulus == 0


def _lure(events: tuple[RelayEvent, ...], rng: random.Random) -> RelayLure:
    present = {event.content_bundle() for event in events}
    for _ in range(256):
        lure = RelayLure(
            time_index=rng.choice(events).time_index,
            operation=rng.choice(events).operation,
            payload=rng.choice(events).payload,
        )
        if lure.as_tuple() not in present:
            return lure
    raise RuntimeError("failed to construct absent chronology lure")


def generate_temporal_relay_examples(
    episode_count: int,
    *,
    config: TemporalRelayConfig | None = None,
    seed: int = 1103,
) -> list[TemporalRelayExample]:
    """Create paired delayed-access variants of ordered last-write queries."""

    if episode_count < 1:
        raise ValueError("episode_count must be >= 1")
    cfg = config or TemporalRelayConfig()
    if cfg.stream_length < 4:
        raise ValueError("stream_length must be >= 4")
    if cfg.entity_vocab_size < 3:
        raise ValueError("entity_vocab_size must be >= 3")
    if cfg.operation_vocab_size < 2 or cfg.payload_vocab_size < 2:
        raise ValueError("operation and payload vocabularies must contain at least two values")

    rng = random.Random(seed)
    examples: list[TemporalRelayExample] = []
    for episode_index in range(episode_count):
        group_id = f"temporal_relay_{seed}_{episode_index}"
        query_entity = rng.randrange(cfg.entity_vocab_size)
        first_query_time, target_time = sorted(
            rng.sample(range(cfg.stream_length), 2)
        )
        entities = []
        alternatives = [
            entity for entity in range(cfg.entity_vocab_size)
            if entity != query_entity
        ]
        for time_index in range(cfg.stream_length):
            if time_index in (first_query_time, target_time):
                entities.append(query_entity)
            else:
                entities.append(rng.choice(alternatives))
        events = tuple(
            RelayEvent(
                event_id=episode_index * cfg.stream_length + time_index,
                time_index=time_index,
                entity=entity,
                operation=rng.randrange(cfg.operation_vocab_size),
                payload=rng.randrange(cfg.payload_vocab_size),
            )
            for time_index, entity in enumerate(entities)
        )
        target_index = max(
            index for index, event in enumerate(events)
            if event.entity == query_entity
        )
        target = events[target_index]
        anchor_candidates = [
            index for index, event in enumerate(events)
            if index != target_index and event.entity != query_entity
        ]
        anchor_index = max(anchor_candidates)
        lure = _lure(events, rng)
        split = "heldout_event_bundle" if _heldout(target, cfg.heldout_modulus) else "train"
        for status in RELAY_STATUSES:
            if status == "missing":
                live, archive, expected = None, None, UNKNOWN_PAYLOAD
            elif status == "live_buffer":
                live, archive, expected = target.payload, None, target.payload
            elif status == "archived":
                live, archive, expected = None, target.payload, target.payload
            else:
                live = (target.payload + rng.randrange(1, cfg.payload_vocab_size)) % cfg.payload_vocab_size
                archive, expected = target.payload, target.payload
            examples.append(
                TemporalRelayExample(
                    example_id=f"{group_id}_{status}",
                    pair_group_id=group_id,
                    events=events,
                    query_entity=query_entity,
                    target_event_index=target_index,
                    target_event_id=target.event_id,
                    target_status=status,
                    current_attention_before=anchor_index,
                    current_attention_after=anchor_index,
                    live_buffer_payload=live,
                    archive_payload=archive,
                    expected_payload=expected,
                    false_chronology_lure=lure,
                    split=split,
                )
            )
    return examples


def validate_temporal_relay_example(example: TemporalRelayExample) -> list[str]:
    failures: list[str] = []
    target = example.target
    query_events = [
        event for event in example.events if event.entity == example.query_entity
    ]
    if len(query_events) < 2:
        failures.append("query_entity_not_updated")
    if target != query_events[-1]:
        failures.append("target_is_not_last_query_entity_event")
    if target.event_id != example.target_event_id:
        failures.append("target_identity_mismatch")
    if example.current_attention_before != example.current_attention_after:
        failures.append("attention_not_held_fixed")
    if example.current_attention_after == example.target_event_index:
        failures.append("target_is_currently_attended")
    bundles = {event.content_bundle() for event in example.events}
    lure = example.false_chronology_lure.as_tuple()
    if lure in bundles:
        failures.append("chronology_lure_present")
    for index, name in enumerate(("time", "operation", "payload")):
        if lure[index] not in {bundle[index] for bundle in bundles}:
            failures.append(f"lure_{name}_not_individually_present")
    if example.target_status == "missing":
        if example.live_buffer_payload is not None or example.archive_payload is not None:
            failures.append("missing_payload_leaked")
        if example.expected_payload != UNKNOWN_PAYLOAD:
            failures.append("missing_answer_not_unknown")
    elif example.target_status == "live_buffer":
        if example.live_buffer_payload != target.payload or example.archive_payload is not None:
            failures.append("live_buffer_signature_invalid")
    elif example.target_status == "archived":
        if example.live_buffer_payload is not None or example.archive_payload != target.payload:
            failures.append("archive_signature_invalid")
    elif example.target_status == "conflict_recoverable":
        if example.live_buffer_payload in (None, target.payload):
            failures.append("conflict_missing")
        if example.archive_payload != target.payload:
            failures.append("conflict_archive_invalid")
    else:
        failures.append("invalid_status")
    return failures


def validate_temporal_relay_group(
    examples: list[TemporalRelayExample],
) -> list[str]:
    if len(examples) != len(RELAY_STATUSES):
        return ["group_does_not_have_four_variants"]
    failures = []
    if {example.target_status for example in examples} != set(RELAY_STATUSES):
        failures.append("status_coverage_incomplete")
    reference = examples[0]
    signature = (
        reference.events,
        reference.query_entity,
        reference.target_event_id,
        reference.false_chronology_lure,
        reference.split,
    )
    if any(
        (
            example.events,
            example.query_entity,
            example.target_event_id,
            example.false_chronology_lure,
            example.split,
        )
        != signature
        for example in examples[1:]
    ):
        failures.append("content_identity_changed_across_statuses")
    return failures
