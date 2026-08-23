from __future__ import annotations

"""Branch F benchmark cases for shared broadcast and ignition experiments."""

from dataclasses import asdict, dataclass
import random
from typing import Any


CONSUMERS = (
    "action",
    "structured_report",
    "uncertainty",
    "reallocation",
    "memory_write",
    "language_report",
)


@dataclass(frozen=True)
class BroadcastConfig:
    content_vocab_size: int = 12
    cue_strength_levels: int = 5
    evidence_levels: int = 4
    num_steps: int = 5
    heldout_modulus: int = 5


@dataclass(frozen=True)
class BroadcastExample:
    example_id: str
    sweep_group: str
    content: int
    cue_strength: int
    evidence_quality: int
    ignition_threshold: int
    ignited: bool
    ignition_step: int | None
    consumer_available: tuple[bool, ...]
    consumer_onset_step: tuple[int | None, ...]
    consumer_targets: tuple[int | None, ...]
    split: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["consumer_available"] = dict(zip(CONSUMERS, self.consumer_available))
        result["consumer_onset_step"] = dict(
            zip(CONSUMERS, self.consumer_onset_step)
        )
        result["consumer_targets"] = dict(zip(CONSUMERS, self.consumer_targets))
        return result


def _heldout_content_strength(
    content: int,
    cue_strength: int,
    evidence_quality: int,
    modulus: int,
) -> bool:
    if modulus < 2:
        raise ValueError("heldout_modulus must be >= 2")
    mixed = (
        content * 73856093
        + cue_strength * 19349663
        + evidence_quality * 83492791
    )
    return mixed % modulus == 0


def _consumer_targets(
    content: int,
    evidence_quality: int,
    available: tuple[bool, ...],
    config: BroadcastConfig,
) -> tuple[int | None, ...]:
    raw_targets = (
        content % 4,
        content,
        config.evidence_levels - 1 - evidence_quality,
        content % 6,
        content,
        (content * 5 + 3) % config.content_vocab_size,
    )
    return tuple(
        target if is_available else None
        for target, is_available in zip(raw_targets, available)
    )


def generate_broadcast_examples(
    count: int,
    *,
    config: BroadcastConfig | None = None,
    seed: int = 17,
) -> list[BroadcastExample]:
    """Generate cue-strength sweeps with local action below a shared ignition threshold."""

    cfg = config or BroadcastConfig()
    if count < cfg.cue_strength_levels:
        raise ValueError("count must cover at least one complete cue-strength sweep")
    if count % cfg.cue_strength_levels:
        raise ValueError("count must be divisible by cue_strength_levels")
    if cfg.cue_strength_levels < 4 or cfg.evidence_levels < 2:
        raise ValueError("benchmark requires >=4 cue and >=2 evidence levels")
    rng = random.Random(seed)
    examples = []
    group_count = count // cfg.cue_strength_levels
    for group in range(group_count):
        content = rng.randrange(cfg.content_vocab_size)
        evidence = rng.randrange(cfg.evidence_levels)
        # Evidence modulates the threshold, while every sweep contains both regimes.
        threshold = max(1, cfg.cue_strength_levels - 2 - evidence // 2)
        group_id = f"broadcast_{seed}_group_{group}"
        for cue_strength in range(cfg.cue_strength_levels):
            ignited = cue_strength >= threshold
            ignition_step = (
                max(0, cfg.num_steps - 1 - cue_strength) if ignited else None
            )
            # A specialized action path can solve locally below threshold. The other five
            # consumers become available together only after the shared event.
            available = (True,) + (ignited,) * (len(CONSUMERS) - 1)
            onset = (0,) + (ignition_step,) * (len(CONSUMERS) - 1)
            split = (
                "heldout_content_strength"
                if _heldout_content_strength(
                    content,
                    cue_strength,
                    evidence,
                    cfg.heldout_modulus,
                )
                else "train"
            )
            examples.append(
                BroadcastExample(
                    example_id=f"{group_id}_strength_{cue_strength}",
                    sweep_group=group_id,
                    content=content,
                    cue_strength=cue_strength,
                    evidence_quality=evidence,
                    ignition_threshold=threshold,
                    ignited=ignited,
                    ignition_step=ignition_step,
                    consumer_available=available,
                    consumer_onset_step=onset,
                    consumer_targets=_consumer_targets(
                        content, evidence, available, cfg
                    ),
                    split=split,
                )
            )
    return examples


def validate_broadcast_example(example: BroadcastExample) -> list[str]:
    failures = []
    if len(example.consumer_available) != len(CONSUMERS):
        failures.append("consumer_availability_width")
    if len(example.consumer_onset_step) != len(CONSUMERS):
        failures.append("consumer_onset_width")
    if len(example.consumer_targets) != len(CONSUMERS):
        failures.append("consumer_target_width")
    if not example.consumer_available[0]:
        failures.append("local_action_unavailable")
    broad_available = example.consumer_available[1:]
    if any(broad_available) != example.ignited or len(set(broad_available)) != 1:
        failures.append("broad_consumers_not_coordinated")
    broad_onsets = example.consumer_onset_step[1:]
    if example.ignited:
        if example.ignition_step is None or set(broad_onsets) != {example.ignition_step}:
            failures.append("ignition_onsets_not_aligned")
    elif any(onset is not None for onset in broad_onsets):
        failures.append("nonignited_consumer_has_onset")
    for available, target in zip(
        example.consumer_available, example.consumer_targets
    ):
        if available != (target is not None):
            failures.append("target_availability_mismatch")
            break
    if example.ignited != (example.cue_strength >= example.ignition_threshold):
        failures.append("ignition_threshold_mismatch")
    return failures


def validate_broadcast_sweep(examples: list[BroadcastExample]) -> list[str]:
    failures = []
    if len({example.content for example in examples}) != 1:
        failures.append("content_not_held_fixed")
    if len({example.evidence_quality for example in examples}) != 1:
        failures.append("evidence_not_held_fixed")
    strengths = {example.cue_strength for example in examples}
    if strengths != set(range(len(examples))):
        failures.append("incomplete_strength_sweep")
    if not any(example.ignited for example in examples):
        failures.append("sweep_never_ignites")
    if all(example.ignited for example in examples):
        failures.append("sweep_always_ignited")
    return failures
