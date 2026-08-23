from __future__ import annotations

"""Branch C benchmark cases for unity/binding experiments.

These cases deliberately separate individual feature availability from correct object binding.
Every false-binding lure is assembled from feature values that occur in the same scene, while
the assembled conjunction itself does not occur. A deterministic target-conjunction hash creates
a held-out split without withholding any individual feature vocabulary.
"""

from dataclasses import asdict, dataclass
import random
from typing import Any


@dataclass(frozen=True)
class BindingConfig:
    grid_size: int = 5
    num_objects: int = 8
    num_visible_types: int = 4
    digit_vocab_size: int = 10
    num_cues: int = 4
    heldout_modulus: int = 5

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size


@dataclass(frozen=True)
class BindingObject:
    location: int
    visible_type: int
    digit: int
    cue_tag: int
    inspected: bool

    def conjunction(self) -> tuple[int, int, int, int, int]:
        return (
            self.location,
            self.visible_type,
            self.digit,
            self.cue_tag,
            int(self.inspected),
        )


@dataclass(frozen=True)
class BindingExample:
    example_id: str
    cue: int
    objects: tuple[BindingObject, ...]
    target_index: int
    false_binding_lure: BindingObject
    split: str

    @property
    def target(self) -> BindingObject:
        return self.objects[self.target_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "cue": self.cue,
            "objects": [asdict(obj) for obj in self.objects],
            "target_index": self.target_index,
            "target": asdict(self.target),
            "false_binding_lure": asdict(self.false_binding_lure),
            "split": self.split,
        }


def _heldout_conjunction(
    target: BindingObject,
    cue: int,
    modulus: int,
) -> bool:
    if modulus < 2:
        raise ValueError("heldout_modulus must be >= 2")
    # Fixed integer mixing keeps the split reproducible across Python processes (unlike hash()).
    mixed = (
        target.location * 73856093
        + target.visible_type * 19349663
        + target.digit * 83492791
        + int(target.inspected) * 2654435761
        + cue * 97531
    )
    return mixed % modulus == 0


def _false_binding_lure(
    objects: tuple[BindingObject, ...],
    cue: int,
    rng: random.Random,
) -> BindingObject:
    real_conjunctions = {obj.conjunction() for obj in objects}
    cue_objects = [obj for obj in objects if obj.cue_tag == cue]
    if not cue_objects:
        raise ValueError("scene has no object matching the cue")

    for _ in range(256):
        # Each feature is copied from a real scene object, but independent donors break the bind.
        location = rng.choice(objects).location
        visible_type = rng.choice(objects).visible_type
        digit = rng.choice(objects).digit
        inspected = rng.choice(objects).inspected
        lure = BindingObject(
            location=location,
            visible_type=visible_type,
            digit=digit,
            cue_tag=rng.choice(cue_objects).cue_tag,
            inspected=inspected,
        )
        if lure.conjunction() not in real_conjunctions:
            return lure
    raise RuntimeError("failed to construct an absent false-binding conjunction")


def generate_binding_examples(
    count: int,
    *,
    config: BindingConfig | None = None,
    seed: int = 7,
) -> list[BindingExample]:
    """Generate deterministic Branch C scenes with held-out binds and false-binding lures."""

    if count < 1:
        raise ValueError("count must be >= 1")
    cfg = config or BindingConfig()
    if not (2 <= cfg.num_objects <= cfg.num_cells):
        raise ValueError("num_objects must be between 2 and the number of grid cells")
    if cfg.num_cues < 2:
        raise ValueError("num_cues must be >= 2")
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        cue = rng.randrange(cfg.num_cues)
        locations = rng.sample(range(cfg.num_cells), cfg.num_objects)
        non_target_cues = tuple(value for value in range(cfg.num_cues) if value != cue)
        objects = [
            BindingObject(
                location=location,
                visible_type=rng.randrange(cfg.num_visible_types),
                digit=rng.randrange(cfg.digit_vocab_size),
                cue_tag=rng.choice(non_target_cues),
                inspected=bool(rng.randrange(2)),
            )
            for location in locations
        ]
        # Guarantee exactly one cue-relevant object without tying its other attributes to cue.
        forced_index = rng.randrange(cfg.num_objects)
        forced = objects[forced_index]
        objects[forced_index] = BindingObject(
            location=forced.location,
            visible_type=forced.visible_type,
            digit=forced.digit,
            cue_tag=cue,
            inspected=forced.inspected,
        )
        object_tuple = tuple(objects)
        candidates = [idx for idx, obj in enumerate(object_tuple) if obj.cue_tag == cue]
        target_index = rng.choice(candidates)
        target = object_tuple[target_index]
        split = (
            "heldout_conjunction"
            if _heldout_conjunction(target, cue, cfg.heldout_modulus)
            else "train"
        )
        examples.append(
            BindingExample(
                example_id=f"binding_{seed}_{index}",
                cue=cue,
                objects=object_tuple,
                target_index=target_index,
                false_binding_lure=_false_binding_lure(object_tuple, cue, rng),
                split=split,
            )
        )
    return examples


def validate_binding_example(example: BindingExample) -> list[str]:
    """Return invariant failures; an empty list means the case is valid."""

    failures = []
    conjunctions = {obj.conjunction() for obj in example.objects}
    lure = example.false_binding_lure
    target = example.target
    if target.cue_tag != example.cue:
        failures.append("target_not_cue_relevant")
    if lure.cue_tag != example.cue:
        failures.append("lure_not_cue_relevant")
    if lure.conjunction() in conjunctions:
        failures.append("lure_conjunction_present")
    for field in ("location", "visible_type", "digit", "cue_tag", "inspected"):
        if getattr(lure, field) not in {getattr(obj, field) for obj in example.objects}:
            failures.append(f"lure_{field}_not_individually_present")
    if len({obj.location for obj in example.objects}) != len(example.objects):
        failures.append("duplicate_object_location")
    return failures
