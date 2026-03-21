from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class TaskConfig:
    grid_size: int = 5
    num_types: int = 4
    digit_vocab_size: int = 10
    min_type_count: int = 4
    num_steps: int = 4

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def visible_feature_dim(self) -> int:
        return self.num_types

    @property
    def hidden_feature_dim(self) -> int:
        return self.num_types + self.digit_vocab_size

    @property
    def scene_feature_dim(self) -> int:
        return self.visible_feature_dim + self.hidden_feature_dim

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TaskConfig":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Batch:
    scene: torch.Tensor
    cue: torch.Tensor
    target: torch.Tensor
    target_pos: torch.Tensor
    visible_types: torch.Tensor
    digits: torch.Tensor
    target_types: torch.Tensor

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            scene=self.scene.to(device),
            cue=self.cue.to(device),
            target=self.target.to(device),
            target_pos=self.target_pos.to(device),
            visible_types=self.visible_types.to(device),
            digits=self.digits.to(device),
            target_types=self.target_types.to(device),
        )


def _randint(
    high: int,
    shape: tuple[int, ...],
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(high, shape, generator=generator, device=device)


def generate_batch(
    batch_size: int,
    num_steps: int,
    task_config: TaskConfig | dict[str, Any],
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> Batch:
    del num_steps
    cfg = task_config if isinstance(task_config, TaskConfig) else TaskConfig.from_dict(task_config)
    device = torch.device(device)
    num_cells = cfg.num_cells

    if cfg.min_type_count * cfg.num_types > num_cells:
        raise ValueError("min_type_count * num_types must fit within the grid")

    scene = torch.zeros(batch_size, num_cells, cfg.scene_feature_dim, device=device)
    cue = _randint(cfg.num_types, (batch_size,), generator, device)
    target = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
    visible_types = torch.zeros(batch_size, num_cells, dtype=torch.long, device=device)
    digits = torch.zeros(batch_size, num_cells, dtype=torch.long, device=device)
    target_types = torch.full((batch_size, num_cells), -1, dtype=torch.long, device=device)

    base_types = torch.arange(cfg.num_types, device=device).repeat_interleave(cfg.min_type_count)
    extra_cells = num_cells - base_types.numel()

    for batch_idx in range(batch_size):
        if extra_cells > 0:
            extra = _randint(cfg.num_types, (extra_cells,), generator, device)
            cell_types = torch.cat([base_types, extra], dim=0)
        else:
            cell_types = base_types.clone()

        perm = torch.randperm(num_cells, generator=generator, device=device)
        cell_types = cell_types[perm]
        cell_digits = _randint(cfg.digit_vocab_size, (num_cells,), generator, device)
        visible = F.one_hot(cell_types, num_classes=cfg.num_types).float()
        hidden_targets = torch.zeros(num_cells, cfg.num_types, device=device)

        cue_targets = torch.full((cfg.num_types,), -1, dtype=torch.long, device=device)
        for type_idx in range(cfg.num_types):
            type_positions = torch.nonzero(cell_types == type_idx, as_tuple=False).flatten()
            target_index = type_positions[
                _randint(type_positions.numel(), (1,), generator, device).item()
            ]
            hidden_targets[target_index, type_idx] = 1.0
            cue_targets[type_idx] = target_index
            target_types[batch_idx, target_index] = type_idx

        hidden_digits = F.one_hot(cell_digits, num_classes=cfg.digit_vocab_size).float()
        scene[batch_idx] = torch.cat([visible, hidden_targets, hidden_digits], dim=-1)
        visible_types[batch_idx] = cell_types
        digits[batch_idx] = cell_digits

        cue_idx = cue[batch_idx].item()
        target_position = cue_targets[cue_idx]
        target_pos[batch_idx] = target_position
        target[batch_idx] = cell_digits[target_position]

    return Batch(
        scene=scene,
        cue=cue,
        target=target,
        target_pos=target_pos,
        visible_types=visible_types,
        digits=digits,
        target_types=target_types,
    )


def expand_cues_for_probe(batch: Batch, num_types: int) -> Batch:
    batch_size, num_cells, feature_dim = batch.scene.shape
    expanded_scene = batch.scene.repeat_interleave(num_types, dim=0)
    expanded_visible_types = batch.visible_types.repeat_interleave(num_types, dim=0)
    expanded_digits = batch.digits.repeat_interleave(num_types, dim=0)
    expanded_target_types = batch.target_types.repeat_interleave(num_types, dim=0)
    cues = torch.arange(num_types, device=batch.scene.device).repeat(batch_size)
    targets = torch.zeros(batch_size * num_types, dtype=torch.long, device=batch.scene.device)
    target_pos = torch.zeros(batch_size * num_types, dtype=torch.long, device=batch.scene.device)

    for scene_idx in range(batch_size):
        source_digits = batch.digits[scene_idx]
        source_target_types = batch.target_types[scene_idx]
        for cue_idx in range(num_types):
            row = scene_idx * num_types + cue_idx
            pos = torch.nonzero(source_target_types == cue_idx, as_tuple=False).flatten().item()
            target_pos[row] = pos
            targets[row] = source_digits[pos]

    return Batch(
        scene=expanded_scene.view(batch_size * num_types, num_cells, feature_dim),
        cue=cues,
        target=targets,
        target_pos=target_pos,
        visible_types=expanded_visible_types,
        digits=expanded_digits,
        target_types=expanded_target_types,
    )
