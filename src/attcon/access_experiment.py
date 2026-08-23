from __future__ import annotations

"""Trained recurrent-access pilot for Branch D."""

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from attcon.counterfactual_access import (
    CounterfactualAccessConfig,
    CounterfactualAccessExample,
    TARGET_STATUSES,
    UNKNOWN_ANSWER,
)


@dataclass(frozen=True)
class AccessTensors:
    events: torch.Tensor
    scene: torch.Tensor
    query: torch.Tensor
    targets: torch.Tensor
    statuses: torch.Tensor
    target_slots: torch.Tensor
    scene_answers: torch.Tensor
    glimpse_answers: torch.Tensor

    def __len__(self) -> int:
        return self.events.shape[0]

    def to(self, device: str | torch.device) -> AccessTensors:
        return AccessTensors(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    def subset(self, indices: torch.Tensor) -> AccessTensors:
        return AccessTensors(
            **{name: value[indices] for name, value in self.__dict__.items()}
        )


def _answer_index(value: int, config: CounterfactualAccessConfig) -> int:
    return config.value_vocab_size if value == UNKNOWN_ANSWER else value


def tensorize_access_examples(
    examples: Iterable[CounterfactualAccessExample],
    config: CounterfactualAccessConfig,
) -> AccessTensors:
    cases = list(examples)
    if not cases:
        raise ValueError("at least one access example is required")
    event_dim = config.key_vocab_size + config.value_vocab_size + 3
    scene_dim = config.key_vocab_size * config.value_vocab_size
    events = torch.zeros(len(cases), config.num_items, event_dim)
    scene = torch.zeros(len(cases), scene_dim)
    query = torch.zeros(len(cases), config.key_vocab_size)
    status_to_index = {status: index for index, status in enumerate(TARGET_STATUSES)}
    for row, example in enumerate(cases):
        query[row, example.switched_query_key] = 1.0
        for item in example.items:
            if item.current_observation_value is not None:
                scene[
                    row,
                    item.key * config.value_vocab_size
                    + item.current_observation_value,
                ] = 1.0
            if item.access_cache_value is None:
                continue
            events[row, item.slot, item.key] = 1.0
            events[
                row,
                item.slot,
                config.key_vocab_size + item.access_cache_value,
            ] = 1.0
            event_type = 0 if item.attended_before else 1
            events[
                row,
                item.slot,
                config.key_vocab_size + config.value_vocab_size + event_type,
            ] = 1.0
            events[
                row,
                item.slot,
                config.key_vocab_size + config.value_vocab_size + 2,
            ] = 1.0
    return AccessTensors(
        events=events,
        scene=scene,
        query=query,
        targets=torch.tensor(
            [_answer_index(case.expected_answer, config) for case in cases]
        ),
        statuses=torch.tensor(
            [status_to_index[case.target_status] for case in cases]
        ),
        target_slots=torch.tensor([case.target_index for case in cases]),
        scene_answers=torch.tensor(
            [_answer_index(case.scene_only_answer, config) for case in cases]
        ),
        glimpse_answers=torch.tensor(
            [_answer_index(case.current_glimpse_answer, config) for case in cases]
        ),
    )


class RecurrentAccessModel(nn.Module):
    """Compress access events into recurrent state before the report query is decoded."""

    def __init__(
        self,
        config: CounterfactualAccessConfig,
        hidden_size: int = 96,
        fusion_size: int = 128,
    ) -> None:
        super().__init__()
        self.config = config
        event_dim = config.key_vocab_size + config.value_vocab_size + 3
        scene_dim = config.key_vocab_size * config.value_vocab_size
        self.memory = nn.GRU(event_dim, hidden_size, batch_first=True)
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, hidden_size),
            nn.Tanh(),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(config.key_vocab_size, hidden_size // 2),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size // 2, fusion_size),
            nn.Tanh(),
            nn.Linear(fusion_size, config.value_vocab_size + 1),
        )

    def forward(
        self,
        events: torch.Tensor,
        scene: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        _, hidden = self.memory(events)
        return self.decoder(
            torch.cat(
                (
                    hidden[-1],
                    self.scene_encoder(scene),
                    self.query_encoder(query),
                ),
                dim=-1,
            )
        )


class RelationalRecurrentAccessModel(nn.Module):
    """Address recurrent value states by query-key equality.

    Keys are used only to select a recurrent output; the GRU never receives the key identity.
    This forces value representations to transfer to held-out key/value conjunctions while
    retaining a recurrent bottleneck between access events and report.
    """

    def __init__(
        self,
        config: CounterfactualAccessConfig,
        hidden_size: int = 96,
        fusion_size: int = 128,
    ) -> None:
        super().__init__()
        self.config = config
        value_event_dim = config.value_vocab_size + 3
        self.memory = nn.GRU(value_event_dim, hidden_size, batch_first=True)
        self.scene_value_encoder = nn.Linear(config.value_vocab_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, fusion_size),
            nn.Tanh(),
            nn.Linear(fusion_size, config.value_vocab_size + 1),
        )

    def forward(
        self,
        events: torch.Tensor,
        scene: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        key_width = self.config.key_vocab_size
        event_values = events[:, :, key_width:]
        memory_outputs, _ = self.memory(event_values)
        key_matches = torch.einsum(
            "bok,bk->bo", events[:, :, :key_width], query
        )
        memory_context = torch.einsum("bo,boh->bh", key_matches, memory_outputs)
        memory_present = key_matches.sum(dim=-1, keepdim=True).clamp(max=1.0)

        scene_by_key = scene.reshape(
            -1, self.config.key_vocab_size, self.config.value_vocab_size
        )
        scene_value = torch.einsum("bk,bkv->bv", query, scene_by_key)
        scene_context = torch.tanh(self.scene_value_encoder(scene_value))
        scene_present = scene_value.sum(dim=-1, keepdim=True).clamp(max=1.0)
        return self.decoder(
            torch.cat(
                (memory_context, scene_context, memory_present, scene_present), dim=-1
            )
        )


class SetTransformerAccessModel(nn.Module):
    """Relational access through a permutation-equivariant event-set transformer."""

    def __init__(
        self,
        config: CounterfactualAccessConfig,
        hidden_size: int = 32,
        fusion_size: int = 64,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.config = config
        value_event_dim = config.value_vocab_size + 3
        self.event_projection = nn.Linear(value_event_dim, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.memory = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.scene_value_encoder = nn.Linear(config.value_vocab_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, fusion_size),
            nn.Tanh(),
            nn.Linear(fusion_size, config.value_vocab_size + 1),
        )

    def forward(
        self,
        events: torch.Tensor,
        scene: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        key_width = self.config.key_vocab_size
        event_values = events[:, :, key_width:]
        valid = event_values[:, :, -1].bool()
        padding_mask = ~valid
        # Transformer attention cannot consume an entirely masked row. The sentinel output is
        # harmless because key_matches remains zero in the no-cache condition.
        empty = ~valid.any(dim=-1)
        padding_mask[empty, 0] = False
        memory_outputs = self.memory(
            self.event_projection(event_values),
            src_key_padding_mask=padding_mask,
        )
        key_matches = torch.einsum(
            "bok,bk->bo", events[:, :, :key_width], query
        )
        memory_context = torch.einsum("bo,boh->bh", key_matches, memory_outputs)
        memory_present = key_matches.sum(dim=-1, keepdim=True).clamp(max=1.0)

        scene_by_key = scene.reshape(
            -1, self.config.key_vocab_size, self.config.value_vocab_size
        )
        scene_value = torch.einsum("bk,bkv->bv", query, scene_by_key)
        scene_context = torch.tanh(self.scene_value_encoder(scene_value))
        scene_present = scene_value.sum(dim=-1, keepdim=True).clamp(max=1.0)
        return self.decoder(
            torch.cat(
                (memory_context, scene_context, memory_present, scene_present), dim=-1
            )
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_access_model(
    model: nn.Module,
    train: AccessTensors,
    *,
    erase_cache: bool = False,
    epochs: int = 35,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    seed: int = 71,
    device: str = "cpu",
) -> list[float]:
    torch.manual_seed(seed)
    model.to(device)
    data = train.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(data), generator=generator)
        total = 0.0
        for start in range(0, len(data), batch_size):
            indices = order[start : start + batch_size].to(device)
            batch = data.subset(indices)
            events = torch.zeros_like(batch.events) if erase_cache else batch.events
            optimizer.zero_grad(set_to_none=True)
            logits = model(events, batch.scene, batch.query)
            loss = nn.functional.cross_entropy(logits, batch.targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(data))
    return losses


@torch.no_grad()
def evaluate_access_model(
    model: nn.Module,
    data: AccessTensors,
    *,
    erase_cache: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    model.eval()
    batch = data.to(device)
    events = torch.zeros_like(batch.events) if erase_cache else batch.events
    predictions = model(events, batch.scene, batch.query).argmax(dim=-1)
    correct = predictions.eq(batch.targets)
    by_status = {}
    for index, status in enumerate(TARGET_STATUSES):
        mask = batch.statuses.eq(index)
        by_status[status] = {
            "count": mask.sum().item(),
            "accuracy": correct[mask].float().mean().item(),
        }
    memory_mask = batch.statuses.eq(2) | batch.statuses.eq(3)
    return {
        "count": len(data),
        "accuracy": correct.float().mean().item(),
        "memory_and_tension_accuracy": correct[memory_mask].float().mean().item(),
        "by_status": by_status,
        "predictions": predictions.cpu(),
    }


@torch.no_grad()
def access_intervention_metrics(
    model: nn.Module,
    data: AccessTensors,
    config: CounterfactualAccessConfig,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    original = model(batch.events, batch.scene, batch.query).argmax(dim=-1)
    target_rows = torch.arange(len(batch), device=device)

    erased_events = batch.events.clone()
    erased_events[target_rows, batch.target_slots] = 0.0
    erased = model(erased_events, batch.scene, batch.query).argmax(dim=-1)
    memory_mask = batch.statuses.eq(2) | batch.statuses.eq(3)
    original_correct = original.eq(batch.targets)
    erased_correct = erased.eq(batch.targets)

    tension_mask = batch.statuses.eq(3)
    changed_scene = batch.scene.clone()
    query_keys = batch.query.argmax(dim=-1)
    for row in torch.nonzero(tension_mask, as_tuple=False).flatten():
        key = query_keys[row]
        segment = slice(
            int(key.item()) * config.value_vocab_size,
            (int(key.item()) + 1) * config.value_vocab_size,
        )
        old_value = changed_scene[row, segment].argmax()
        changed_scene[row, segment] = 0.0
        changed_scene[
            row,
            segment.start + (int(old_value.item()) + 1) % config.value_vocab_size,
        ] = 1.0
    scene_changed = model(batch.events, changed_scene, batch.query).argmax(dim=-1)
    return {
        "memory_target_cache_erasure_accuracy_drop": (
            original_correct[memory_mask].float().mean()
            - erased_correct[memory_mask].float().mean()
        ).item(),
        "counterfactual_observation_change_invariance": scene_changed[
            tension_mask
        ].eq(original[tension_mask]).float().mean().item(),
        "counterfactual_cache_answer_retention_after_observation_change": scene_changed[
            tension_mask
        ].eq(batch.targets[tension_mask]).float().mean().item(),
    }
