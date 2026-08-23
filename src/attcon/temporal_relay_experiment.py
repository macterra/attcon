from __future__ import annotations

"""Sequence models and causal metrics for the temporal-relay benchmark."""

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from torch import nn

from attcon.temporal_relay import (
    RELAY_STATUSES,
    UNKNOWN_PAYLOAD,
    TemporalRelayConfig,
    TemporalRelayExample,
)


@dataclass(frozen=True)
class TemporalRelayTensors:
    events: torch.Tensor
    query: torch.Tensor
    transition: torch.Tensor
    time_index: torch.Tensor
    operation: torch.Tensor
    payload: torch.Tensor
    access_answer: torch.Tensor
    status: torch.Tensor

    def __len__(self) -> int:
        return self.events.shape[0]

    def to(self, device: str | torch.device) -> TemporalRelayTensors:
        return TemporalRelayTensors(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    def subset(self, indices: torch.Tensor) -> TemporalRelayTensors:
        return TemporalRelayTensors(
            **{name: value[indices] for name, value in self.__dict__.items()}
        )


@dataclass(frozen=True)
class TemporalRelayPrediction:
    time_index: torch.Tensor
    operation: torch.Tensor
    payload: torch.Tensor
    access_answer: torch.Tensor
    binding_state: torch.Tensor
    access_initial_state: torch.Tensor


def event_dim(config: TemporalRelayConfig) -> int:
    return (
        config.entity_vocab_size
        + config.operation_vocab_size
        + config.payload_vocab_size
    )


def tensorize_temporal_relay_examples(
    examples: Iterable[TemporalRelayExample],
    config: TemporalRelayConfig,
) -> TemporalRelayTensors:
    cases = list(examples)
    if not cases:
        raise ValueError("at least one temporal-relay example is required")
    events = torch.zeros(len(cases), config.stream_length, event_dim(config))
    query = torch.zeros(len(cases), config.entity_vocab_size)
    transition = torch.zeros(len(cases), 4)
    status_index = {status: index for index, status in enumerate(RELAY_STATUSES)}
    for row, example in enumerate(cases):
        query[row, example.query_entity] = 1.0
        for event in example.events:
            events[row, event.time_index, event.entity] = 1.0
            events[
                row,
                event.time_index,
                config.entity_vocab_size + event.operation,
            ] = 1.0
            events[
                row,
                event.time_index,
                config.entity_vocab_size
                + config.operation_vocab_size
                + event.payload,
            ] = 1.0
        transition[row, 0] = float(example.live_buffer_payload is not None)
        transition[row, 1] = float(example.archive_payload is not None)
        transition[row, 2] = float(example.target_status == "archived")
        transition[row, 3] = float(example.target_status == "conflict_recoverable")
    return TemporalRelayTensors(
        events=events,
        query=query,
        transition=transition,
        time_index=torch.tensor([case.target.time_index for case in cases]),
        operation=torch.tensor([case.target.operation for case in cases]),
        payload=torch.tensor([case.target.payload for case in cases]),
        access_answer=torch.tensor(
            [
                config.payload_vocab_size
                if case.expected_payload == UNKNOWN_PAYLOAD
                else case.expected_payload
                for case in cases
            ]
        ),
        status=torch.tensor([status_index[case.target_status] for case in cases]),
    )


class TemporalRelayModel(nn.Module):
    """Resolve ordered updates into shared, split, or order-destroyed states."""

    def __init__(
        self,
        config: TemporalRelayConfig,
        hidden_size: int = 64,
        *,
        mode: Literal["shared", "split", "pooled"] = "shared",
    ) -> None:
        super().__init__()
        if mode not in {"shared", "split", "pooled"}:
            raise ValueError(f"unknown temporal-relay mode: {mode}")
        self.config = config
        self.mode = mode
        input_size = event_dim(config) + config.entity_vocab_size
        self.binding_memory = nn.GRU(input_size, hidden_size, batch_first=True)
        self.access_memory = nn.GRU(input_size, hidden_size, batch_first=True)
        self.transition = nn.GRUCell(4, hidden_size)
        self.time_head = nn.Linear(hidden_size, config.stream_length)
        self.operation_head = nn.Linear(hidden_size, config.operation_vocab_size)
        self.payload_head = nn.Linear(hidden_size, config.payload_vocab_size)
        self.access_head = nn.Linear(hidden_size, config.payload_vocab_size + 1)

    def initial_states(
        self, events: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "pooled":
            events = events.mean(dim=1, keepdim=True).expand_as(events)
        expanded_query = query[:, None, :].expand(-1, events.shape[1], -1)
        stream = torch.cat((events, expanded_query), dim=-1)
        _, binding = self.binding_memory(stream)
        _, access = self.access_memory(stream)
        binding_state, access_state = binding[-1], access[-1]
        if self.mode in {"shared", "pooled"}:
            shared = (binding_state + access_state) / 2.0
            return shared, shared
        return binding_state, access_state

    def forward(
        self,
        events: torch.Tensor,
        query: torch.Tensor,
        transition: torch.Tensor,
        *,
        binding_state_override: torch.Tensor | None = None,
    ) -> TemporalRelayPrediction:
        binding_state, access_initial = self.initial_states(events, query)
        if binding_state_override is not None:
            binding_state = binding_state_override
            if self.mode in {"shared", "pooled"}:
                access_initial = binding_state_override
        access_state = self.transition(transition, access_initial)
        return TemporalRelayPrediction(
            time_index=self.time_head(binding_state),
            operation=self.operation_head(binding_state),
            payload=self.payload_head(binding_state),
            access_answer=self.access_head(access_state),
            binding_state=binding_state,
            access_initial_state=access_initial,
        )


class RelationalTemporalRelayModel(TemporalRelayModel):
    """Remove entity identity after an exact query match, while preserving stream order."""

    def __init__(
        self,
        config: TemporalRelayConfig,
        hidden_size: int = 64,
        *,
        mode: Literal["shared", "split", "pooled"] = "shared",
    ) -> None:
        super().__init__(config, hidden_size, mode=mode)
        relational_dim = (
            config.operation_vocab_size + config.payload_vocab_size + 1
        )
        self.binding_memory = nn.GRU(relational_dim, hidden_size, batch_first=True)
        self.access_memory = nn.GRU(relational_dim, hidden_size, batch_first=True)

    def initial_states(
        self, events: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entity_width = self.config.entity_vocab_size
        matches = torch.einsum("bte,be->bt", events[:, :, :entity_width], query)
        content = events[:, :, entity_width:] * matches[:, :, None]
        stream = torch.cat((content, matches[:, :, None]), dim=-1)
        if self.mode == "pooled":
            stream = stream.mean(dim=1, keepdim=True).expand_as(stream)
        _, binding = self.binding_memory(stream)
        _, access = self.access_memory(stream)
        binding_state, access_state = binding[-1], access[-1]
        if self.mode in {"shared", "pooled"}:
            shared = (binding_state + access_state) / 2.0
            return shared, shared
        return binding_state, access_state


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_temporal_relay_model(
    model: TemporalRelayModel,
    train: TemporalRelayTensors,
    *,
    epochs: int = 35,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    seed: int = 1123,
    device: str = "cpu",
) -> list[float]:
    torch.manual_seed(seed)
    model.to(device)
    data = train.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(data), generator=generator)
        total = 0.0
        for start in range(0, len(data), batch_size):
            indices = order[start : start + batch_size].to(device)
            batch = data.subset(indices)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.events, batch.query, batch.transition)
            loss = sum(
                nn.functional.cross_entropy(
                    getattr(output, name), getattr(batch, name)
                )
                for name in ("time_index", "operation", "payload", "access_answer")
            )
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(data))
    return losses


@torch.no_grad()
def evaluate_temporal_relay_model(
    model: TemporalRelayModel,
    data: TemporalRelayTensors,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    model.eval()
    batch = data.to(device)
    output = model(batch.events, batch.query, batch.transition)
    binding = (
        output.time_index.argmax(dim=-1).eq(batch.time_index)
        & output.operation.argmax(dim=-1).eq(batch.operation)
        & output.payload.argmax(dim=-1).eq(batch.payload)
    )
    access = output.access_answer.argmax(dim=-1).eq(batch.access_answer)
    by_status = {}
    for index, status in enumerate(RELAY_STATUSES):
        mask = batch.status.eq(index)
        by_status[status] = access[mask].float().mean().item()
    return {
        "count": len(data),
        "binding_joint_accuracy": binding.float().mean().item(),
        "access_accuracy": access.float().mean().item(),
        "binding_and_access_joint_accuracy": (binding & access).float().mean().item(),
        "access_accuracy_by_status": by_status,
    }


@torch.no_grad()
def payload_direction_metrics(
    model: TemporalRelayModel,
    fit_data: TemporalRelayTensors,
    test_data: TemporalRelayTensors,
    *,
    permute_fit_labels: bool = False,
    seed: int = 1129,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    fit, test = fit_data.to(device), test_data.to(device)
    fit_states, _ = model.initial_states(fit.events, fit.query)
    labels = fit.payload.clone()
    if permute_fit_labels:
        generator = torch.Generator().manual_seed(seed)
        labels = labels[torch.randperm(len(labels), generator=generator).to(device)]
    centroids = torch.stack(
        [fit_states[labels.eq(value)].mean(dim=0) for value in range(model.config.payload_vocab_size)]
    )
    test_states, _ = model.initial_states(test.events, test.query)
    donor = (test.payload + 1) % model.config.payload_vocab_size
    edited_state = test_states + centroids[donor] - centroids[test.payload]
    edited = model(
        test.events,
        test.query,
        test.transition,
        binding_state_override=edited_state,
    )
    binding_follow = edited.payload.argmax(dim=-1).eq(donor)
    other_stable = (
        edited.time_index.argmax(dim=-1).eq(test.time_index)
        & edited.operation.argmax(dim=-1).eq(test.operation)
    )
    accessible = ~test.status.eq(0)
    access_follow = edited.access_answer.argmax(dim=-1).eq(donor)
    return {
        "binding_payload_donor_follow_rate": binding_follow.float().mean().item(),
        "binding_other_fields_stability": other_stable.float().mean().item(),
        "accessible_access_donor_follow_rate": access_follow[accessible].float().mean().item(),
        "accessible_joint_donor_follow_rate": (binding_follow & access_follow)[accessible]
        .float()
        .mean()
        .item(),
    }
