from __future__ import annotations

"""Shared-versus-private consumer experiment for Branch F."""

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from attcon.broadcast import BroadcastConfig, BroadcastExample, CONSUMERS


@dataclass(frozen=True)
class BroadcastTensors:
    shared_input: torch.Tensor
    private_inputs: torch.Tensor
    targets: tuple[torch.Tensor, ...]
    onset_target: torch.Tensor
    ignited: torch.Tensor
    content: torch.Tensor

    def __len__(self) -> int:
        return self.shared_input.shape[0]

    def to(self, device: str | torch.device) -> BroadcastTensors:
        return BroadcastTensors(
            shared_input=self.shared_input.to(device),
            private_inputs=self.private_inputs.to(device),
            targets=tuple(target.to(device) for target in self.targets),
            onset_target=self.onset_target.to(device),
            ignited=self.ignited.to(device),
            content=self.content.to(device),
        )

    def subset(self, indices: torch.Tensor) -> BroadcastTensors:
        return BroadcastTensors(
            shared_input=self.shared_input[indices],
            private_inputs=self.private_inputs[indices],
            targets=tuple(target[indices] for target in self.targets),
            onset_target=self.onset_target[indices],
            ignited=self.ignited[indices],
            content=self.content[indices],
        )


@dataclass(frozen=True)
class BroadcastPrediction:
    action: torch.Tensor
    broad: tuple[torch.Tensor, ...]
    onset: tuple[torch.Tensor, ...]
    action_state: torch.Tensor
    broad_states: tuple[torch.Tensor, ...]


def consumer_class_sizes(config: BroadcastConfig) -> tuple[int, ...]:
    return (
        4,
        config.content_vocab_size + 1,
        config.evidence_levels + 1,
        7,
        config.content_vocab_size + 1,
        config.content_vocab_size + 1,
    )


def tensorize_broadcast_examples(
    examples: Iterable[BroadcastExample],
    config: BroadcastConfig,
) -> BroadcastTensors:
    cases = list(examples)
    if not cases:
        raise ValueError("at least one broadcast example is required")
    input_size = (
        config.content_vocab_size
        + config.cue_strength_levels
        + config.evidence_levels
    )
    class_sizes = consumer_class_sizes(config)
    shared_rows = []
    private_rows = []
    target_rows = [[] for _ in CONSUMERS]
    for example in cases:
        content = nn.functional.one_hot(
            torch.tensor(example.content), config.content_vocab_size
        ).float()
        cue = nn.functional.one_hot(
            torch.tensor(example.cue_strength), config.cue_strength_levels
        ).float()
        evidence = nn.functional.one_hot(
            torch.tensor(example.evidence_quality), config.evidence_levels
        ).float()
        shared_rows.append(torch.cat((content, cue, evidence)))
        broad_private = []
        for consumer_index in range(1, len(CONSUMERS)):
            target = example.consumer_targets[consumer_index]
            target_index = (
                class_sizes[consumer_index] - 1 if target is None else target
            )
            shortcut = torch.zeros(input_size)
            shortcut[target_index] = 1.0
            broad_private.append(shortcut)
        private_rows.append(torch.stack(broad_private))
        for consumer_index, target in enumerate(example.consumer_targets):
            target_rows[consumer_index].append(
                class_sizes[consumer_index] - 1 if target is None else target
            )
    return BroadcastTensors(
        shared_input=torch.stack(shared_rows),
        private_inputs=torch.stack(private_rows),
        targets=tuple(torch.tensor(row) for row in target_rows),
        onset_target=torch.tensor(
            [
                config.num_steps
                if example.ignition_step is None
                else example.ignition_step
                for example in cases
            ]
        ),
        ignited=torch.tensor([example.ignited for example in cases]),
        content=torch.tensor([example.content for example in cases]),
    )


class BroadcastConsumerModel(nn.Module):
    """Exactly matched shared-state or private-shortcut consumer routes."""

    def __init__(
        self,
        config: BroadcastConfig,
        hidden_size: int = 32,
        *,
        shared: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.shared = shared
        input_size = (
            config.content_vocab_size
            + config.cue_strength_levels
            + config.evidence_levels
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh()
        )
        self.action_head = nn.Linear(hidden_size, 4)
        self.broad_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                )
                for _ in range(len(CONSUMERS) - 1)
            ]
        )
        sizes = consumer_class_sizes(config)[1:]
        self.broad_heads = nn.ModuleList(
            [nn.Linear(hidden_size, size) for size in sizes]
        )
        self.onset_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, config.num_steps + 1)
                for _ in range(len(CONSUMERS) - 1)
            ]
        )

    def encode(
        self,
        shared_input: torch.Tensor,
        private_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        action_state = self.action_encoder(shared_input)
        if self.shared:
            candidate_states = tuple(
                encoder(shared_input) for encoder in self.broad_encoders
            )
            shared_state = torch.stack(candidate_states, dim=0).mean(dim=0)
            broad_states = (shared_state,) * len(self.broad_encoders)
        else:
            broad_states = tuple(
                encoder(private_inputs[:, index])
                for index, encoder in enumerate(self.broad_encoders)
            )
        return action_state, broad_states

    def decode(
        self,
        action_state: torch.Tensor,
        broad_states: tuple[torch.Tensor, ...],
    ) -> BroadcastPrediction:
        if self.shared:
            onset_logits = torch.stack(
                [head(broad_states[0]) for head in self.onset_heads], dim=0
            ).mean(dim=0)
            onset = (onset_logits,) * len(self.onset_heads)
        else:
            onset = tuple(
                head(state)
                for head, state in zip(self.onset_heads, broad_states)
            )
        return BroadcastPrediction(
            action=self.action_head(action_state),
            broad=tuple(
                head(state)
                for head, state in zip(self.broad_heads, broad_states)
            ),
            onset=onset,
            action_state=action_state,
            broad_states=broad_states,
        )

    def forward(
        self,
        shared_input: torch.Tensor,
        private_inputs: torch.Tensor,
    ) -> BroadcastPrediction:
        return self.decode(*self.encode(shared_input, private_inputs))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def broadcast_loss(
    prediction: BroadcastPrediction,
    batch: BroadcastTensors,
) -> torch.Tensor:
    loss = nn.functional.cross_entropy(prediction.action, batch.targets[0])
    for index, logits in enumerate(prediction.broad):
        loss = loss + nn.functional.cross_entropy(logits, batch.targets[index + 1])
    for logits in prediction.onset:
        loss = loss + 0.25 * nn.functional.cross_entropy(
            logits, batch.onset_target
        )
    return loss


def train_broadcast_model(
    model: BroadcastConsumerModel,
    train: BroadcastTensors,
    *,
    epochs: int = 35,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    seed: int = 107,
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
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.shared_input, batch.private_inputs)
            loss = broadcast_loss(prediction, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(data))
    return losses


@torch.no_grad()
def broadcast_metrics(
    model: BroadcastConsumerModel,
    data: BroadcastTensors,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    model.eval()
    batch = data.to(device)
    prediction = model(batch.shared_input, batch.private_inputs)
    correct = [prediction.action.argmax(dim=-1).eq(batch.targets[0])]
    correct.extend(
        logits.argmax(dim=-1).eq(batch.targets[index + 1])
        for index, logits in enumerate(prediction.broad)
    )
    onset_predictions = torch.stack(
        [logits.argmax(dim=-1) for logits in prediction.onset], dim=-1
    )
    return {
        "consumer_accuracy": {
            consumer: values.float().mean().item()
            for consumer, values in zip(CONSUMERS, correct)
        },
        "broad_joint_accuracy": torch.stack(correct[1:], dim=-1).all(
            dim=-1
        ).float().mean().item(),
        "onset_accuracy": onset_predictions.eq(
            batch.onset_target[:, None]
        ).float().mean().item(),
        "onset_alignment_rate": onset_predictions.eq(
            onset_predictions[:, :1]
        ).all(dim=-1).float().mean().item(),
    }


def _mean_consumer_accuracy(
    prediction: BroadcastPrediction,
    targets: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    correct = torch.stack(
        [
            logits.argmax(dim=-1).eq(targets[index + 1])
            for index, logits in enumerate(prediction.broad)
        ],
        dim=-1,
    )
    return correct.float().mean()


@torch.no_grad()
def broadcast_intervention_metrics(
    shared_model: BroadcastConsumerModel,
    private_model: BroadcastConsumerModel,
    data: BroadcastTensors,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    batch = data.to(device)
    mask = batch.ignited
    shared_batch = batch.subset(torch.nonzero(mask, as_tuple=False).flatten())
    shared_model.eval()
    private_model.eval()

    shared_original = shared_model(
        shared_batch.shared_input, shared_batch.private_inputs
    )
    shared_zero_states = tuple(
        torch.zeros_like(state) for state in shared_original.broad_states
    )
    shared_ablated = shared_model.decode(
        shared_original.action_state, shared_zero_states
    )
    private_original = private_model(
        shared_batch.shared_input, shared_batch.private_inputs
    )
    private_states = list(private_original.broad_states)
    private_states[0] = torch.zeros_like(private_states[0])
    private_ablated = private_model.decode(
        private_original.action_state, tuple(private_states)
    )

    shared_original_accuracy = _mean_consumer_accuracy(
        shared_original, shared_batch.targets
    )
    private_original_accuracy = _mean_consumer_accuracy(
        private_original, shared_batch.targets
    )
    shared_drop = shared_original_accuracy - _mean_consumer_accuracy(
        shared_ablated, shared_batch.targets
    )
    private_drop = private_original_accuracy - _mean_consumer_accuracy(
        private_ablated, shared_batch.targets
    )

    count = len(shared_batch)
    donor = torch.arange(count, device=device)
    for row in range(count):
        for offset in range(1, count):
            candidate = (row + offset) % count
            if shared_batch.content[candidate] != shared_batch.content[row]:
                donor[row] = candidate
                break
    donor_state = shared_original.broad_states[0][donor]
    swapped = shared_model.decode(
        shared_original.action_state, (donor_state,) * (len(CONSUMERS) - 1)
    )
    donor_targets = tuple(target[donor] for target in shared_batch.targets)
    swap_follow = _mean_consumer_accuracy(swapped, donor_targets)
    action_invariance = swapped.action.argmax(dim=-1).eq(
        shared_original.action.argmax(dim=-1)
    ).float().mean()
    return {
        "shared_broad_accuracy_drop_after_zero": shared_drop.item(),
        "private_mean_broad_accuracy_drop_after_one_route_zero": private_drop.item(),
        "coordinated_ablation_drop_advantage": (shared_drop - private_drop).item(),
        "shared_content_swap_broad_follow_rate": swap_follow.item(),
        "local_action_invariance_under_shared_swap": action_invariance.item(),
        "ignited_intervention_count": float(count),
    }
