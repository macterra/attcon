from __future__ import annotations

"""Engineering pilot for Branch E higher-order state representation."""

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import nn

from attcon.higher_order import (
    HIGHER_ORDER_STATUSES,
    HigherOrderConfig,
    HigherOrderExample,
)


@dataclass(frozen=True)
class HigherOrderTensors:
    model_input: torch.Tensor
    first_order_features: torch.Tensor
    observation_features: torch.Tensor
    report_target: torch.Tensor
    confidence_target: torch.Tensor
    reinspect_target: torch.Tensor
    correction_target: torch.Tensor
    status_target: torch.Tensor

    def __len__(self) -> int:
        return self.model_input.shape[0]

    def to(self, device: str | torch.device) -> HigherOrderTensors:
        return HigherOrderTensors(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    def subset(self, indices: torch.Tensor) -> HigherOrderTensors:
        return HigherOrderTensors(
            **{name: value[indices] for name, value in self.__dict__.items()}
        )


@dataclass(frozen=True)
class HigherOrderPrediction:
    hidden: torch.Tensor
    report: torch.Tensor
    confidence: torch.Tensor
    reinspect: torch.Tensor
    correction: torch.Tensor


def _source_one_hot(
    value: int | None,
    width: int,
) -> torch.Tensor:
    result = torch.zeros(width + 1)
    result[width if value is None else value] = 1.0
    return result


def tensorize_higher_order_examples(
    examples: Iterable[HigherOrderExample],
    config: HigherOrderConfig,
) -> HigherOrderTensors:
    cases = list(examples)
    if not cases:
        raise ValueError("at least one higher-order example is required")
    status_to_index = {
        status: index for index, status in enumerate(HIGHER_ORDER_STATUSES)
    }
    model_rows = []
    first_order_rows = []
    observation_rows = []
    for example in cases:
        key = torch.nn.functional.one_hot(
            torch.tensor(example.content_key), config.key_vocab_size
        ).float()
        content = torch.nn.functional.one_hot(
            torch.tensor(example.content_value), config.value_vocab_size
        ).float()
        observation = _source_one_hot(
            example.current_observation_value, config.value_vocab_size
        )
        memory = _source_one_hot(example.memory_value, config.value_vocab_size)
        inference = _source_one_hot(example.inferred_value, config.value_vocab_size)
        gate = torch.tensor([float(example.access_gate_open)])
        model_rows.append(torch.cat((key, observation, memory, inference, gate)))
        first_order_rows.append(torch.cat((key, content)))
        observation_rows.append(torch.cat((key, observation)))
    unknown = config.value_vocab_size
    return HigherOrderTensors(
        model_input=torch.stack(model_rows),
        first_order_features=torch.stack(first_order_rows),
        observation_features=torch.stack(observation_rows),
        report_target=torch.tensor(
            [
                case.content_value if case.access_available else unknown
                for case in cases
            ]
        ),
        confidence_target=torch.tensor([case.confidence_band for case in cases]),
        reinspect_target=torch.tensor(
            [int(case.should_reinspect) for case in cases]
        ),
        correction_target=torch.tensor(
            [int(case.should_correct) for case in cases]
        ),
        status_target=torch.tensor([status_to_index[case.status] for case in cases]),
    )


class HigherOrderBehaviorModel(nn.Module):
    """Learn a shared latent only through downstream behavior objectives."""

    def __init__(
        self,
        config: HigherOrderConfig,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.config = config
        input_size = config.key_vocab_size + 3 * (config.value_vocab_size + 1) + 1
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.report_head = nn.Linear(hidden_size, config.value_vocab_size + 1)
        self.confidence_head = nn.Linear(hidden_size, 4)
        self.reinspect_head = nn.Linear(hidden_size, 2)
        self.correction_head = nn.Linear(hidden_size, 2)

    def decode(self, hidden: torch.Tensor) -> HigherOrderPrediction:
        return HigherOrderPrediction(
            hidden=hidden,
            report=self.report_head(hidden),
            confidence=self.confidence_head(hidden),
            reinspect=self.reinspect_head(hidden),
            correction=self.correction_head(hidden),
        )

    def forward(self, model_input: torch.Tensor) -> HigherOrderPrediction:
        return self.decode(self.encoder(model_input))


def behavior_loss(
    prediction: HigherOrderPrediction,
    batch: HigherOrderTensors,
) -> torch.Tensor:
    return sum(
        nn.functional.cross_entropy(logits, target)
        for logits, target in (
            (prediction.report, batch.report_target),
            (prediction.confidence, batch.confidence_target),
            (prediction.reinspect, batch.reinspect_target),
            (prediction.correction, batch.correction_target),
        )
    )


def train_higher_order_behavior_model(
    model: HigherOrderBehaviorModel,
    train: HigherOrderTensors,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    seed: int = 83,
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
            prediction = model(batch.model_input)
            loss = behavior_loss(prediction, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(data))
    return losses


@torch.no_grad()
def behavior_metrics(
    model: HigherOrderBehaviorModel,
    data: HigherOrderTensors,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    prediction = model(batch.model_input)
    metrics = {}
    for name, target in (
        ("report", batch.report_target),
        ("confidence", batch.confidence_target),
        ("reinspect", batch.reinspect_target),
        ("correction", batch.correction_target),
    ):
        metrics[f"{name}_accuracy"] = getattr(prediction, name).argmax(
            dim=-1
        ).eq(target).float().mean().item()
    return metrics


def fixed_capacity_lift(
    features: torch.Tensor,
    output_size: int,
    *,
    seed: int,
) -> torch.Tensor:
    if features.shape[1] == output_size:
        return features
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(
        features.shape[1], output_size, generator=generator
    ) / math.sqrt(features.shape[1])
    return features @ projection


class StatusProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(HIGHER_ORDER_STATUSES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def train_status_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    hidden_size: int = 64,
    steps: int = 300,
    learning_rate: float = 1e-2,
    seed: int = 97,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    probe = StatusProbe(train_features.shape[1], hidden_size)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate)
    probe.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(train_features), train_targets)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        train_accuracy = probe(train_features).argmax(dim=-1).eq(
            train_targets
        ).float().mean().item()
        test_predictions = probe(test_features).argmax(dim=-1)
        test_accuracy = test_predictions.eq(test_targets).float().mean().item()
        by_status = {}
        for index, status in enumerate(HIGHER_ORDER_STATUSES):
            mask = test_targets.eq(index)
            by_status[status] = test_predictions[mask].eq(
                test_targets[mask]
            ).float().mean().item()
    return {
        "train_accuracy": train_accuracy,
        "heldout_accuracy": test_accuracy,
        "heldout_by_status_accuracy": by_status,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in probe.parameters()
        ),
    }


@torch.no_grad()
def paired_wrong_access_intervention(
    model: HigherOrderBehaviorModel,
    examples: list[HigherOrderExample],
    config: HigherOrderConfig,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    groups: dict[str, dict[str, HigherOrderExample]] = {}
    for example in examples:
        groups.setdefault(example.counterbalance_group, {})[example.status] = example
    pairs = [
        (cases["wrong_access_lure"], cases["fresh_current"])
        for cases in groups.values()
        if cases["wrong_access_lure"].split == "heldout_content_status"
    ]
    if not pairs:
        raise ValueError("no held-out wrong-access pairs available")
    wrong_examples = [pair[0] for pair in pairs]
    current_examples = [pair[1] for pair in pairs]
    wrong = tensorize_higher_order_examples(wrong_examples, config).to(device)
    current = tensorize_higher_order_examples(current_examples, config).to(device)
    model.eval()
    wrong_prediction = model(wrong.model_input)
    current_hidden = model(current.model_input).hidden
    swapped = model.decode(current_hidden)
    wrong_confidence = wrong_prediction.confidence.argmax(dim=-1)
    swapped_confidence = swapped.confidence.argmax(dim=-1)
    return {
        "pair_count": float(len(pairs)),
        "first_order_content_held_fixed_rate": sum(
            wrong_case.content_key == current_case.content_key
            and wrong_case.content_value == current_case.content_value
            and wrong_case.current_observation_value
            == current_case.current_observation_value
            for wrong_case, current_case in pairs
        )
        / len(pairs),
        "confidence_increase_rate": swapped_confidence.gt(
            wrong_confidence
        ).float().mean().item(),
        "reinspection_turns_off_rate": swapped.reinspect.argmax(dim=-1).eq(
            0
        ).float().mean().item(),
        "correction_turns_off_rate": swapped.correction.argmax(dim=-1).eq(
            0
        ).float().mean().item(),
        "newly_accessible_report_content_accuracy": swapped.report.argmax(
            dim=-1
        ).eq(current.report_target).float().mean().item(),
        "mean_hidden_swap_l2": current_hidden.sub(wrong_prediction.hidden).norm(
            dim=-1
        ).mean().item(),
    }
