from __future__ import annotations

"""Models and measurements for the Branch C object-binding pilot."""

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable

import torch
from torch import nn

from attcon.binding import BindingConfig, BindingExample


@dataclass(frozen=True)
class BindingTensors:
    attributes: torch.Tensor
    cues: torch.Tensor
    target_index: torch.Tensor
    location: torch.Tensor
    visible_type: torch.Tensor
    digit: torch.Tensor
    inspected: torch.Tensor
    lure_location: torch.Tensor
    lure_visible_type: torch.Tensor
    lure_digit: torch.Tensor
    lure_inspected: torch.Tensor

    def to(self, device: torch.device | str) -> BindingTensors:
        return BindingTensors(
            **{
                name: value.to(device)
                for name, value in self.__dict__.items()
            }
        )

    def subset(self, indices: torch.Tensor) -> BindingTensors:
        return BindingTensors(
            **{
                name: value[indices]
                for name, value in self.__dict__.items()
            }
        )

    def __len__(self) -> int:
        return self.attributes.shape[0]


@dataclass(frozen=True)
class BindingPrediction:
    location: torch.Tensor
    visible_type: torch.Tensor
    digit: torch.Tensor
    inspected: torch.Tensor
    attention: torch.Tensor | None = None


def attribute_slices(config: BindingConfig) -> dict[str, slice]:
    start = 0
    result = {}
    for name, width in (
        ("location", config.num_cells),
        ("visible_type", config.num_visible_types),
        ("digit", config.digit_vocab_size),
        ("cue_tag", config.num_cues),
        ("inspected", 2),
    ):
        result[name] = slice(start, start + width)
        start += width
    return result


def attribute_dim(config: BindingConfig) -> int:
    return (
        config.num_cells
        + config.num_visible_types
        + config.digit_vocab_size
        + config.num_cues
        + 2
    )


def tensorize_binding_examples(
    examples: Iterable[BindingExample],
    config: BindingConfig,
) -> BindingTensors:
    cases = list(examples)
    if not cases:
        raise ValueError("at least one binding example is required")
    attributes = torch.zeros(
        len(cases), config.num_objects, attribute_dim(config), dtype=torch.float32
    )
    cues = torch.zeros(len(cases), config.num_cues, dtype=torch.float32)
    slices = attribute_slices(config)
    for row, example in enumerate(cases):
        if len(example.objects) != config.num_objects:
            raise ValueError("example object count does not match config")
        cues[row, example.cue] = 1.0
        for column, obj in enumerate(example.objects):
            attributes[row, column, slices["location"].start + obj.location] = 1.0
            attributes[row, column, slices["visible_type"].start + obj.visible_type] = 1.0
            attributes[row, column, slices["digit"].start + obj.digit] = 1.0
            attributes[row, column, slices["cue_tag"].start + obj.cue_tag] = 1.0
            attributes[row, column, slices["inspected"].start + int(obj.inspected)] = 1.0
    return BindingTensors(
        attributes=attributes,
        cues=cues,
        target_index=torch.tensor([case.target_index for case in cases]),
        location=torch.tensor([case.target.location for case in cases]),
        visible_type=torch.tensor([case.target.visible_type for case in cases]),
        digit=torch.tensor([case.target.digit for case in cases]),
        inspected=torch.tensor([int(case.target.inspected) for case in cases]),
        lure_location=torch.tensor([case.false_binding_lure.location for case in cases]),
        lure_visible_type=torch.tensor(
            [case.false_binding_lure.visible_type for case in cases]
        ),
        lure_digit=torch.tensor([case.false_binding_lure.digit for case in cases]),
        lure_inspected=torch.tensor(
            [int(case.false_binding_lure.inspected) for case in cases]
        ),
    )


class _BindingHeads(nn.Module):
    def __init__(self, hidden_size: int, config: BindingConfig) -> None:
        super().__init__()
        self.location = nn.Linear(hidden_size, config.num_cells)
        self.visible_type = nn.Linear(hidden_size, config.num_visible_types)
        self.digit = nn.Linear(hidden_size, config.digit_vocab_size)
        self.inspected = nn.Linear(hidden_size, 2)

    def forward(self, state: torch.Tensor) -> BindingPrediction:
        return BindingPrediction(
            location=self.location(state),
            visible_type=self.visible_type(state),
            digit=self.digit(state),
            inspected=self.inspected(state),
        )


class SharedSelectionBindingModel(nn.Module):
    """Route every predicted attribute through one shared object selection."""

    def __init__(self, config: BindingConfig, hidden_size: int = 64) -> None:
        super().__init__()
        self.config = config
        input_size = attribute_dim(config) + config.num_cues
        self.selector = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.bound_encoder = nn.Sequential(
            nn.Linear(attribute_dim(config), hidden_size),
            nn.Tanh(),
        )
        self.heads = _BindingHeads(hidden_size, config)

    def forward(
        self, attributes: torch.Tensor, cues: torch.Tensor
    ) -> BindingPrediction:
        expanded_cue = cues[:, None, :].expand(-1, attributes.shape[1], -1)
        scores = self.selector(torch.cat((attributes, expanded_cue), dim=-1)).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        bound = torch.einsum("bo,boa->ba", attention, attributes)
        prediction = self.heads(self.bound_encoder(bound))
        return BindingPrediction(
            location=prediction.location,
            visible_type=prediction.visible_type,
            digit=prediction.digit,
            inspected=prediction.inspected,
            attention=attention,
        )


class IndependentFeatureBaseline(nn.Module):
    """Predict from a bag of feature frequencies with object identity removed."""

    def __init__(self, config: BindingConfig, hidden_size: int = 64) -> None:
        super().__init__()
        self.config = config
        input_size = attribute_dim(config) + config.num_cues
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.heads = _BindingHeads(hidden_size, config)

    def forward(
        self, attributes: torch.Tensor, cues: torch.Tensor
    ) -> BindingPrediction:
        pooled = attributes.mean(dim=1)
        return self.heads(self.encoder(torch.cat((pooled, cues), dim=-1)))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def binding_loss(prediction: BindingPrediction, batch: BindingTensors) -> torch.Tensor:
    loss = torch.zeros((), device=prediction.location.device)
    for name in ("location", "visible_type", "digit", "inspected"):
        loss = loss + nn.functional.cross_entropy(
            getattr(prediction, name), getattr(batch, name)
        )
    return loss


def train_binding_model(
    model: nn.Module,
    train: BindingTensors,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    seed: int = 31,
    device: str = "cpu",
) -> list[float]:
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    data = train.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        order = torch.randperm(len(data), generator=generator)
        for start in range(0, len(data), batch_size):
            indices = order[start : start + batch_size].to(device)
            batch = data.subset(indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.attributes, batch.cues)
            loss = binding_loss(prediction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        losses.append(epoch_loss / len(data))
    return losses


def _predicted_labels(prediction: BindingPrediction) -> dict[str, torch.Tensor]:
    return {
        name: getattr(prediction, name).argmax(dim=-1)
        for name in ("location", "visible_type", "digit", "inspected")
    }


def _tuple_log_score(
    prediction: BindingPrediction,
    labels: dict[str, torch.Tensor],
) -> torch.Tensor:
    score = torch.zeros(prediction.location.shape[0], device=prediction.location.device)
    for name in ("location", "visible_type", "digit", "inspected"):
        logits = getattr(prediction, name)
        score += torch.log_softmax(logits, dim=-1).gather(
            1, labels[name][:, None]
        ).squeeze(1)
    return score


@torch.no_grad()
def evaluate_binding_model(
    model: nn.Module,
    data: BindingTensors,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    model.eval()
    batch = data.to(device)
    prediction = model(batch.attributes, batch.cues)
    predicted = _predicted_labels(prediction)
    correct = {
        name: predicted[name].eq(getattr(batch, name))
        for name in predicted
    }
    true_labels = {name: getattr(batch, name) for name in predicted}
    lure_labels = {name: getattr(batch, f"lure_{name}") for name in predicted}
    joint = torch.stack(tuple(correct.values()), dim=-1).all(dim=-1)
    result: dict[str, Any] = {
        "count": len(data),
        "field_accuracy": {
            name: values.float().mean().item() for name, values in correct.items()
        },
        "joint_accuracy": joint.float().mean().item(),
        "false_binding_lure_rejection": (
            _tuple_log_score(prediction, true_labels)
            > _tuple_log_score(prediction, lure_labels)
        ).float().mean().item(),
    }
    if prediction.attention is not None:
        selected = prediction.attention.argmax(dim=-1)
        target_attention = prediction.attention.gather(
            1, batch.target_index[:, None]
        ).squeeze(1)
        result["target_selection_accuracy"] = selected.eq(
            batch.target_index
        ).float().mean().item()
        result["mean_target_attention"] = target_attention.mean().item()
    return result


def _replace_type(
    attributes: torch.Tensor,
    rows: torch.Tensor,
    object_indices: torch.Tensor,
    new_types: torch.Tensor,
    config: BindingConfig,
) -> torch.Tensor:
    changed = attributes.clone()
    type_slice = attribute_slices(config)["visible_type"]
    changed[rows, object_indices, type_slice] = 0.0
    changed[rows, object_indices, type_slice.start + new_types] = 1.0
    return changed


@torch.no_grad()
def binding_intervention_metrics(
    model: nn.Module,
    data: BindingTensors,
    config: BindingConfig,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    rows = torch.arange(len(batch), device=device)
    original = model(batch.attributes, batch.cues)
    original_labels = _predicted_labels(original)

    new_target_types = (batch.visible_type + 1) % config.num_visible_types
    target_changed_attributes = _replace_type(
        batch.attributes,
        rows,
        batch.target_index,
        new_target_types,
        config,
    )
    target_changed = model(target_changed_attributes, batch.cues)
    target_labels = _predicted_labels(target_changed)
    stable_fields = ("location", "digit", "inspected")
    stable = torch.stack(
        [target_labels[name].eq(original_labels[name]) for name in stable_fields],
        dim=-1,
    )

    non_target_index = (batch.target_index + 1) % config.num_objects
    type_slice = attribute_slices(config)["visible_type"]
    non_target_type = batch.attributes[rows, non_target_index, type_slice].argmax(dim=-1)
    new_non_target_type = (non_target_type + 1) % config.num_visible_types
    non_target_attributes = _replace_type(
        batch.attributes,
        rows,
        non_target_index,
        new_non_target_type,
        config,
    )
    non_target_changed = model(non_target_attributes, batch.cues)
    non_target_labels = _predicted_labels(non_target_changed)
    non_target_stable = torch.stack(
        [
            non_target_labels[name].eq(original_labels[name])
            for name in original_labels
        ],
        dim=-1,
    )
    result = {
        "target_type_follow_rate": target_labels["visible_type"].eq(
            new_target_types
        ).float().mean().item(),
        "target_other_field_mean_stability": stable.float().mean().item(),
        "target_other_field_joint_stability": stable.all(dim=-1).float().mean().item(),
        "non_target_all_field_invariance": non_target_stable.all(
            dim=-1
        ).float().mean().item(),
    }
    if original.attention is not None and target_changed.attention is not None:
        result["target_selection_stability"] = target_changed.attention.argmax(
            dim=-1
        ).eq(original.attention.argmax(dim=-1)).float().mean().item()
    return result


def wilson_lower_bound(success_rate: float, count: int) -> float:
    """Wilson 95% lower bound for compact audit uncertainty reporting."""

    if count < 1:
        raise ValueError("count must be positive")
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    centre = success_rate + z * z / (2.0 * count)
    radius = z * math.sqrt(
        success_rate * (1.0 - success_rate) / count + z * z / (4.0 * count * count)
    )
    return (centre - radius) / denominator
