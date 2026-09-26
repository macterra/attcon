from __future__ import annotations

"""Validation-selected readouts of a frozen action-trained state."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from attcon.history_reporting import HistoryBatch, HistoryConfig, pad_features, report_metrics


L2_GRID = (0.0, 0.0001, 0.001, 0.01)


class NormalizedReadout(nn.Module):
    def __init__(self, fit_features: torch.Tensor, classes: int, standardize: bool) -> None:
        super().__init__()
        features = fit_features.detach()
        self.register_buffer("center", features.mean(0) if standardize else torch.zeros(features.shape[1]))
        self.register_buffer("scale", features.std(0, unbiased=False).clamp_min(0.01)
                             if standardize else torch.ones(features.shape[1]))
        self.linear = nn.Linear(features.shape[1], classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear((features - self.center) / self.scale)


def fit_readout(
    features: torch.Tensor, labels: torch.Tensor, classes: int, *,
    standardize: bool, l2: float, seed: int, steps: int = 500,
) -> NormalizedReadout:
    torch.manual_seed(seed)
    model = NormalizedReadout(features, classes, standardize)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    features = features.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), labels) + 0.5 * l2 * model.linear.weight.square().sum()
        loss.backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


def selection_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        min(metrics["seen_value_accuracy"], metrics["unseen_unknown_accuracy"]),
        metrics["paired_accuracy"], metrics["balanced_accuracy"],
    )


@dataclass
class ReadoutSelection:
    model: nn.Module
    selected: dict
    candidates: list[dict]


def select_readout(
    fit_features: torch.Tensor, fit_labels: torch.Tensor,
    validation_features: torch.Tensor, validation: HistoryBatch,
    config: HistoryConfig, *, seed: int, steps: int = 500,
) -> ReadoutSelection:
    """No test inputs are accepted by this API; normalization uses fitting only."""
    candidates = []
    best = None
    best_score = None
    for standardize in (False, True):
        for l2 in L2_GRID:
            model = fit_readout(fit_features, fit_labels, config.values + 1,
                                standardize=standardize, l2=l2, seed=seed, steps=steps)
            metrics = report_metrics(model, validation_features, validation, config)
            candidate = {"standardize": standardize, "l2": l2, "validation": metrics}
            candidates.append(candidate)
            score = selection_score(metrics)
            if best_score is None or score > best_score:
                best = (model, candidate)
                best_score = score
    assert best is not None
    return ReadoutSelection(best[0], best[1], candidates)


class ActionScoreReporter(nn.Module):
    """Apply a fitted score readout to states without retaining a mutable agent."""

    def __init__(self, choice: nn.Linear, readout: nn.Module, width: int) -> None:
        super().__init__()
        self.register_buffer("choice_weight", choice.weight.detach().clone())
        self.register_buffer("choice_bias", choice.bias.detach().clone())
        self.readout = readout
        self.width = width

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        scores = F.linear(states, self.choice_weight, self.choice_bias)
        return self.readout(pad_features(scores, self.width))


class EntropyReporter(nn.Module):
    def __init__(self, choice: nn.Linear, threshold: float) -> None:
        super().__init__()
        self.register_buffer("choice_weight", choice.weight.detach().clone())
        self.register_buffer("choice_bias", choice.bias.detach().clone())
        self.register_buffer("threshold", torch.tensor(threshold))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = F.linear(states, self.choice_weight, self.choice_bias)
        probabilities = logits.softmax(-1)
        entropy = -(probabilities * logits.log_softmax(-1)).sum(-1) / math.log(logits.shape[1])
        answer = torch.where(entropy > self.threshold, logits.shape[1], logits.argmax(-1))
        return F.one_hot(answer, logits.shape[1] + 1).float()


def select_entropy_reporter(
    choice: nn.Linear, validation_states: torch.Tensor,
    validation: HistoryBatch, config: HistoryConfig,
) -> ReadoutSelection:
    best = None
    best_score = None
    candidates = []
    for index in range(101):
        threshold = index / 100
        model = EntropyReporter(choice, threshold)
        metrics = report_metrics(model, validation_states, validation, config)
        candidate = {"threshold": threshold, "validation": metrics}
        candidates.append(candidate)
        score = selection_score(metrics)
        if best_score is None or score > best_score:
            best = (model, candidate)
            best_score = score
    assert best is not None
    return ReadoutSelection(best[0], best[1], candidates)
