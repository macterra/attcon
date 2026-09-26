from __future__ import annotations

"""Matched nonlinear measurement instruments for task-trained memory."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from attcon.history_reporting import HistoryAgent, HistoryBatch, HistoryConfig, make_history_splits, pad_features, report_metrics
from attcon.report_calibration import ReadoutSelection, selection_score


WIDTH = 96


class SequenceAgent(HistoryAgent):
    def __init__(self, config: HistoryConfig, architecture: str = "gru") -> None:
        super().__init__(config)
        if architecture == "rnn":
            self.memory = nn.RNN(config.input_size, config.hidden, batch_first=True)
        elif architecture != "gru":
            raise ValueError(f"unsupported architecture: {architecture}")


def make_sufficiency_splits(seed: int, fit_groups: int = 128) -> dict[str, HistoryBatch]:
    if not 1 <= fit_groups <= 1024:
        raise ValueError("reporter fitting groups must be in [1, 1024]")
    splits = make_history_splits(
        {"agent_train": 1024, "validation": 64, "test": 128, "report_pool": 1024}, seed=seed,
    )
    pool = splits.pop("report_pool")
    groups = pool.group.unique(sorted=True)[:fit_groups]
    splits["report_fit"] = pool.subset(torch.where(torch.isin(pool.group, groups))[0])
    return splits


def batch_fingerprint(data: HistoryBatch) -> str:
    return hashlib.sha256(b"".join(tensor.numpy().tobytes() for tensor in data.__dict__.values())).hexdigest()


@torch.no_grad()
def history_oracle(events: torch.Tensor, config: HistoryConfig) -> torch.Tensor:
    """Information upper bound, with no access to target labels or status."""
    query = events[:, -1, :config.keys].argmax(-1)
    history = events[:, :-1]
    matching = history[:, :, :config.keys].gather(2, query[:, None, None].expand(-1, history.shape[1], 1)).squeeze(-1) > 0
    has_value = history[:, :, config.keys:config.keys + config.values].sum(-1) > 0
    matching &= has_value
    times = torch.arange(history.shape[1]).expand(len(events), -1)
    last = torch.where(matching, times, -1).max(-1).values
    values = history[torch.arange(len(events)), last.clamp_min(0), config.keys:config.keys + config.values].argmax(-1)
    return torch.where(last >= 0, values, config.values)


class NonlinearReadout(nn.Module):
    def __init__(self, fit_features: torch.Tensor, classes: int = 7) -> None:
        super().__init__()
        features = fit_features.detach()
        self.register_buffer("center", features.mean(0))
        self.register_buffer("scale", features.std(0, unbiased=False).clamp_min(0.01))
        self.network = nn.Sequential(nn.Linear(WIDTH, 64), nn.Tanh(), nn.Linear(64, classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network((features - self.center) / self.scale)


class StateInputReporter(nn.Module):
    def __init__(self, readout: nn.Module) -> None:
        super().__init__()
        self.readout = readout

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.readout(pad_features(states, WIDTH))


def select_nonlinear(
    fit_features: torch.Tensor, labels: torch.Tensor,
    validation_features: torch.Tensor, validation: HistoryBatch,
    config: HistoryConfig, *, seed: int, steps: int = 300,
) -> ReadoutSelection:
    candidates, best, best_score = [], None, None
    for l2 in (0.0, 0.001):
        torch.manual_seed(seed)
        model = NonlinearReadout(fit_features, config.values + 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        features = fit_features.detach()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            penalty = sum(module.weight.square().sum() for module in model.network if isinstance(module, nn.Linear))
            loss = F.cross_entropy(model(features), labels) + 0.5 * l2 * penalty
            loss.backward()
            optimizer.step()
        model.eval().requires_grad_(False)
        metrics = report_metrics(model, validation_features, validation, config)
        candidate = {"l2": l2, "validation": metrics}
        candidates.append(candidate)
        score = selection_score(metrics)
        if best_score is None or score > best_score:
            best, best_score = (model, candidate), score
    assert best is not None
    return ReadoutSelection(best[0], best[1], candidates)


def load_sufficiency_checkpoint(path: str) -> tuple[dict, SequenceAgent, dict[str, nn.Module], nn.Module]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = HistoryConfig(**saved["config"])
    agent = SequenceAgent(config, saved["settings"]["architecture"])
    agent.load_state_dict(saved["agent"])
    agent.eval().requires_grad_(False)
    reporters = {}
    for name, state in saved["readouts"].items():
        reporter = NonlinearReadout(torch.zeros(2, WIDTH), config.values + 1)
        reporter.load_state_dict(state)
        reporters[name] = reporter.eval().requires_grad_(False)
    query = nn.Linear(config.hidden, config.keys)
    query.load_state_dict(saved["query_probe"])
    return saved, agent, reporters, query.eval().requires_grad_(False)
