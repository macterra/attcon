from __future__ import annotations

"""Task-only delay training and measurement utilities for regulation audits."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from attcon.history_reporting import HistoryConfig, pad_features, report_metrics
from attcon.report_calibration import ReadoutSelection, selection_score
from attcon.report_delay import insert_delay
from attcon.report_sufficiency import SequenceAgent


WIDTH = 128
DELAYS = (0, 1, 3, 6)


def agent_for(architecture: str) -> tuple[SequenceAgent, HistoryConfig]:
    if architecture not in ("gru", "rnn_matched"):
        raise ValueError("unsupported architecture")
    config = HistoryConfig(hidden=64 if architecture == "gru" else 115)
    return SequenceAgent(config, "gru" if architecture == "gru" else "rnn"), config


def weight_fingerprint(model: nn.Module) -> str:
    return hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()


def assigned_delays(data, seed: int) -> torch.Tensor:
    groups, inverse = data.group.unique(sorted=True, return_inverse=True)
    order = torch.randperm(len(groups), generator=torch.Generator().manual_seed(seed))
    assignment = torch.empty(len(groups), dtype=torch.long)
    assignment[order] = torch.arange(len(groups)) % len(DELAYS)
    return torch.tensor(DELAYS)[assignment[inverse]]


@torch.no_grad()
def mixed_states(agent, data, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    delays = assigned_delays(data, seed)
    states = torch.empty(len(data), agent.choice.in_features)
    for delay in DELAYS:
        indices = torch.where(delays == delay)[0]
        if len(indices):
            states[indices] = agent.state(insert_delay(data.subset(indices), delay).events)
    return states, delays


def train_delay_agent(agent, data, *, seed: int, recipe: str, epochs: int = 160) -> dict:
    if recipe not in ("fixed", "variable"):
        raise ValueError("unknown delay recipe")
    order_rng = torch.Generator().manual_seed(seed)
    delay_rng = torch.Generator().manual_seed(seed + 4242)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=0.003)
    losses, updates, recurrent_steps = [], 0, 0
    initial = weight_fingerprint(agent)
    agent.train()
    for _ in range(epochs):
        total = 0.0
        for indices in torch.randperm(len(data), generator=order_rng).split(256):
            delay = DELAYS[torch.randint(4, (), generator=delay_rng).item()] if recipe == "variable" else 0
            batch = insert_delay(data.subset(indices), delay)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(agent(batch.events), batch.value)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            updates += 1
            recurrent_steps += len(batch) * batch.events.shape[1]
        losses.append(total / len(data))
    agent.eval().requires_grad_(False)
    return {"losses": losses, "updates": updates, "recurrent_example_steps": recurrent_steps,
            "initial_sha256": initial, "final_sha256": weight_fingerprint(agent)}


@torch.no_grad()
def choice_metrics(agent, states, data) -> dict:
    logits = agent.choice(states)
    entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    return {"seen_accuracy": (logits.argmax(-1)[data.seen] == data.value[data.seen]).float().mean().item(),
            "unseen_accuracy": (logits.argmax(-1)[~data.seen] == data.value[~data.seen]).float().mean().item(),
            "seen_entropy": entropy[data.seen].mean().item(), "unseen_entropy": entropy[~data.seen].mean().item()}


class Readout(nn.Module):
    def __init__(self, features: torch.Tensor, classes: int = 7, hidden: int = 64):
        super().__init__()
        features = features.detach()
        self.register_buffer("center", features.mean(0))
        self.register_buffer("scale", features.std(0, unbiased=False).clamp_min(0.01))
        self.network = nn.Sequential(nn.Linear(WIDTH, hidden), nn.Tanh(), nn.Linear(hidden, classes))

    def forward(self, features):
        return self.network((features - self.center) / self.scale)


def fit_readout(features, target, *, seed, l2, steps=300, reward=False):
    torch.manual_seed(seed)
    model = Readout(features, 1 if reward else 7, 32 if reward else 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    features = features.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits[:, 0], target.float()) if reward else F.cross_entropy(logits, target)
        penalty = sum(layer.weight.square().sum() for layer in model.network if isinstance(layer, nn.Linear))
        (loss + 0.5 * l2 * penalty).backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


def select_report(features, labels, validation_features, validation, config, *, seed, steps=300):
    candidates, best, best_score = [], None, None
    for l2 in (0.0, 0.001):
        model = fit_readout(features, labels, seed=seed, l2=l2, steps=steps)
        metrics = report_metrics(model, validation_features, validation, config)
        candidate = {"l2": l2, "validation": metrics}
        candidates.append(candidate)
        score = selection_score(metrics)
        if best_score is None or score > best_score:
            best, best_score = (model, candidate), score
    return ReadoutSelection(best[0], best[1], candidates)


class StateReporter(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.readout = readout

    def forward(self, states):
        return self.readout(pad_features(states, WIDTH))


def confidence_features(logits):
    probabilities = logits.softmax(-1)
    entropy = -(probabilities * logits.log_softmax(-1)).sum(-1) / math.log(logits.shape[1])
    return pad_features(torch.stack((probabilities.max(-1).values, entropy), dim=-1), WIDTH)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    agent, config = agent_for(saved["settings"]["architecture"])
    agent.load_state_dict(saved["agent"])
    agent.eval().requires_grad_(False)
    readouts = {}
    for name, weights in saved["readouts"].items():
        model = Readout(torch.zeros(2, WIDTH))
        model.load_state_dict(weights)
        readouts[name] = model.eval().requires_grad_(False)
    return saved, agent, config, readouts
