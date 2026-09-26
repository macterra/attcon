from __future__ import annotations

"""Action-trained memory and independently fitted paired-history reports.

Report labels never enter agent training. The assay measures frozen-state
decodability and causal coupling, not spontaneous introspection.
"""

from dataclasses import dataclass
import itertools
import random

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HistoryConfig:
    keys: int = 8
    values: int = 6
    positions: int = 4
    hidden: int = 64

    @property
    def input_size(self) -> int:
        return self.keys + self.values + 1


@dataclass(frozen=True)
class HistoryBatch:
    events: torch.Tensor
    value: torch.Tensor
    seen: torch.Tensor
    query: torch.Tensor
    group: torch.Tensor

    def __len__(self) -> int:
        return len(self.value)

    def subset(self, index: torch.Tensor) -> HistoryBatch:
        return HistoryBatch(**{k: v[index] for k, v in self.__dict__.items()})

    def report_labels(self, config: HistoryConfig) -> torch.Tensor:
        return torch.where(self.seen, self.value, config.values)


def make_history_splits(
    sizes: dict[str, int], *, seed: int, config: HistoryConfig = HistoryConfig()
) -> dict[str, HistoryBatch]:
    """Keep every content and availability variant in its context's split.

    Contexts are uniquely recoverable from the observed distractor and query,
    including record positions. Unseen cases have identical inputs for every
    hidden answer, making their answer distribution exactly uniform.
    """
    if config.keys < 3 or config.values < 2 or config.positions < 2:
        raise ValueError("requires >=3 keys, >=2 values, and >=2 record positions")
    contexts = list(itertools.product(
        range(config.keys), range(config.keys), range(config.values),
        range(config.positions), range(config.positions),
    ))
    contexts = [c for c in contexts if c[0] != c[1] and c[3] != c[4]]
    if any(size <= 0 for size in sizes.values()) or sum(sizes.values()) > len(contexts):
        raise ValueError("split sizes must be positive and fit the distinct contexts")
    rng = random.Random(seed)
    rng.shuffle(contexts)
    result = {}
    cursor = 0
    previous_inputs: set[bytes] = set()
    for name, size in sizes.items():
        rows, values, seen, queries, groups = [], [], [], [], []
        for group in range(cursor, cursor + size):
            query, distractor, distractor_value, target_pos, distractor_pos = contexts[group]
            foil = rng.choice([k for k in range(config.keys) if k not in (query, distractor)])
            foil_value = rng.randrange(config.values)
            # Two unseen contexts can exchange the roles of their historical
            # distractors. Reject actual cross-split duplicates, not just IDs.
            for attempt in range(1000):
                unseen_events = torch.zeros(config.positions + 2, config.input_size)
                unseen_events[distractor_pos, distractor] = 1
                unseen_events[distractor_pos, config.keys + distractor_value] = 1
                unseen_events[target_pos, foil] = 1
                unseen_events[target_pos, config.keys + foil_value] = 1
                unseen_events[-1, query] = 1
                unseen_events[-1, -1] = 1
                if unseen_events.numpy().tobytes() not in previous_inputs:
                    break
                foil = rng.choice([k for k in range(config.keys) if k not in (query, distractor)])
                foil_value = rng.randrange(config.values)
            else:
                raise ValueError("unable to construct distinct unseen history")
            for value in range(config.values):
                for available in (True, False):
                    events = torch.zeros(config.positions + 2, config.input_size)
                    events[distractor_pos, distractor] = 1
                    events[distractor_pos, config.keys + distractor_value] = 1
                    events[target_pos, query if available else foil] = 1
                    events[target_pos, config.keys + (value if available else foil_value)] = 1
                    events[-1, query] = 1
                    events[-1, -1] = 1  # query event, not an availability flag
                    rows.append(events)
                    values.append(value)
                    seen.append(available)
                    queries.append(query)
                    groups.append(group)
        result[name] = HistoryBatch(
            torch.stack(rows), torch.tensor(values), torch.tensor(seen),
            torch.tensor(queries), torch.tensor(groups),
        )
        previous_inputs.update(row.numpy().tobytes() for row in result[name].events)
        cursor += size
    return result


class HistoryAgent(nn.Module):
    def __init__(self, config: HistoryConfig = HistoryConfig()) -> None:
        super().__init__()
        self.memory = nn.GRU(config.input_size, config.hidden, batch_first=True)
        self.choice = nn.Linear(config.hidden, config.values)

    def state(self, events: torch.Tensor) -> torch.Tensor:
        return self.memory(events)[1][-1]

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.choice(self.state(events))


def train_agent(
    model: HistoryAgent, data: HistoryBatch, *, seed: int, epochs: int = 40,
) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for indices in torch.randperm(len(data), generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            # Only the environmental value choice trains the agent.
            loss = F.cross_entropy(model(data.events[indices]), data.value[indices])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        losses.append(total / len(data))
    model.eval()
    model.requires_grad_(False)
    return losses


def pad_features(features: torch.Tensor, width: int) -> torch.Tensor:
    if features.shape[1] > width:
        raise ValueError("probe width must accommodate every input without truncation")
    return F.pad(features.detach(), (0, width - features.shape[1]))


def fit_probe(
    features: torch.Tensor, labels: torch.Tensor, classes: int, *, seed: int,
    steps: int = 250,
) -> nn.Linear:
    torch.manual_seed(seed)
    probe = nn.Linear(features.shape[1], classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.03)
    features = features.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(features), labels)
        loss.backward()
        optimizer.step()
    probe.eval()
    probe.requires_grad_(False)
    return probe


@torch.no_grad()
def report_metrics(
    probe: nn.Module, features: torch.Tensor, data: HistoryBatch, config: HistoryConfig,
) -> dict[str, float]:
    predicted = probe(features).argmax(-1)
    correct = predicted == data.report_labels(config)
    seen_accuracy = correct[data.seen].float().mean().item()
    unseen_accuracy = correct[~data.seen].float().mean().item()
    # Generator puts the seen/unseen history for each value consecutively.
    paired = correct.reshape(-1, 2).all(dim=1).float().mean().item()
    return {
        "seen_value_accuracy": seen_accuracy,
        "unseen_unknown_accuracy": unseen_accuracy,
        "balanced_accuracy": (seen_accuracy + unseen_accuracy) / 2,
        "paired_accuracy": paired,
        "availability_balanced_accuracy": (
            (predicted[data.seen] != config.values).float().mean().item()
            + (predicted[~data.seen] == config.values).float().mean().item()
        ) / 2,
    }


@torch.no_grad()
def action_metrics(model: HistoryAgent, data: HistoryBatch) -> dict[str, float]:
    logits = model(data.events)
    probabilities = logits.softmax(-1)
    entropy = -(probabilities * logits.log_softmax(-1)).sum(-1)
    return {
        "seen_accuracy": (logits.argmax(-1)[data.seen] == data.value[data.seen]).float().mean().item(),
        "unseen_accuracy": (logits.argmax(-1)[~data.seen] == data.value[~data.seen]).float().mean().item(),
        "seen_entropy": entropy[data.seen].mean().item(),
        "unseen_entropy": entropy[~data.seen].mean().item(),
    }


def content_basis(
    states: torch.Tensor, data: HistoryBatch, config: HistoryConfig, *,
    permute_seed: int | None = None,
) -> torch.Tensor:
    labels = data.value[data.seen].clone()
    features = states[data.seen]
    if permute_seed is not None:
        labels = labels[torch.randperm(len(labels), generator=torch.Generator().manual_seed(permute_seed))]
    means = torch.stack([features[labels == v].mean(0) for v in range(config.values)])
    centered = means - means.mean(0)
    # SVD avoids retaining the extra arbitrary QR direction from rank deficiency.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:config.values - 1].T.contiguous()


def transplant(states: torch.Tensor, donors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Replace only the selected orthonormal subspace; preserve its complement."""
    return states + ((donors - states) @ basis) @ basis.T


@torch.no_grad()
def intervention_metrics(
    agent: HistoryAgent, reporter: nn.Module, query_probe: nn.Module,
    states: torch.Tensor, data: HistoryBatch, basis: torch.Tensor,
    config: HistoryConfig,
    *, norm_reference_basis: torch.Tensor | None = None,
) -> dict[str, float | int | None]:
    indices = torch.where(data.seen)[0]
    # Within each context, use the next distinct value as the donor.
    donor_indices = indices.reshape(-1, config.values).roll(-1, dims=1).reshape(-1)
    recipients, donors = states[indices], states[donor_indices]
    changed = transplant(recipients, donors, basis)
    if norm_reference_basis is not None:
        reference = transplant(recipients, donors, norm_reference_basis) - recipients
        delta = changed - recipients
        changed = recipients + delta * (reference.norm(dim=1, keepdim=True) / delta.norm(dim=1, keepdim=True).clamp_min(1e-12))
    target = data.value[donor_indices]
    action = agent.choice(changed).argmax(-1)
    report = reporter(changed).argmax(-1)
    joint = (action == target) & (report == target)
    eligible = (
        (agent.choice(recipients).argmax(-1) == data.value[indices])
        & (reporter(recipients).argmax(-1) == data.value[indices])
        & (agent.choice(donors).argmax(-1) == target)
        & (reporter(donors).argmax(-1) == target)
    )
    return {
        "count": len(indices),
        "action_donor_follow": (action == target).float().mean().item(),
        "report_donor_follow": (report == target).float().mean().item(),
        "joint_donor_follow": joint.float().mean().item(),
        "eligible_fraction": eligible.float().mean().item(),
        "eligible_joint_donor_follow": joint[eligible].float().mean().item() if eligible.any() else None,
        "report_access_stability": (report != config.values).float().mean().item(),
        "query_identity_accuracy_before": (query_probe(recipients).argmax(-1) == data.query[indices]).float().mean().item(),
        "query_identity_accuracy_after": (query_probe(changed).argmax(-1) == data.query[indices]).float().mean().item(),
        "query_identity_stability": (query_probe(changed).argmax(-1) == query_probe(recipients).argmax(-1)).float().mean().item(),
        "mean_state_change_norm": (changed - recipients).norm(dim=1).mean().item(),
    }


def split_checks(splits: dict[str, HistoryBatch], config: HistoryConfig) -> dict[str, bool]:
    groups = [set(data.group.tolist()) for data in splits.values()]
    disjoint = all(not a.intersection(b) for a, b in itertools.combinations(groups, 2))
    # Check actual input sequences too; disjoint IDs alone cannot rule out leakage.
    signatures = [set(row.numpy().tobytes() for row in data.events) for data in splits.values()]
    observations_disjoint = all(not a.intersection(b) for a, b in itertools.combinations(signatures, 2))
    matched = all(torch.equal(d.events[::2, -1], d.events[1::2, -1]) for d in splits.values())
    unseen_uniform = all(
        torch.equal(
            d.events[~d.seen].reshape(-1, config.values, *d.events.shape[1:]),
            d.events[~d.seen].reshape(-1, config.values, *d.events.shape[1:])[:, :1].expand(-1, config.values, -1, -1),
        ) for d in splits.values()
    )
    return {
        "context_groups_disjoint": disjoint,
        "input_sequences_disjoint": observations_disjoint,
        "paired_final_observations_identical": matched,
        "unseen_inputs_identical_across_answers": unseen_uniform,
    }
