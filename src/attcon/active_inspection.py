"""Finite-horizon environmental acquisition; no access labels train controllers."""
from dataclasses import dataclass
import hashlib
import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

VALUES = 6
COSTS = (0.1, 0.25, 0.4)
CONDITIONS = ('fresh', 'stale', 'missing')


@dataclass
class Episodes:
    context: torch.Tensor
    value: torch.Tensor
    condition: torch.Tensor
    cost: torch.Tensor
    old: torch.Tensor
    sample: torch.Tensor
    group: torch.Tensor

    def __len__(self):
        return len(self.value)

    def subset(self, indices):
        return Episodes(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})

    def fingerprint(self):
        return hashlib.sha256(b''.join(getattr(self, name).numpy().tobytes() for name in self.__dataclass_fields__)).hexdigest()


def make_splits(seed, reliability=0.75):
    if not 0 <= reliability <= 1:
        raise ValueError('invalid sensor reliability')
    contexts = torch.tensor(list(itertools.product(range(VALUES), repeat=4)))
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(contexts), generator=rng)
    old = torch.randint(VALUES, (len(contexts),), generator=rng)
    uniform = torch.rand(len(contexts), VALUES, generator=rng)
    offset = torch.randint(1, VALUES, (len(contexts), VALUES), generator=rng)
    splits, start = {}, 0
    variants = torch.tensor(list(itertools.product(range(VALUES), range(3), range(3))))
    for name, count in (('train', 128), ('validation', 32), ('report_fit', 64), ('test', 64), ('stress', 64)):
        groups = order[start:start + count].repeat_interleave(len(variants))
        value, condition, cost_index = variants.repeat(count, 1).T
        sample = torch.where(uniform[groups, value] < reliability, value, (value + offset[groups, value]) % VALUES)
        splits[name] = Episodes(contexts[groups], value, condition, torch.tensor(COSTS)[cost_index], old[groups], sample, groups)
        start += count
    return splits


def query_event(data, budget):
    event = torch.zeros(len(data), 1, 12)
    event[:, 0, 10] = data.cost
    event[:, 0, 11] = budget / 2
    return event


def initial_events(data, delay=1):
    if delay < 0:
        raise ValueError('negative delay')
    events = torch.zeros(len(data), 7 + delay, 12)
    events[:, :4, :6] = F.one_hot(data.context, VALUES).float()
    events[:, :4, 6] = 1
    value = torch.where(data.condition == 0, data.value, data.old)
    events[:, 4, :6] = F.one_hot(value, VALUES).float() * (data.condition != 2)[:, None]
    events[:, 5, 7] = (data.condition == 0).float()
    events[:, 5, 8] = (data.condition == 1).float()
    events[:, -1:] = query_event(data, 2)
    return events


def acquired_events(data, stage, delay=1):
    if stage not in (1, 2) or delay < 0:
        raise ValueError('invalid acquisition stage/delay')
    events = torch.zeros(len(data), 2 + delay, 12)
    events[:, 0, :6] = F.one_hot(data.sample if stage == 1 else data.value, VALUES).float()
    events[:, 0, 9] = 1
    events[:, -1:] = query_event(data, 2 - stage)
    return events


class AcquisitionAgent(nn.Module):
    def __init__(self, family='state'):
        super().__init__()
        if family not in ('state', 'action', 'confidence'):
            raise ValueError('unknown controller family')
        self.family = family
        self.recurrent = nn.GRU(12, 48, batch_first=True)
        self.answer = nn.Linear(48, VALUES)
        self.inspection = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))

    def advance(self, events, state=None):
        _, hidden = self.recurrent(events, None if state is None else state[None])
        return hidden[0]

    def values(self, state, cost, budget):
        logits = self.answer(state)
        probability = logits.softmax(-1)
        if self.family == 'state':
            feature = state
        elif self.family == 'action':
            feature = logits
        else:
            entropy = -(probability * logits.log_softmax(-1)).sum(-1) / math.log(VALUES)
            feature = torch.stack((probability.max(-1).values, entropy), dim=1)
        feature = F.pad(torch.cat((feature, cost[:, None], torch.full_like(cost[:, None], budget / 2)), dim=1), (0, 128 - feature.shape[1] - 2))
        inspect = self.inspection(feature)[:, 0]
        return logits, inspect


def all_states(agent, data, delay=1):
    states = [agent.advance(initial_events(data, delay))]
    for stage in (1, 2):
        states.append(agent.advance(acquired_events(data, stage, delay), states[-1]))
    return states


@torch.no_grad()
def rollout(agent, data, delay=1, forced=None, initial_state=None):
    if forced not in (None, 0, 1, 2):
        raise ValueError('invalid fixed stopping time')
    state = agent.advance(initial_events(data, delay)) if initial_state is None else initial_state.clone()
    active = torch.ones(len(data), dtype=torch.bool)
    count = torch.zeros(len(data), dtype=torch.long)
    answer = torch.full_like(count, -1)
    visited = []
    for stage in (0, 1, 2):
        visited.append(active.clone())
        logits, inspection = agent.values(state, data.cost, 2 - stage)
        want = inspection > logits.softmax(-1).max(-1).values if forced is None else torch.full_like(active, stage < forced)
        buy = active & want & (stage < 2)
        stop = active & ~buy
        answer[stop] = logits.argmax(-1)[stop]
        count += buy.long()
        active = buy
        if stage < 2:
            state = agent.advance(acquired_events(data, stage + 1, delay), state)
    correct = (answer == data.value).float()
    return {'return': correct - data.cost * count, 'correct': correct, 'inspections': count, 'answer': answer, 'visited': torch.stack(visited)}


def analytic_policy(data, reliability=0.75):
    """Bayes policy using known reliability; never consult hidden value to decide."""
    # Fresh histories retain certainty. Uninformed histories can buy a noisy sample,
    # then either answer with its known reliability or buy a perfect observation.
    best_after_sample = torch.maximum(torch.full_like(data.cost, reliability), 1 - data.cost)
    inspect_first = (data.condition != 0) & (-data.cost + best_after_sample > 1 / VALUES)
    inspect_second = inspect_first & (1 - data.cost > reliability)
    count = inspect_first.long() + inspect_second.long()
    prediction = torch.where(data.condition == 0, data.value, torch.zeros_like(data.value))
    prediction = torch.where(inspect_first, data.sample, prediction)
    prediction = torch.where(inspect_second, data.value, prediction)
    correct = (prediction == data.value).float()
    expected = torch.where(data.condition == 0, torch.ones_like(data.cost), torch.maximum(torch.full_like(data.cost, 1 / VALUES), -data.cost + best_after_sample))
    return {'return': correct - data.cost * count, 'correct': correct, 'inspections': count, 'expected_return': expected}


def verified_labels(data, stage):
    verified = (data.condition == 0) | (stage == 2)
    return torch.where(verified, data.value, torch.full_like(data.value, VALUES))
