"""Prospective quality, serial verification, and alternative sensor routing."""
from dataclasses import dataclass
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from attcon.active_inspection import make_splits as base_splits, initial_events as base_events, COSTS, CONDITIONS


@dataclass
class ProspectiveBatch:
    base: object
    quality: torch.Tensor
    sample_a: torch.Tensor
    sample_b: torch.Tensor
    actual_a: torch.Tensor
    actual_b: torch.Tensor
    task: str

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def subset(self, indices):
        return ProspectiveBatch(self.base.subset(indices), *(getattr(self, name)[indices] for name in ('quality','sample_a','sample_b','actual_a','actual_b')), self.task)

    def fingerprint(self):
        payload = self.base.fingerprint().encode() + self.task.encode()
        payload += b''.join(getattr(self, name).numpy().tobytes() for name in ('quality','sample_a','sample_b','actual_a','actual_b'))
        return hashlib.sha256(payload).hexdigest()


def make_splits(seed, task, degradation=0.):
    if task not in ('serial', 'routing') or not 0 <= degradation <= .15:
        raise ValueError('invalid task or degradation')
    rng = torch.Generator().manual_seed(seed + 12000)
    uniforms = torch.rand(2, 1296, 6, generator=rng)
    offsets = torch.randint(1, 6, (2, 1296, 6), generator=rng)
    result = {}
    for name, original in base_splits(seed).items():
        data = original.subset(torch.arange(len(original)).repeat_interleave(2))
        quality = torch.tensor([.55, .90]).repeat(len(original))
        actual_a, actual_b = quality - degradation, 1.45 - quality - degradation
        samples = [torch.where(uniforms[i, data.group, data.value] < q, data.value, (data.value + offsets[i, data.group, data.value]) % 6) for i, q in enumerate((actual_a, actual_b))]
        result[name] = ProspectiveBatch(data, quality, *samples, actual_a, actual_b, task)
    return result


def root_events(data, delay=1):
    base = base_events(data.base, delay)
    events = torch.zeros(len(data), base.shape[1] + 1, 14)
    events[:, :4, :12] = base[:, :4]
    events[:, 4, 12] = data.quality
    events[:, 5:, :12] = base[:, 4:]
    events[:, -1, 11] = 1 if data.task == 'serial' else .5
    return events


def acquired_events(data, branch, delay=1):
    if branch not in (1, 2):
        raise ValueError('invalid branch')
    value = data.sample_a if branch == 1 else (data.value if data.task == 'serial' else data.sample_b)
    event = torch.zeros(len(data), 2 + delay, 14)
    event[:, 0, :6] = F.one_hot(value, 6).float()
    event[:, 0, 9] = 1
    event[:, 0, 13] = (1 if branch == 1 else -1) if data.task == 'routing' else 0
    event[:, -1, 10] = data.cost
    event[:, -1, 11] = .5 if data.task == 'serial' and branch == 1 else 0
    return event


class ProspectiveAgent(nn.Module):
    def __init__(self, architecture='gru', family='state'):
        super().__init__()
        if architecture not in ('gru','rnn') or family not in ('state','blind','cue'):
            raise ValueError('invalid model')
        self.architecture, self.family = architecture, family
        self.hidden = 48 if architecture == 'gru' else 87
        self.recurrent = (nn.GRU if architecture == 'gru' else nn.RNN)(14, self.hidden, batch_first=True)
        self.answer = nn.Linear(self.hidden, 6)
        self.inspection = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 2))

    def advance(self, events, state=None):
        _, hidden = self.recurrent(events, None if state is None else state[None])
        return hidden[0]

    def values(self, state, data, budget):
        logits = self.answer(state)
        probabilities = logits.softmax(-1)
        entropy = -(probabilities * logits.log_softmax(-1)).sum(-1) / math.log(6)
        feature = state if self.family == 'state' else torch.stack((probabilities.max(-1).values, entropy), 1)
        extra = [data.cost[:, None], torch.full_like(data.cost[:, None], budget / 2)]
        if self.family == 'cue':
            extra.append(data.quality[:, None])
        feature = torch.cat([feature] + extra, 1)
        inspection = self.inspection(F.pad(feature, (0, 128 - feature.shape[1])))
        return logits, inspection


def states_for(agent, data, delay=1, initial_state=None):
    root = agent.advance(root_events(data, delay)) if initial_state is None else initial_state
    first = agent.advance(acquired_events(data, 1, delay), root)
    second = agent.advance(acquired_events(data, 2, delay), first if data.task == 'serial' else root)
    return [root, first, second]


def budget_for(task, node):
    return 2 - node if task == 'serial' else int(node == 0)


def action_values(agent, state, data, node, native=False):
    budget = budget_for(data.task, node)
    logits, inspect = agent.values(state, data, budget)
    decline = torch.full_like(data.cost[:, None], .30 if native else -1e9)
    values = torch.cat((logits.softmax(-1), decline, inspect), 1)
    if budget == 0:
        values[:, 7:] = -1e9
    elif data.task == 'serial':
        values[:, 8] = -1e9
    return logits, values


@torch.no_grad()
def rollout(agent, data, delay=1, forced=None, initial_state=None, native=False):
    states = states_for(agent, data, delay, initial_state)
    decisions = [action_values(agent, state, data, node, native)[1].argmax(-1) for node, state in enumerate(states)]
    if forced is not None:
        decisions[0] = decisions[0].clamp_max(6) if forced == 0 else torch.full_like(decisions[0], 7 if forced == 1 or data.task == 'serial' else 8)
        # Fixed policies still choose their best answer, never reinterpret an inspection as decline.
        if forced == 0:
            decisions[0] = action_values(agent, states[0], data, 2 if data.task == 'serial' else 1, native)[1].argmax(-1)
        if data.task == 'serial':
            decisions[1] = action_values(agent, states[1], data, 2, native)[1].argmax(-1) if forced == 1 else (torch.full_like(decisions[1], 7) if forced == 2 else decisions[1])
    initial = decisions[0]
    inspect = initial >= 7
    count = inspect.long()
    answer = initial.clone()
    visited = torch.zeros(3, len(data), dtype=torch.bool)
    visited[0] = True
    if data.task == 'serial':
        visited[1] = inspect
        answer = torch.where(inspect, decisions[1], answer)
        second = inspect & (decisions[1] == 7)
        count += second.long()
        visited[2] = second
        answer = torch.where(second, decisions[2], answer)
    else:
        visited[1], visited[2] = initial == 7, initial == 8
        answer = torch.where(initial == 7, decisions[1], answer)
        answer = torch.where(initial == 8, decisions[2], answer)
    declined = answer == 6
    correct = (answer == data.value).float()
    reward = torch.where(declined, torch.full_like(data.cost, .30), correct)
    return {'return': reward - data.cost * count, 'correct': correct, 'inspections': count,
        'answer': answer, 'declined': declined, 'initial_action': initial, 'visited': visited}


def analytic_policy(data, native=False):
    prior = .30 if native else 1 / 6
    fresh = data.condition == 0
    if data.task == 'serial':
        continuation = torch.maximum(data.actual_a, 1 - data.cost)
        buy = (~fresh) & (continuation - data.cost > prior)
        verify = buy & (1 - data.cost > data.actual_a)
        count = buy.long() + verify.long()
        answer = torch.where(fresh, data.value, torch.full_like(data.value, 6 if native else 0))
        answer = torch.where(buy, data.sample_a, answer)
        answer = torch.where(verify, data.value, answer)
        initial = torch.where(buy, torch.full_like(answer, 7), answer)
        expected = torch.where(fresh, torch.ones_like(data.cost), torch.maximum(torch.full_like(data.cost, prior), continuation - data.cost))
    else:
        use_a = data.actual_a >= data.actual_b
        quality = torch.maximum(data.actual_a, data.actual_b)
        buy = (~fresh) & (quality - data.cost > prior)
        answer = torch.where(fresh, data.value, torch.full_like(data.value, 6 if native else 0))
        answer = torch.where(buy, torch.where(use_a, data.sample_a, data.sample_b), answer)
        count = buy.long()
        initial = torch.where(buy, torch.where(use_a, 7, 8), answer)
        expected = torch.where(fresh, torch.ones_like(data.cost), torch.maximum(torch.full_like(data.cost, prior), quality - data.cost))
    correct = (answer == data.value).float()
    reward = torch.where(answer == 6, torch.full_like(data.cost, .30), correct)
    return {'return': reward - count * data.cost, 'correct': correct, 'inspections': count,
        'answer': answer, 'declined': answer == 6, 'initial_action': initial, 'expected_return': expected}
