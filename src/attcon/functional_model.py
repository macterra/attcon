"""Learned attention forecasts grounded in paired physical access processes."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .predictive_attention import DECAY, Forecast


@dataclass
class Process:
    observations: torch.Tensor
    allocation: torch.Tensor
    recovery: torch.Tensor
    next_effects: torch.Tensor


def paired_process(seed, count=128, steps=8):
    g = torch.Generator().manual_seed(seed)
    command = torch.randint(4, (count, steps), generator=g)
    replay = torch.randint(4, (count,), generator=g)
    direction = torch.randint(2, (count,), generator=g) * 2 - 1
    quality = .1 + .9 * torch.rand(count, steps, 2, 1, generator=g)
    histories = [[], []]
    allocations = [[], []]
    recoveries = [[], []]
    q = [torch.zeros(count, 2, 4), torch.zeros(count, 2, 4)]
    for t in range(steps):
        replay = (replay + direction) % 4
        for owner in (0, 1):
            slot = replay[:, None].expand(-1, 2).clone()
            slot[:, owner] = command[:, t]
            a = F.one_hot(slot, 4).float()
            glimpse = a * quality[:, t]
            q[owner] = q[owner] * DECAY * (1-a) + glimpse
            x = torch.cat((a.flatten(1), glimpse.flatten(1), F.one_hot(command[:, t], 4).float()), -1)
            histories[owner].append(x)
            allocations[owner].append(a)
            recoveries[owner].append(q[owner])
    next_replay = (replay + direction) % 4
    out = []
    for owner in (0, 1):
        effects = F.one_hot(next_replay[:, None, None].expand(-1, 4, 2), 4).float()
        for c in range(4):
            effects[:, c, owner] = F.one_hot(torch.full((count,), c), 4).float()
        out.append(Process(torch.stack(histories[owner], 1), torch.stack(allocations[owner], 1),
                           torch.stack(recoveries[owner], 1), effects))
    return out


def buffer_contents(visual, recovery):
    """Explicit categorical uncertainty bridge, shared by both channels."""
    return visual * recovery[..., None] + .25 * (1-recovery[..., None])


@torch.no_grad()
def alternatives(model, history, effects=None, quality=.8):
    forecast, hidden = model(history)
    current = Forecast(forecast.allocation[:, -1], forecast.access[:, -1], forecast.effects[:, -1])
    command_effects = current.effects if effects is None else effects
    predicted = []
    for c in range(4):
        a = command_effects[:, c]
        x = torch.cat((a.flatten(1), (quality*a).flatten(1),
                       F.one_hot(torch.full((len(history),), c), 4).float()), -1)
        next_state, _ = model(x[:, None], hidden.clone())
        predicted.append(next_state.access[:, 0, 0])
    return current, hidden, torch.stack(predicted, 1)


def execute_alternatives(process, quality=.8, effects=None):
    a = process.next_effects if effects is None else effects
    return process.recovery[:, -1, None] * DECAY * (1-a) + quality*a


def neutral_record(visual, current, predicted_recovery, process, row, order=(0, 1, 2)):
    """State and observed history only; evaluator identities are excluded."""
    if tuple(sorted(order)) != (0, 1, 2):
        raise ValueError('invalid node order')
    names = {physical: f'n{shown}' for shown, physical in enumerate(order)}
    now = buffer_contents(visual, current.access[:, 0])
    next_content = buffer_contents(visual[:, None], predicted_recovery)
    def rounded(tensor):
        return [[round(float(v), 5) for v in x] for x in tensor]
    def nodes(values):
        return [{'node': names[p], 'color_and_shape_distributions': rounded(values[p] if p < 2 else values[0])}
                for p in order]
    history = []
    for t in range(process.observations.shape[1]):
        obs = process.observations[row, t]
        allocation = obs[:8].reshape(2, 4)
        glimpse = obs[8:16].reshape(2, 4)
        history.append({'command': f'k{int(obs[-4:].argmax())}',
            'observed': [{'node': names[p], 'allocation': [round(float(x), 5) for x in allocation[p]],
                          'acquisition': [round(float(x), 5) for x in glimpse[p]]} for p in order if p < 2]})
    return {'color_order': ['red', 'green', 'blue', 'yellow'],
            'shape_order': ['circle', 'square', 'triangle', 'cross'],
            'output_node': names[2], 'observed_history': history,
            'predicted_current': nodes(now[row]),
            'predicted_by_command': [{'command': f'k{c}', 'nodes': nodes(next_content[row, c])} for c in range(4)]}
