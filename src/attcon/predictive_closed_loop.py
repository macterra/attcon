"""Execute learned forecast-guided commands in the simulated attention process."""
import torch
from torch.nn import functional as F
from .predictive_attention import DECAY, Forecast


@torch.no_grad()
def rollout(model, seed, count=512, steps=16, intervention='none'):
    if intervention not in ('none', 'rotate_effects', 'restore', 'shuffle_effects'):
        raise ValueError(intervention)
    g = torch.Generator().manual_seed(seed)
    owner = torch.randint(2, (count,), generator=g)
    direction = torch.randint(2, (count,), generator=g) * 2 - 1
    replay = torch.randint(4, (count,), generator=g)
    quality = .1 + .9 * torch.rand(count, steps, 2, 1, generator=g)
    queries = torch.randint(4, (count, steps), generator=g)
    query_channels = torch.randint(2, (count, steps), generator=g)
    random_commands = torch.randint(4, (count, steps), generator=g)
    reconstruction_draws = torch.rand(count, steps, 2, 4, generator=g)
    rows = torch.arange(count)
    q = torch.zeros(count, 2, 4)
    h = None
    forecast = None
    records = []
    for t in range(steps):
        channel, target = query_channels[:, t], queries[:, t]
        if t < 4:
            command = random_commands[:, t]
        else:
            state = forecast.clone()
            if intervention in ('rotate_effects', 'restore'):
                state = state.intervene('effects', state.effects.roll(1, 1))
            if intervention == 'restore':
                state = state.intervene('effects', forecast.effects)
            if intervention == 'shuffle_effects':
                state = state.intervene('effects', state.effects.roll(1, 0))
            command = state.command(channel, target)
        replay = (replay + direction) % 4
        selected = replay[:, None].expand(-1, 2).clone()
        selected[rows, owner] = command
        allocation = F.one_hot(selected, 4).float()
        glimpses = allocation * quality[:, t]
        q = q * DECAY * (1 - allocation) + glimpses
        observation = torch.cat((allocation.flatten(1), glimpses.flatten(1), F.one_hot(command, 4).float()), -1)
        prediction, h = model(observation[:, None], h)
        forecast = Forecast(prediction.allocation[:, 0], prediction.access[:, 0], prediction.effects[:, 0])
        records.append({'command': command, 'query_channel': channel, 'query_slot': target,
                        'query_controlled': channel == owner,
                        'query_selected': selected[rows, channel] == target,
                        'query_access': q[rows, channel, target],
                        'query_reconstructed': reconstruction_draws[:, t][rows, channel, target] < q[rows, channel, target],
                        'model_access': forecast.access[:, 0], 'physical_access': q.clone(),
                        'model_effects': forecast.effects})
    trace = {key: torch.stack([r[key] for r in records], 1) for key in records[0]}
    own = trace['query_controlled'][:, 4:]
    def avg(key, mask=own):
        return trace[key][:, 4:][mask].double().mean().item()
    metrics = {'controlled_query_count': int(own.sum()), 'controlled_query_selected': avg('query_selected'),
               'controlled_query_access': avg('query_access'), 'controlled_query_reconstructed': avg('query_reconstructed'),
               'replay_query_selected': avg('query_selected', ~own),
               'access_mae': (trace['model_access'][:, 4:] - trace['physical_access'][:, 4:]).abs().double().mean().item()}
    return metrics, trace
