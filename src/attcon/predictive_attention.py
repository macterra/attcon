"""Predict a simulated allocation/reconstruction process, without experience labels.

The forecast tensor is the identified attention-control model state. The GRU is
its estimator; object properties and simulator truth are not inputs to the policy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS, SLOTS, DECAY = 2, 4, .75


@dataclass
class Episode:
    observations: torch.Tensor
    allocation: torch.Tensor
    access: torch.Tensor
    next_allocation: torch.Tensor
    controlled: torch.Tensor
    objects: torch.Tensor
    commands: torch.Tensor
    reconstruction: torch.Tensor


def simulate(seed: int, count: int = 128, steps: int = 12) -> Episode:
    g = torch.Generator().manual_seed(seed)
    controlled = torch.randint(2, (count,), generator=g)
    direction = torch.randint(2, (count,), generator=g) * 2 - 1
    replay = torch.randint(4, (count,), generator=g)
    commands = torch.randint(4, (count, steps), generator=g)
    # V is independent of control and signal quality, and never an A input.
    objects = torch.randint(1000000, (count, 2, 4), generator=g)
    q = torch.zeros(count, 2, 4)
    xs, allocations, accesses, futures, reconstructions = [], [], [], [], []
    rows = torch.arange(count)
    for t in range(steps):
        replay = (replay + direction) % 4
        slots = replay[:, None].expand(-1, 2).clone()
        slots[rows, controlled] = commands[:, t]
        allocation = F.one_hot(slots, 4).float()
        quality = .1 + .9 * torch.rand(count, 2, 1, generator=g)
        glimpses = allocation * quality
        q = q * DECAY * (1 - allocation) + glimpses
        future = F.one_hot(((replay + direction) % 4)[:, None, None].expand(-1, 4, 2), 4).float()
        for command in range(4):
            future[rows, command, controlled] = F.one_hot(torch.full((count,), command), 4).float()
        xs.append(torch.cat((allocation.flatten(1), glimpses.flatten(1), F.one_hot(commands[:, t], 4).float()), -1))
        allocations.append(allocation)
        accesses.append(torch.stack([q, q * DECAY, q * DECAY ** 2], dim=1))
        futures.append(future)
        reconstructions.append((torch.rand(q.shape, generator=g) < q).float())
    return Episode(torch.stack(xs, 1), torch.stack(allocations, 1), torch.stack(accesses, 1),
                   torch.stack(futures, 1), controlled, objects, commands, torch.stack(reconstructions, 1))


@dataclass
class Forecast:
    allocation: torch.Tensor  # [..., channel, slot]
    access: torch.Tensor  # [..., unattended delay (0,1,2), channel, slot]
    effects: torch.Tensor  # [..., alternative command, channel, slot]

    def clone(self):
        return Forecast(*(x.clone() for x in (self.allocation, self.access, self.effects)))

    def controllability(self):
        # Total-variation spread over command-conditioned allocation predictions.
        return (self.effects - self.effects.mean(-3, keepdim=True)).abs().sum(-1).mean(-2)

    def command(self, channel: torch.Tensor, slot: torch.Tensor):
        """Choose the command maximizing predicted next allocation to the query.

        Policy consumes only A and the task query. No owner flag or simulator
        allocation is consulted. Batched single-decision forecasts required.
        """
        rows = torch.arange(len(channel))
        utility = self.effects[rows, :, channel, slot]
        return utility.argmax(-1)

    def intervene(self, field: str, value: torch.Tensor):
        if field not in ('allocation', 'access', 'effects'):
            raise ValueError('unknown forecast field')
        if value.shape != getattr(self, field).shape:
            raise ValueError('intervention shape mismatch')
        if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError('forecasts must be finite probabilities')
        if field != 'access' and not torch.allclose(value.sum(-1), torch.ones_like(value.sum(-1)), atol=1e-6):
            raise ValueError('allocation distributions must sum to one')
        return replace(self.clone(), **{field: value.clone()})


class PredictiveAttention(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(20, hidden, batch_first=True)
        self.allocation_head = nn.Linear(hidden, 8)
        self.access_head = nn.Linear(hidden, 24)
        self.effect_head = nn.Linear(hidden, 32)

    def forward(self, observations, h=None):
        z, h = self.gru(observations, h)
        shape = z.shape[:-1]
        forecast = Forecast(self.allocation_head(z).reshape(*shape, 2, 4).softmax(-1),
                            self.access_head(z).reshape(*shape, 3, 2, 4).sigmoid(),
                            self.effect_head(z).reshape(*shape, 4, 2, 4).softmax(-1))
        return forecast, h


def prediction_loss(forecast, episode):
    # Expected reconstruction outcome is a simulator probability, not a report
    # or confidence label. Cross entropy learns allocation dynamics.
    allocation = -(episode.allocation * forecast.allocation.clamp_min(1e-8).log()).sum(-1).mean()
    effects = -(episode.next_allocation[:, 3:] * forecast.effects[:, 3:].clamp_min(1e-8).log()).sum(-1).mean()
    access = F.mse_loss(forecast.access, episode.access)
    return allocation + effects + 10 * access


@torch.no_grad()
def evaluate(model, episode):
    f, _ = model(episode.observations)
    f = Forecast(*(x[:, 4:] for x in (f.allocation, f.access, f.effects)))
    access = episode.access[:, 4:]
    controlled = episode.controlled[:, None].expand(-1, f.allocation.shape[1])
    return {
        'allocation_accuracy': (f.allocation.argmax(-1) == episode.allocation[:, 4:].argmax(-1)).float().mean().item(),
        'effect_accuracy': (f.effects.argmax(-1) == episode.next_allocation[:, 4:].argmax(-1)).float().mean().item(),
        'access_mae': (f.access - access).abs().mean().item(),
        'access_max_error': (f.access - access).abs().max().item(),
        'controlled_channel_accuracy': (f.controllability().argmax(-1) == controlled).float().mean().item(),
        'reconstruction_brier': ((f.access[:, :, 0] - episode.reconstruction[:, 4:]) ** 2).mean().item(),
        'oracle_reconstruction_brier': ((access[:, :, 0] - episode.reconstruction[:, 4:]) ** 2).mean().item(),
    }
