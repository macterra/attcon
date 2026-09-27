"""Visual contents bound into a predictive attention model used for control.

The binding is explicit architecture. Perceptual and process predictions are
learned; none of these tensors is trained to denote consciousness or experience.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .predictive_attention import Forecast

COLORS = ('red', 'green', 'blue', 'yellow')
SHAPES = ('circle', 'square', 'triangle', 'cross')
LOCATIONS = ('upper', 'right', 'lower', 'left')
PALETTE = torch.tensor([[1., .08, .08], [.08, 1., .08], [.08, .08, 1.], [1., 1., .08]])


def render(colors, shapes, generator):
    leading = colors.shape
    colors, shapes = colors.flatten(), shapes.flatten()
    n = len(colors)
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
    cx = 7.5 + torch.rand(n, 1, 1, generator=generator) * 2 - 1
    cy = 7.5 + torch.rand(n, 1, 1, generator=generator) * 2 - 1
    dx, dy = xx[None] - cx, yy[None] - cy
    r = 3.7 + torch.rand(n, 1, 1, generator=generator) * 1.5
    masks = torch.stack((dx.square() + dy.square() <= r.square(),
                         (dx.abs() <= r) & (dy.abs() <= r),
                         (dy >= -r) & (dy <= r) & (dx.abs() <= (dy + r) * .5),
                         ((dx.abs() <= 1.6) & (dy.abs() <= r)) | ((dy.abs() <= 1.6) & (dx.abs() <= r))), 1)
    mask = masks[torch.arange(n), shapes].float()
    intensity = .65 + .35 * torch.rand(n, 1, 1, 1, generator=generator)
    patches = mask[:, None] * PALETTE[colors, :, None, None] * intensity
    patches += .015 * torch.randn(n, 3, 16, 16, generator=generator)
    return patches.clamp(0, 1).reshape(*leading, 3, 16, 16)


def scenes(seed, count):
    g = torch.Generator().manual_seed(seed)
    colors = torch.rand(count, 2, 4, generator=g).argsort(-1)
    shapes = torch.rand(count, 2, 4, generator=g).argsort(-1)
    return render(colors, shapes, g), colors, shapes


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(24 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 8))

    def forward(self, patches):
        logits = self.net(patches.reshape(-1, 3, 16, 16)).reshape(*patches.shape[:-3], 8)
        return torch.cat((logits[..., :4].softmax(-1), logits[..., 4:].softmax(-1)), -1)


@dataclass
class BoundState:
    attention: Forecast
    visual: torch.Tensor  # [batch, view, remembered object, color+shape]
    binding: torch.Tensor  # [batch, view, attention location, remembered object]
    remembered: torch.Tensor  # [batch, view, remembered object]

    @property
    def contents(self):
        return torch.einsum('bcso,bcof->bcsf', self.binding, self.visual)

    @property
    def known(self):
        return torch.einsum('bcso,bco->bcs', self.binding, self.remembered.float()) > .5

    def content_policy(self, view, color, shape):
        """Select a command for a content query using bindings and modeled effects.

        No simulator object labels, actual channel owner, or true allocations
        are inputs. The object-addressed model determines predicted command value.
        """
        rows = torch.arange(len(view))
        represented = self.contents[rows, view]
        match = represented[rows, :, color] * represented[rows, :, 4 + shape]
        match = match * self.known[rows, view]
        effects = self.attention.effects[rows, :, view, :]
        utility = (effects * match[:, None, :]).sum(-1)
        return utility.argmax(-1)

    def replace_visual(self, visual):
        if visual.shape != self.visual.shape or not torch.isfinite(visual).all():
            raise ValueError('invalid visual state')
        for component in (visual[..., :4], visual[..., 4:]):
            if (component < 0).any() or not torch.allclose(component.sum(-1), torch.ones_like(component[..., 0]), atol=1e-5):
                raise ValueError('visual components must be distributions')
        return BoundState(self.attention.clone(), visual.clone(), self.binding.clone(), self.remembered.clone())

    def replace_binding(self, binding):
        if binding.shape != self.binding.shape or not torch.isfinite(binding).all() or (binding < 0).any():
            raise ValueError('invalid binding')
        if not torch.allclose(binding.sum(-1), torch.ones_like(binding[..., 0]), atol=1e-6):
            raise ValueError('binding rows must sum to one')
        return BoundState(self.attention.clone(), self.visual.clone(), binding.clone(), self.remembered.clone())

    def replace_attention(self, attention):
        return BoundState(attention, self.visual.clone(), self.binding.clone(), self.remembered.clone())


@torch.no_grad()
def bind(encoder, patches, attention, observed_allocation):
    visual = encoder(patches)
    remembered = observed_allocation.sum(1) > 0
    visual = torch.where(remembered[..., None], visual, torch.full_like(visual, .25))
    binding = torch.eye(4).expand(len(patches), 2, 4, 4).clone()
    return BoundState(attention, visual, binding, remembered)


def full_record(state, row):
    """Lossless-in-structure view of all linked model entries, not a target summary.

    Only rounding and semantic names for trained perceptual classes are applied.
    No thresholds or consciousness-like classifications are added to the state.
    """
    result = []
    for view in range(2):
        objects = []
        for location in range(4):
            objects.append({'location': LOCATIONS[location],
                'color_distribution': {k: round(float(v), 5) for k, v in zip(COLORS, state.contents[row, view, location, :4])},
                'shape_distribution': {k: round(float(v), 5) for k, v in zip(SHAPES, state.contents[row, view, location, 4:])},
                'selection_probability': round(float(state.attention.allocation[row, view, location]), 5),
                'recoverability_now_then_one_then_two_steps': [round(float(x), 5) for x in state.attention.access[row, :, view, location]],
                'next_selection_by_command': {LOCATIONS[c]: round(float(state.attention.effects[row, c, view, location]), 5) for c in range(4)}})
        result.append({'view': 'A' if view == 0 else 'B', 'objects': objects})
    return result
