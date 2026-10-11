"""New DEVELOPMENT spatial task; task truth is never a native consumer input.

No checkpoint loader, fit loop, dataset collector, attention model or Reporter.
The 4x4 occupancy grid is task supervision, not an interpretation of encoder RFs.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from attention_content_form_primitives_v1 import (
    AddressedItem, AddressedState, PendingOperator, SpatialBuffer, block_average,
    execute_endogenous, execute_exogenous, restore_saved,
)

ADDRESSES = ('l0', 'l1', 'l2', 'l3')
ROLES = ('red', 'blue')
OCCUPANCY_CLASSES = ('background', 'red', 'blue')
COARSE_TOLERANCE = 1e-6


def _index(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def _finite_grid(value, channels):
    if (not isinstance(value, torch.Tensor) or value.ndim < 3
            or value.shape[-3:] != (channels, 4, 4)
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError(f'expected finite floating [...,{channels},4,4] tensor')


@dataclass(frozen=True)
class EvaluatorGridTruth:
    """Immutable exact cells [outer item][red/blue][row,column], evaluator only."""
    cells: tuple

    def __post_init__(self):
        if type(self.cells) is not tuple or len(self.cells) != 4:
            raise ValueError('truth requires four outer items')
        for pair in self.cells:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError('each item requires red and blue cells')
            for cell in pair:
                if (type(cell) is not tuple or len(cell) != 2
                        or any(type(x) is not int or not 0 <= x < 4 for x in cell)):
                    raise ValueError('invalid task grid cell')
            if pair[0][1] // 2 == pair[1][1] // 2:
                raise ValueError('roles must occupy opposite halves')

    def occupancy(self):
        target = torch.zeros(4, 4, 4, dtype=torch.long)
        for item, pair in enumerate(self.cells):
            for role, (row, column) in enumerate(pair, 1):
                target[item, row, column] = role
        return target


def layout_for_world(index):
    """Deterministic 128-layout orbit; every world has two of each coarse order.

    In each aligned 32-world block every address/role occupies every cell twice;
    vertical order is symmetric (12 above, 12 below, 8 equal). Across 128 worlds,
    each address covers all 128 opposite-half ordered position pairs exactly once.
    """
    _index(index, 'world index')
    cells = []
    for address in range(4):
        code = (index % 128) ^ ((address * 32) | (address % 2))
        side = code & 1
        red = ((code >> 2) & 3, ((code >> 1) & 1) + 2 * side)
        blue_row = (((code >> 3) ^ (code >> 5)) & 1) + 2 * ((code >> 4) & 1)
        blue_col = (((code >> 1) ^ (code >> 4) ^ (code >> 6)) & 1) + 2 * (1 - side)
        cells.append((red, (blue_row, blue_col)))
    return EvaluatorGridTruth(tuple(cells))


def balanced_layouts(world_count):
    _index(world_count, 'world count')
    if world_count == 0 or world_count % 32:
        raise ValueError('balanced world count must be a positive multiple of 32')
    return tuple(layout_for_world(i) for i in range(world_count))


@dataclass(frozen=True)
class RenderedWorld:
    pixels: torch.Tensor
    truth: EvaluatorGridTruth
    token_shapes: tuple
    intensities: tuple
    noise_sigma: float


def render_world(seed, index):
    """Render four patches only; caller owns any future development collection.

    Neutral RGB background .08; 4x4 pixel cells contain a centered 2x2 square
    or five-pixel cross. Independent role intensity U(.65,1), shape Bernoulli(.5),
    and additive per-pixel Gaussian noise sigma U(.005,.02), then clip to [0,1].
    Noise sigma varies independently per world. Seed/index SHA256 isolates streams
    and rendering never consumes or alters the global torch RNG.
    """
    _index(seed, 'seed'); _index(index, 'world index')
    stream = hashlib.sha256(f'spatial_native_task_v1:{seed}:{index}'.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(stream[:8], 'big') % (2**63))
    truth = layout_for_world(index)
    shapes = torch.randint(2, (4, 2), generator=generator)
    intensities = .65 + .35 * torch.rand(4, 2, generator=generator)
    noise_sigma = float(.005 + .015 * torch.rand((), generator=generator))
    pixels = torch.full((4, 3, 16, 16), .08)
    square = ((1, 1), (1, 2), (2, 1), (2, 2))
    cross = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
    for item, pair in enumerate(truth.cells):
        for role, (row, column) in enumerate(pair):
            token = square if int(shapes[item, role]) == 0 else cross
            color = torch.tensor((1., .08, .08) if role == 0 else (.08, .08, 1.))
            for dy, dx in token:
                pixels[item, :, row * 4 + dy, column * 4 + dx] = color * intensities[item, role]
    pixels = (pixels + noise_sigma * torch.randn(4, 3, 16, 16, generator=generator)).clamp(0, 1)
    return RenderedWorld(pixels, truth, tuple(tuple(int(x) for x in r) for r in shapes),
                         tuple(tuple(float(x) for x in r) for r in intensities), noise_sigma)


class FrozenSpatialExtractor:
    """Copied frozen source and control pathways; caller supplies loaded modules.

    Constructing this helper performs no loading or forward pass. The category
    suffix and standardized hidden64 orientation pathway keep their native units.
    """
    def __init__(self, encoder, orientation_head, mean64, scale64):
        if (mean64.shape != (64,) or scale64.shape != (64,)
                or mean64.dtype != torch.float32 or scale64.dtype != torch.float32
                or not torch.isfinite(mean64).all() or not torch.isfinite(scale64).all()
                or (scale64 <= 0).any()):
            raise ValueError('orientation standardizer needs finite float32 mean and positive scale64')
        self.encoder = deepcopy(encoder).cpu().eval()
        self.orientation_head = deepcopy(orientation_head).cpu().eval()
        self.mean64, self.scale64 = mean64.detach().cpu().clone(), scale64.detach().cpu().clone()
        for module in (self.encoder, self.orientation_head):
            for parameter in module.parameters():
                if parameter.dtype != torch.float32:
                    raise ValueError('frozen modules must use float32')
                parameter.requires_grad_(False)

    @torch.no_grad()
    def extract(self, pixels):
        if (pixels.shape[-3:] != (3, 16, 16) or pixels.dtype != torch.float32
                or pixels.device.type != 'cpu' or not torch.isfinite(pixels).all()):
            raise ValueError('pixels must be finite float32 CPU [...,3,16,16]')
        result = self.encoder.net[:6](pixels.reshape(-1, 3, 16, 16))
        _finite_grid(result, 24)
        return result.reshape(*pixels.shape[:-3], 24, 4, 4).clone()

    @torch.no_grad()
    def controls(self, actual_w):
        _finite_grid(actual_w, 24)
        flat = actual_w.reshape(-1, 24, 4, 4)
        hidden = self.encoder.net[6:9](flat)
        logits = self.encoder.net[6:](flat)
        orientation = self.orientation_head((hidden - self.mean64) / self.scale64)[:, 0]
        leading = actual_w.shape[:-3]
        return {'category_logits': logits.reshape(*leading, 8),
                'category_probabilities': torch.cat((logits[:, :4].softmax(-1), logits[:, 4:].softmax(-1)), -1).reshape(*leading, 8),
                'orientation_hidden': hidden.reshape(*leading, 64),
                'orientation_logits': orientation.reshape(leading)}


class LocalOccupancyHead(nn.Module):
    """Same local ReLU MLP at every site; default 899 parameters (raw-grid 227)."""
    def __init__(self, channels=24):
        super().__init__()
        if type(channels) is not int or channels not in (24, 3):
            raise ValueError('only primary24 and raw-grid3 channel tasks are supported')
        self.channels = channels
        self.net = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, actual_w):
        _finite_grid(actual_w, self.channels)
        return self.net(actual_w.movedim(-3, -1)).movedim(-1, -3)


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: tuple
    scale: tuple

    def __post_init__(self):
        if (type(self.mean) is not tuple or type(self.scale) is not tuple
                or len(self.mean) not in (3, 24) or len(self.mean) != len(self.scale)
                or any(not math.isfinite(x) for x in self.mean + self.scale)
                or any(x <= 0 for x in self.scale)):
            raise ValueError('invalid channel standardizer')

    @classmethod
    def fit(cls, fit_w):
        """Fit-only population statistics; caller must enforce whole-world split."""
        _finite_grid(fit_w, fit_w.shape[-3])
        if fit_w.dtype != torch.float32 or fit_w.numel() == 0:
            raise ValueError('fit maps must be nonempty float32')
        rows = fit_w.detach().movedim(-3, -1).reshape(-1, fit_w.shape[-3])
        mean, scale = rows.mean(0), rows.std(0, correction=0)
        scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
        return cls(tuple(mean.tolist()), tuple(scale.tolist()))

    def apply(self, actual_w):
        _finite_grid(actual_w, len(self.mean))
        mean = actual_w.new_tensor(self.mean).reshape(-1, 1, 1)
        scale = actual_w.new_tensor(self.scale).reshape(-1, 1, 1)
        return (actual_w - mean) / scale


def occupancy_loss(logits, evaluator_target):
    """Unweighted mean per-site CE; targets enter only this training/evaluator API."""
    _finite_grid(logits, 3)
    if (evaluator_target.shape != logits.shape[:-3] + (4, 4)
            or evaluator_target.dtype != torch.long
            or ((evaluator_target < 0) | (evaluator_target > 2)).any()):
        raise ValueError('occupancy targets must be long [...,4,4] classes 0/1/2')
    return F.cross_entropy(logits.reshape(-1, 3, 4, 4), evaluator_target.reshape(-1, 4, 4))


def role_fields(logits):
    """Spatial softmax of role/background contrasts, computed in float64.

    Float64 subtraction avoids float32 contrast overflow. Return original dtype;
    identical scores have explicit uniform 1/16 fields. No nonuniform evidence
    threshold, trained role confidence or presence default is introduced.
    """
    _finite_grid(logits, 3)
    scores = logits[..., 1:3, :, :].double() - logits[..., :1, :, :].double()
    flat = scores.flatten(-2)
    uniform = (flat == flat[..., :1]).all(-1, keepdim=True)
    fields = torch.where(uniform, torch.full_like(flat, 1 / 16), flat.softmax(-1))
    return fields.reshape(*logits.shape[:-3], 2, 4, 4).to(logits.dtype)


@dataclass(frozen=True)
class NativeItemEvidence:
    address: str
    local_logits: SpatialBuffer
    role_fields: SpatialBuffer


@dataclass(frozen=True)
class NativeSnapshot:
    """Completed actual-W consumer output; no pixels, truth, V or future estimate."""
    time: int
    items: tuple[NativeItemEvidence, ...]

    def item(self, address):
        for item in self.items:
            if item.address == address:
                return item
        raise ValueError('unknown or unseen native item address')


@dataclass(frozen=True)
class TargetAction:
    address: str
    role: str
    row: int
    column: int
    tied_cells: tuple


@dataclass(frozen=True)
class CoarseAction:
    address: str
    order: str
    expected_columns: tuple


class NativeController:
    """Native executed grid targeting from current W through H alone."""
    def __init__(self, head, standardizer=None):
        if not isinstance(head, LocalOccupancyHead) or head.channels != 24:
            raise ValueError('native controller needs primary 24-channel head')
        if standardizer is not None and (not isinstance(standardizer, ChannelStandardizer)
                                        or len(standardizer.mean) != 24):
            raise ValueError('native standardizer must have 24 channels')
        self.head, self.standardizer = head, standardizer

    @torch.no_grad()
    def consume(self, state):
        if not isinstance(state, AddressedState):
            raise ValueError('completed addressed state required; pending operators are separate')
        items = []
        for item in state.items:
            if not item.observed:
                continue
            actual_w = item.actual_w.tensor()
            _finite_grid(actual_w, 24)
            head_input = self.standardizer.apply(actual_w) if self.standardizer else actual_w
            logits = self.head(head_input)
            items.append(NativeItemEvidence(item.address, SpatialBuffer.from_tensor(logits),
                                            SpatialBuffer.from_tensor(role_fields(logits))))
        return NativeSnapshot(state.time, tuple(items))

    def target(self, snapshot, address, role):
        if not isinstance(snapshot, NativeSnapshot) or role not in ROLES:
            raise ValueError('native snapshot and red/blue role query required')
        flat = snapshot.item(address).role_fields.tensor()[ROLES.index(role)].flatten()
        maximum = flat.max()
        ties = tuple((int(i) // 4, int(i) % 4) for i in (flat == maximum).nonzero().flatten())
        return TargetAction(address, role, ties[0][0], ties[0][1], ties)

    def coarse_order(self, snapshot, address):
        if not isinstance(snapshot, NativeSnapshot):
            raise ValueError('native snapshot required')
        fields = snapshot.item(address).role_fields.tensor().double()
        expected = (fields * torch.arange(4, dtype=torch.float64)).sum((1, 2))
        difference = float(expected[0] - expected[1])
        order = ('unknown' if abs(difference) <= COARSE_TOLERANCE else
                 'red_left_of_blue' if difference < 0 else 'blue_left_of_red')
        return CoarseAction(address, order, tuple(expected.tolist()))
