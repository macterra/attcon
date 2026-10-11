"""Reviewed attention-content core; no loading, collection, fit loop or Reporter.

All prediction inputs are permitted pre-reveal observations. Feedback is a
separate post-scoring API. Raw map outputs are estimates, not completed W.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import permutations
import math

import torch
from torch import nn
from torch.nn import functional as F

from attention_content_form_primitives_v1 import block_average
from spatial_native_task_v1 import ChannelStandardizer, LocalOccupancyHead, role_fields

ADDRESSES = ('l0', 'l1', 'l2', 'l3')
SYMBOLS = ('c0', 'c1', 'c2')
MAPPINGS = tuple(permutations((1, 2, 4)))
PARAMETER_COUNT = 197328


def _time(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def _float(value, shape, name):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.dtype != torch.float32 or value.device.type != 'cpu'
            or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite CPU float32 {tuple(shape)}')


def _maps(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[0] < 1:
        raise ValueError(f'{name} must be [B,4,24,4,4]')
    _float(value, (value.shape[0], 4, 24, 4, 4), name)
    return value.shape[0]


def _indices(value, batch_size, maximum, name):
    if (not isinstance(value, torch.Tensor) or value.shape != (batch_size,)
            or value.dtype != torch.long or value.device.type != 'cpu'
            or ((value < 0) | (value >= maximum)).any()):
        raise ValueError(f'{name} must be CPU long [B] in 0..{maximum-1}')


def _source_key(value):
    if not isinstance(value, str) or not value:
        raise ValueError('source_key must be a nonempty audit-only string')


@dataclass(frozen=True)
class FitNormalizer:
    mean: tuple
    scale: tuple

    def __post_init__(self):
        if (type(self.mean) is not tuple or type(self.scale) is not tuple
                or len(self.mean) != 24 or len(self.scale) != 24
                or any(not math.isfinite(x) for x in self.mean + self.scale)
                or any(x <= 0 for x in self.scale)):
            raise ValueError('normalizer needs finite 24-channel mean/positive scale')

    @classmethod
    def fit(cls, fit_v, *, split):
        if split != 'fit':
            raise ValueError('normalizer statistics may use fit split only')
        _maps(fit_v, 'fit_v')
        rows = fit_v.detach().movedim(-3, -1).reshape(-1, 24)
        mean, scale = rows.mean(0), rows.std(0, correction=0)
        scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
        return cls(tuple(mean.tolist()), tuple(scale.tolist()))

    def _coordinates(self, value):
        if (not isinstance(value, torch.Tensor) or value.ndim < 3
                or value.shape[-3:] != (24, 4, 4)):
            raise ValueError('normalizer input must end in [24,4,4]')
        _float(value, value.shape, 'normalizer input')
        return (value.new_tensor(self.mean).reshape(24, 1, 1),
                value.new_tensor(self.scale).reshape(24, 1, 1))

    def normalize(self, value):
        mean, scale = self._coordinates(value)
        result = (value - mean) / scale
        _float(result, value.shape, 'normalized maps')
        return result

    def denormalize(self, value):
        mean, scale = self._coordinates(value)
        result = value * scale + mean
        _float(result, value.shape, 'denormalized maps')
        return result


@dataclass(frozen=True)
class CommandBatch:
    address: torch.Tensor
    symbol: torch.Tensor

    def validate(self, batch_size):
        _indices(self.address, batch_size, 4, 'command address')
        _indices(self.symbol, batch_size, 3, 'command symbol')


@dataclass(frozen=True)
class ObservationBatch:
    v: torch.Tensor
    last_revealed_w: torch.Tensor
    decision_time: int
    last_revealed_at: int
    previous_command: CommandBatch | None
    observed: torch.Tensor

    def validate(self):
        batch_size = _maps(self.v, 'V')
        _float(self.last_revealed_w, self.v.shape, 'last revealed W')
        _time(self.decision_time, 'decision_time')
        _time(self.last_revealed_at, 'last_revealed_at')
        if self.last_revealed_at != max(0, self.decision_time - 1):
            raise ValueError('prediction must precede current completed-map reveal')
        if (not isinstance(self.observed, torch.Tensor)
                or self.observed.shape != (batch_size, 4)
                or self.observed.dtype != torch.bool or self.observed.device.type != 'cpu'
                or not self.observed.all()):
            raise ValueError('this source contract requires four observed addresses')
        if self.decision_time == 0:
            if self.previous_command is not None or not torch.equal(self.last_revealed_w, self.v):
                raise ValueError('initial state is exactly observed V without a command')
        else:
            if not isinstance(self.previous_command, CommandBatch):
                raise ValueError('current estimate needs the last issued command')
            self.previous_command.validate(batch_size)
        return batch_size


def _history(revealed_through, update_command_indices):
    _time(revealed_through, 'revealed_through')
    if update_command_indices != tuple(range(revealed_through)):
        raise ValueError('state must contain exactly commands 0..revealed_through-1')


@dataclass(frozen=True)
class SourceState:
    hidden: torch.Tensor
    revealed_through: int
    update_command_indices: tuple
    source_key: str

    def validate(self, batch_size):
        _float(self.hidden, (batch_size, 4, 48), 'addressed h')
        _history(self.revealed_through, self.update_command_indices)
        _source_key(self.source_key)


@dataclass(frozen=True)
class RevealedFeedback:
    completed_w: torch.Tensor
    command: CommandBatch
    completed_at: int
    command_index: int
    scored_decision_at: int

    def validate(self, batch_size, revealed_through):
        _float(self.completed_w, (batch_size, 4, 24, 4, 4), 'revealed completion')
        if not isinstance(self.command, CommandBatch):
            raise ValueError('completed feedback needs its issued command')
        self.command.validate(batch_size)
        for name in ('completed_at', 'command_index', 'scored_decision_at'):
            _time(getattr(self, name), name)
        if (self.completed_at != revealed_through + 1
                or self.command_index != self.completed_at - 1
                or self.scored_decision_at != self.completed_at):
            raise ValueError('feedback must follow scoring and reveal the next completion once')


@dataclass(frozen=True)
class PredictionBatch:
    current: torch.Tensor
    next: torch.Tensor
    decision_time: int
    source_key: str

    def validate(self):
        batch_size = _maps(self.current, 'estimated current maps')
        _float(self.next, (batch_size, 3, 4, 24, 4, 4), 'prospective maps')
        _time(self.decision_time, 'decision_time')
        _source_key(self.source_key)
        return batch_size


def _prediction_permissions(observation, state, candidate_address, source_key):
    if not isinstance(observation, ObservationBatch):
        raise ValueError('permitted ObservationBatch required')
    batch_size = observation.validate()
    state.validate(batch_size)
    if state.source_key != source_key or state.revealed_through != observation.last_revealed_at:
        raise ValueError('source identity/history does not match the pre-reveal observation')
    _indices(candidate_address, batch_size, 4, 'candidate address')
    return batch_size


def _assemble(observation, state, candidate_address, decoder, source_key):
    batch_size = _prediction_permissions(observation, state, candidate_address, source_key)
    rows = torch.arange(batch_size)
    current = observation.last_revealed_w.clone()
    if observation.decision_time:
        command = observation.previous_command
        current[rows, command.address] = decoder(command.address, command.symbol)
    alternatives = []
    for symbol in range(3):
        next_maps = current.clone()
        next_maps[rows, candidate_address] = decoder(
            candidate_address, torch.full((batch_size,), symbol, dtype=torch.long))
        alternatives.append(next_maps)
    result = PredictionBatch(current, torch.stack(alternatives, 1),
                             observation.decision_time, source_key)
    result.validate()
    return result


class LearnedContentModel(nn.Module):
    """Exact reviewed architecture, shared over addresses; no analytic T path."""
    def __init__(self, normalizer, init_seed, source_key='A'):
        super().__init__()
        if not isinstance(normalizer, FitNormalizer):
            raise ValueError('fit-only normalizer required')
        _time(init_seed, 'init_seed'); _source_key(source_key)
        self.normalizer, self.source_key = normalizer, source_key
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.gru = nn.GRUCell(771, 48).to(device='cpu', dtype=torch.float32)
            self.decoder = nn.Sequential(nn.Linear(435, 96), nn.ReLU(), nn.Linear(96, 384)).to(
                device='cpu', dtype=torch.float32)

    def initial_state(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        return SourceState(torch.zeros(batch_size, 4, 48, dtype=torch.float32), 0, (), self.source_key)

    def decode(self, v_item, hidden, symbols):
        if not isinstance(v_item, torch.Tensor) or v_item.ndim != 4:
            raise ValueError('selected V must be [B,24,4,4]')
        batch_size = v_item.shape[0]
        _float(v_item, (batch_size, 24, 4, 4), 'selected V')
        _float(hidden, (batch_size, 48), 'selected h')
        _indices(symbols, batch_size, 3, 'decoder symbols')
        normalized_v = self.normalizer.normalize(v_item)
        features = torch.cat((normalized_v.flatten(1), hidden,
                              F.one_hot(symbols, 3).to(torch.float32)), 1)
        residual = self.decoder(features).reshape(batch_size, 24, 4, 4)
        return self.normalizer.denormalize(normalized_v + residual)

    def update(self, state, v, feedback):
        batch_size = _maps(v, 'V')
        if not isinstance(state, SourceState) or not isinstance(feedback, RevealedFeedback):
            raise ValueError('source state and post-scoring feedback required')
        state.validate(batch_size)
        if state.source_key != self.source_key:
            raise ValueError('feedback cannot cross prediction source identities')
        feedback.validate(batch_size, state.revealed_through)
        rows, addresses = torch.arange(batch_size), feedback.command.address
        features = torch.cat((self.normalizer.normalize(v[rows, addresses]).flatten(1),
                              self.normalizer.normalize(feedback.completed_w[rows, addresses]).flatten(1),
                              F.one_hot(feedback.command.symbol, 3).to(torch.float32)), 1)
        hidden = state.hidden.clone()
        hidden[rows, addresses] = self.gru(features, state.hidden[rows, addresses])
        result = SourceState(hidden, feedback.completed_at,
                             state.update_command_indices + (feedback.command_index,), self.source_key)
        result.validate(batch_size)
        return result

    def predict(self, observation, state, candidate_address):
        if not isinstance(state, SourceState):
            raise ValueError('learned addressed source state required')
        _prediction_permissions(observation, state, candidate_address, self.source_key)
        rows = torch.arange(observation.v.shape[0])
        return _assemble(observation, state, candidate_address,
                         lambda address, symbol: self.decode(
                             observation.v[rows, address], state.hidden[rows, address], symbol),
                         self.source_key)

    forward = predict


class ExactAlias:
    """Same computation and weights, alternate audit name; no independent source."""
    def __init__(self, model):
        if not isinstance(model, LearnedContentModel):
            raise ValueError('alias requires the identical learned model')
        self.model = model

    def predict(self, observation, state, candidate_address):
        return self.model.predict(observation, state, candidate_address)


def build_learned_sources(normalizer, encoder_seed):
    _time(encoder_seed, 'encoder_seed')
    seed = 320000000 + encoder_seed
    attention = LearnedContentModel(normalizer, seed, 'A')
    ordinary = LearnedContentModel(normalizer, seed + 1000000, 'ordinary')
    return attention, ordinary, ExactAlias(attention)


@dataclass(frozen=True)
class AnalyticState:
    survivors: torch.Tensor
    revealed_symbols: torch.Tensor
    revealed_through: int
    update_command_indices: tuple
    source_key: str = 'analytic'

    def validate(self, batch_size):
        for value, shape in ((self.survivors, (batch_size, 4, 6)),
                             (self.revealed_symbols, (batch_size, 4, 3))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.dtype != torch.bool or value.device.type != 'cpu'):
                raise ValueError('analytic state masks have invalid shape/dtype/device')
        if not self.survivors.any(-1).all():
            raise ValueError('zero-survivor posterior is an invariant failure')
        _history(self.revealed_through, self.update_command_indices)
        _source_key(self.source_key)

    def summary(self):
        self.validate(self.survivors.shape[0])
        counts = self.survivors.sum(-1)
        return {'survivor_count': counts, 'posterior_entropy': counts.to(torch.float64).log(),
                'revealed_symbol_count': self.revealed_symbols.sum(-1)}


class AnalyticMappingSource:
    """Inference from revealed transitions, not evaluator-provided mapping truth."""
    source_key = 'analytic'

    def initial_state(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        return AnalyticState(torch.ones(batch_size, 4, 6, dtype=torch.bool),
                             torch.zeros(batch_size, 4, 3, dtype=torch.bool), 0, ())

    @staticmethod
    def operator_family(v_item):
        if not isinstance(v_item, torch.Tensor) or v_item.ndim != 4:
            raise ValueError('operator family input must be [B,24,4,4]')
        _float(v_item, (v_item.shape[0], 24, 4, 4), 'operator family V')
        return torch.stack([torch.stack([block_average(item, bandwidth) for bandwidth in (1, 2, 4)])
                            for item in v_item])

    def update(self, state, v, feedback):
        batch_size = _maps(v, 'V')
        if not isinstance(state, AnalyticState) or not isinstance(feedback, RevealedFeedback):
            raise ValueError('analytic state and post-scoring feedback required')
        state.validate(batch_size)
        if state.source_key != self.source_key:
            raise ValueError('wrong analytic source identity')
        feedback.validate(batch_size, state.revealed_through)
        rows, address = torch.arange(batch_size), feedback.command.address
        family = self.operator_family(v[rows, address])
        candidate_indices = torch.tensor([[ (1, 2, 4).index(mapping[int(symbol)])
                                           for mapping in MAPPINGS]
                                          for symbol in feedback.command.symbol])
        candidates = family[rows[:, None], candidate_indices]
        matches = torch.isclose(candidates, feedback.completed_w[rows, address, None],
                                atol=1e-6, rtol=1e-5).flatten(2).all(-1)
        survivors, seen = state.survivors.clone(), state.revealed_symbols.clone()
        survivors[rows, address] &= matches
        seen[rows, address, feedback.command.symbol] = True
        result = AnalyticState(survivors, seen, feedback.completed_at,
                               state.update_command_indices + (feedback.command_index,))
        result.validate(batch_size)
        return result

    def decode(self, v_item, survivors, symbols):
        batch_size = v_item.shape[0]
        _indices(symbols, batch_size, 3, 'analytic decoder symbols')
        if (survivors.shape != (batch_size, 6) or survivors.dtype != torch.bool
                or survivors.device.type != 'cpu' or not survivors.any(-1).all()):
            raise ValueError('analytic decoder requires nonempty [B,6] survivor masks')
        family = self.operator_family(v_item)
        rows = torch.arange(batch_size)
        indices = torch.tensor([[(1, 2, 4).index(mapping[int(symbol)]) for mapping in MAPPINGS]
                                for symbol in symbols])
        hypotheses = family[rows[:, None], indices]
        weights = survivors.to(torch.float64)
        weights /= weights.sum(-1, keepdim=True)
        result = (hypotheses.double() * weights[:, :, None, None, None]).sum(1).float()
        _float(result, (batch_size, 24, 4, 4), 'analytic posterior mean')
        return result

    def predict(self, observation, state, candidate_address):
        if not isinstance(state, AnalyticState):
            raise ValueError('analytic state required')
        _prediction_permissions(observation, state, candidate_address, self.source_key)
        rows = torch.arange(observation.v.shape[0])
        return _assemble(observation, state, candidate_address,
                         lambda address, symbol: self.decode(
                             observation.v[rows, address], state.survivors[rows, address], symbol),
                         self.source_key)


def snapshot_state(state):
    """Clone storage without detaching the permitted training computation graph."""
    if isinstance(state, SourceState):
        state.validate(state.hidden.shape[0])
        return replace(state, hidden=state.hidden.clone())
    if isinstance(state, AnalyticState):
        state.validate(state.survivors.shape[0])
        return replace(state, survivors=state.survivors.clone(),
                       revealed_symbols=state.revealed_symbols.clone())
    raise ValueError('unknown source state')


def patch_addressed_state(recipient, donor, address):
    """Reachable donor state must be separately replayed; no executor exists here."""
    if type(recipient) is not type(donor):
        raise ValueError('patch cannot cross source state types')
    patched = snapshot_state(recipient)
    batch_size = patched.hidden.shape[0] if isinstance(patched, SourceState) else patched.survivors.shape[0]
    donor.validate(batch_size)
    _indices(address, batch_size, 4, 'patch address')
    if (recipient.source_key != donor.source_key
            or recipient.revealed_through != donor.revealed_through
            or recipient.update_command_indices != donor.update_command_indices):
        raise ValueError('donor source/time/update indices must match recipient')
    rows = torch.arange(batch_size)
    if isinstance(patched, SourceState):
        patched.hidden[rows, address] = donor.hidden[rows, address]
    else:
        patched.survivors[rows, address] = donor.survivors[rows, address]
        patched.revealed_symbols[rows, address] = donor.revealed_symbols[rows, address]
    patched.validate(batch_size)
    return patched


def restore_state(saved_state):
    return snapshot_state(saved_state)


@dataclass(frozen=True)
class PredictedEvidence:
    local_logits: torch.Tensor
    role_fields: torch.Tensor


@dataclass(frozen=True)
class SymbolChoice:
    symbol: torch.Tensor
    maxima: torch.Tensor
    candidate_fields: torch.Tensor


def choose_from_maxima(maxima):
    if not isinstance(maxima, torch.Tensor) or maxima.ndim != 2 or maxima.shape[1] != 3:
        raise ValueError('three symbol maxima per world required')
    _float(maxima, maxima.shape, 'symbol maxima')
    return maxima.argmax(-1)  # torch argmax returns first index on exact ties.


class NativePredictionReadout:
    """Frozen H on raw predicted maps, itemwise; not an actual-W consumer."""
    def __init__(self, head, native_standardizer):
        if not isinstance(head, LocalOccupancyHead) or head.channels != 24:
            raise ValueError('native 24-channel H required')
        if not isinstance(native_standardizer, ChannelStandardizer) or len(native_standardizer.mean) != 24:
            raise ValueError('frozen native H standardizer required')
        self.head = deepcopy(head).cpu().eval()
        if any(parameter.dtype != torch.float32 for parameter in self.head.parameters()):
            raise ValueError('native H weights must be float32')
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        self.standardizer = native_standardizer

    @torch.no_grad()
    def consume(self, raw_maps):
        if not isinstance(raw_maps, torch.Tensor) or raw_maps.ndim < 3 or raw_maps.shape[-3:] != (24, 4, 4):
            raise ValueError('predicted maps must end in [24,4,4]')
        _float(raw_maps, raw_maps.shape, 'raw predicted maps')
        flat = raw_maps.reshape(-1, 24, 4, 4)
        if not flat.shape[0]:
            raise ValueError('empty predicted map readout')
        logits = torch.stack([self.head(self.standardizer.apply(item)) for item in flat])
        fields = torch.stack([role_fields(item) for item in logits])
        leading = raw_maps.shape[:-3]
        return PredictedEvidence(logits.reshape(*leading, 3, 4, 4),
                                 fields.reshape(*leading, 2, 4, 4))

    def choose(self, prediction, address, role):
        if not isinstance(prediction, PredictionBatch):
            raise ValueError('source PredictionBatch required; no actual-W argument')
        batch_size = prediction.validate()
        _indices(address, batch_size, 4, 'query address')
        _indices(role, batch_size, 2, 'query role')
        rows = torch.arange(batch_size)
        candidate_maps = prediction.next[rows[:, None], torch.arange(3)[None, :], address[:, None]]
        fields = self.consume(candidate_maps).role_fields
        maxima = fields[rows[:, None], torch.arange(3)[None, :], role[:, None]].flatten(-2).max(-1).values
        return SymbolChoice(choose_from_maxima(maxima), maxima, fields)
