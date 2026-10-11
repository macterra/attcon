"""Copied mechanical spatial-state primitives; no learned model or presentation claim.

Immutable tuple storage owns V/W bytes numerically; tensor access returns fresh CPU
copies. Block averaging is lossy. Restoration copies a saved state, not an inverse.
"""
from dataclasses import dataclass, replace
import math
from typing import Mapping

import torch

BANDWIDTHS = (1, 2, 4)
CURRENT_SCHEMA = 'attention_content_form_actual_current_v1'
PROSPECTIVE_SCHEMA = 'attention_content_form_operator_alternative_v1'


def _time(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def _bandwidth(value):
    if type(value) is not int or value not in BANDWIDTHS:
        raise ValueError('bandwidth must be 1, 2 or 4')


def _map(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError('spatial map must be a [channels,height,width] tensor')
    if not value.is_floating_point() or min(value.shape) < 1:
        raise ValueError('spatial map must be nonempty and floating point')
    if not torch.isfinite(value).all():
        raise ValueError('spatial map must be finite')
    return value.detach().cpu().clone()


def block_average(value, bandwidth):
    """Nonoverlapping block means, repeated into original shape; no padding/edges.

    Arithmetic uses float64 with predivision to avoid overflowing finite block
    sums; the returned map retains input dtype. Bandwidth 1 is an exact copy.
    """
    _bandwidth(bandwidth)
    value = _map(value)
    channels, height, width = value.shape
    if height % bandwidth or width % bandwidth:
        raise ValueError('height and width must be divisible by bandwidth')
    if bandwidth == 1:
        return value
    grouped = value.double().reshape(channels, height//bandwidth, bandwidth,
                                     width//bandwidth, bandwidth)
    means = (grouped / (bandwidth * bandwidth)).sum((2, 4))
    result = means.repeat_interleave(bandwidth, 1).repeat_interleave(bandwidth, 2)
    result = result.to(value.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('block-average output must be finite')
    return result


@dataclass(frozen=True)
class SpatialBuffer:
    shape: tuple[int, int, int]
    values: tuple[float, ...]
    dtype: torch.dtype

    def __post_init__(self):
        if (type(self.shape) is not tuple or len(self.shape) != 3
                or any(type(x) is not int or x < 1 for x in self.shape)
                or type(self.values) is not tuple
                or len(self.values) != math.prod(self.shape)
                or any(type(x) is not float or not math.isfinite(x) for x in self.values)
                or self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
            raise ValueError('invalid immutable spatial buffer')
        if not torch.isfinite(self.tensor()).all():
            raise ValueError('buffer values must be finite in their stored dtype')

    @classmethod
    def from_tensor(cls, value):
        value = _map(value)
        return cls(tuple(value.shape), tuple(value.flatten().tolist()), value.dtype)

    def tensor(self):
        return torch.tensor(self.values, dtype=self.dtype).reshape(self.shape)


@dataclass(frozen=True)
class AddressedItem:
    address: str
    source_v: SpatialBuffer
    actual_w: SpatialBuffer
    observed: bool
    identity: str | None
    raw_logit: float
    delivered_at: int | None
    w_completed_at: int | None

    def __post_init__(self):
        if not isinstance(self.address, str) or not self.address:
            raise ValueError('address must be a nonempty string')
        if not isinstance(self.source_v, SpatialBuffer) or not isinstance(self.actual_w, SpatialBuffer):
            raise ValueError('V and W must be immutable spatial buffers')
        if self.source_v.shape != self.actual_w.shape or self.source_v.dtype != self.actual_w.dtype:
            raise ValueError('V/W shape and dtype must match')
        if any(x % 4 for x in self.source_v.shape[-2:]):
            raise ValueError('state maps must support the complete 1,2,4 bandwidth family')
        if type(self.observed) is not bool or type(self.raw_logit) is not float or not math.isfinite(self.raw_logit):
            raise ValueError('invalid observed flag or raw logit')
        if self.observed:
            if not isinstance(self.identity, str) or not self.identity:
                raise ValueError('observed item needs a literal identity record')
            _time(self.delivered_at, 'delivered_at')
            _time(self.w_completed_at, 'w_completed_at')
            if self.w_completed_at < self.delivered_at:
                raise ValueError('W cannot precede the retained delivery')
        elif self.identity is not None or self.delivered_at is not None or self.w_completed_at is not None:
            raise ValueError('unseen item has no observed identity or event times')

    @classmethod
    def from_tensors(cls, address, source_v, actual_w, observed, identity,
                     raw_logit, delivered_at, w_completed_at):
        return cls(address, SpatialBuffer.from_tensor(source_v), SpatialBuffer.from_tensor(actual_w),
                   observed, identity, raw_logit, delivered_at, w_completed_at)


@dataclass(frozen=True)
class CompletedEvent:
    address: str
    bandwidth: int
    issued_at: int
    completed_at: int
    driver: str  # Audit-only provenance; absent from current/prospective exports.

    def __post_init__(self):
        if not isinstance(self.address, str) or not self.address:
            raise ValueError('event needs an address')
        _bandwidth(self.bandwidth)
        _time(self.issued_at, 'issued_at')
        _time(self.completed_at, 'completed_at')
        if self.completed_at <= self.issued_at or self.driver not in ('endogenous', 'exogenous'):
            raise ValueError('invalid event ordering or driver')


@dataclass(frozen=True)
class AddressedState:
    items: tuple[AddressedItem, ...]
    time: int
    history: tuple[CompletedEvent, ...] = ()

    def __post_init__(self):
        _time(self.time, 'state time')
        if (type(self.items) is not tuple or not self.items
                or any(not isinstance(item, AddressedItem) for item in self.items)):
            raise ValueError('state needs an immutable nonempty item tuple')
        if len({item.address for item in self.items}) != len(self.items):
            raise ValueError('duplicate address')
        if len({(item.source_v.shape, item.source_v.dtype) for item in self.items}) != 1:
            raise ValueError('all addressed maps must have equal shape and dtype')
        if any(item.observed and item.w_completed_at > self.time for item in self.items):
            raise ValueError('item completion is later than current snapshot')
        if type(self.history) is not tuple or any(not isinstance(e, CompletedEvent) for e in self.history):
            raise ValueError('history must be an immutable event tuple')
        previous = -1
        for event in self.history:
            if event.completed_at > self.time or event.issued_at < previous:
                raise ValueError('history is not a completed chronological sequence')
            if not self.item(event.address).observed:
                raise ValueError('history refers to an unseen item')
            previous = event.completed_at

    def item(self, address):
        for item in self.items:
            if item.address == address:
                return item
        raise ValueError('unknown address')


@dataclass(frozen=True)
class PendingOperator:
    address: str
    bandwidth: int
    issued_at: int
    prospective_at: int

    def __post_init__(self):
        if not isinstance(self.address, str) or not self.address:
            raise ValueError('operator needs an address')
        _bandwidth(self.bandwidth)
        _time(self.issued_at, 'issued_at')
        _time(self.prospective_at, 'prospective_at')
        if self.prospective_at <= self.issued_at:
            raise ValueError('prospective completion must follow issue time')


def _target(state, pending):
    if not isinstance(state, AddressedState) or not isinstance(pending, PendingOperator):
        raise ValueError('state and pending operator required')
    if pending.issued_at != state.time:
        raise ValueError('pending issue time must match the current snapshot')
    item = state.item(pending.address)
    if not item.observed:
        raise ValueError('cannot operate on an unseen item')
    return item


def _execute(state, pending, driver):
    item = _target(state, pending)
    # Always transform retained V, never cumulatively transform a lossy prior W.
    updated = replace(item, actual_w=SpatialBuffer.from_tensor(
        block_average(item.source_v.tensor(), pending.bandwidth)),
        w_completed_at=pending.prospective_at)
    event = CompletedEvent(item.address, pending.bandwidth, pending.issued_at,
                           pending.prospective_at, driver)
    return AddressedState(tuple(updated if x.address == item.address else x for x in state.items),
                          pending.prospective_at, state.history + (event,))


def execute_endogenous(state, pending):
    """Execute the declared internal operation; no psychological attention claim."""
    return _execute(state, pending, 'endogenous')


def execute_exogenous(state, pending):
    """Matched mechanism control: identical T/V, separately audited driver label."""
    return _execute(state, pending, 'exogenous')


def restore_saved(saved_state):
    """Restore a saved snapshot including W, event times and history; no T inverse."""
    if not isinstance(saved_state, AddressedState):
        raise ValueError('saved state required')
    items = tuple(replace(x, source_v=SpatialBuffer.from_tensor(x.source_v.tensor()),
                          actual_w=SpatialBuffer.from_tensor(x.actual_w.tensor()))
                  for x in saved_state.items)
    return AddressedState(items, saved_state.time, tuple(replace(e) for e in saved_state.history))


def _rows(state):
    return [{'address': item.address, 'observed': item.observed, 'identity_record': item.identity,
             'raw_task_logit': item.raw_logit, 'delivered_at': item.delivered_at,
             'actual_w_completed_at': item.w_completed_at,
             'actual_current_w': item.actual_w.tensor().tolist() if item.observed else None}
            for item in state.items]


def current_export(state):
    """Native actual current W; no next-state estimate, learned A, or provenance cue."""
    if not isinstance(state, AddressedState):
        raise ValueError('state required')
    return {'schema_version': CURRENT_SCHEMA, 'snapshot_time': state.time,
            'quantity': 'actual_completed_internal_spatial_map', 'items': _rows(state)}


def prospective_export(state, pending):
    """Pure known-operator alternative; does not execute or record a future event."""
    item = _target(state, pending)
    return {'schema_version': PROSPECTIVE_SCHEMA, 'source_snapshot_time': state.time,
            'issued_at': pending.issued_at, 'prospective_at': pending.prospective_at,
            'address': pending.address, 'bandwidth': pending.bandwidth,
            'quantity': 'analytic_operator_result_not_observed_or_learned',
            'operator_result_from_saved_v': block_average(item.source_v.tensor(), pending.bandwidth).tolist(),
            'learned_prediction': False}


def known_operator_baseline(state, command, mapping: Mapping, prospective_at):
    """Analytic comparator with evaluator-supplied mapping; explicitly an oracle.

    This function does not estimate an unknown command mapping from history.
    No empirical mapping learner has been implemented or qualified here.
    """
    if not isinstance(state, AddressedState) or not isinstance(mapping, Mapping):
        raise ValueError('state and explicit mapping required')
    if command not in mapping:
        raise ValueError('unknown command')
    operation = mapping[command]
    if not isinstance(operation, tuple) or len(operation) != 2:
        raise ValueError('mapping entry must be (address, bandwidth)')
    pending = PendingOperator(operation[0], operation[1], state.time, prospective_at)
    result = prospective_export(state, pending)
    result.update(baseline_kind='analytic_known_operator',
                  mapping_knowledge='oracle_not_empirically_learned')
    return result


def audit_history(state):
    """Audit-only completed operation provenance, excluded from native exports."""
    if not isinstance(state, AddressedState):
        raise ValueError('state required')
    return [{'address': e.address, 'bandwidth': e.bandwidth, 'issued_at': e.issued_at,
             'completed_at': e.completed_at, 'driver': e.driver} for e in state.history]
