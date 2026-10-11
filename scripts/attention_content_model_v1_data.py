"""Evaluator-only episode construction for the registered learned-content study.

Importing defines functions only. The phase runner must bind reviewed bytes and
release a split before calling collect_worlds. Mapping labels, seeds, pixels and
targets returned here are evaluator records, never source inference arguments.
"""
from itertools import permutations

import torch

from attention_content_form_primitives_v1 import block_average
from spatial_native_task_v1 import render_world

ENCODER_SEEDS = (1301, 1311, 1321)
SPLITS = ('fit', 'dev', 'heldout')
WORLDS = 96
MAPPINGS = tuple(permutations((1, 2, 4)))


def _indices(seed, split):
    if seed not in ENCODER_SEEDS or split not in SPLITS:
        raise ValueError('only the registered encoder seeds and split names are allowed')
    return SPLITS.index(split)


def schedule(seed, split):
    """Fixed balanced mappings and mapping-independent commands; no rendering."""
    split_index = _indices(seed, split)
    mapping_index = torch.empty(WORLDS, 4, dtype=torch.long)
    for address in range(4):
        generator = torch.Generator().manual_seed(
            330000000 + seed * 1000 + split_index * 100000 + address)
        mapping_index[:, address] = torch.arange(6).repeat_interleave(16)[
            torch.randperm(WORLDS, generator=generator)]
    mapping = torch.tensor(MAPPINGS, dtype=torch.long)[mapping_index]
    addresses = torch.empty(WORLDS, 15, dtype=torch.long)
    symbols = torch.empty_like(addresses)
    roles = torch.full_like(addresses, -1)
    for world in range(WORLDS):
        per_address = []
        for address in range(4):
            generator = torch.Generator().manual_seed(
                340000000 + seed * 1000 + split_index * 100000 + world * 4 + address)
            per_address.append(torch.randperm(3, generator=generator))
        for round_index in range(3):
            for address in range(4):
                time = round_index * 4 + address
                addresses[world, time] = address
                symbols[world, time] = per_address[address][round_index]
        for query in range(3):
            time = 12 + query
            addresses[world, time] = (world + query) % 4
            symbols[world, time] = (world + query) % 3
            roles[world, time] = (world + query) % 2
    return {'evaluator_mapping_indices': mapping_index, 'evaluator_mappings': mapping,
            'command_addresses': addresses, 'command_symbols': symbols,
            'query_roles': roles}


def operator_family(v):
    """Actual executor alternatives, not an analytic shortcut in learned A."""
    if (v.shape[-4:] != (4, 24, 4, 4) or v.dtype != torch.float32
            or v.device.type != 'cpu' or not torch.isfinite(v).all()):
        raise ValueError('expected finite CPU float32 [...,4,24,4,4] V')
    items = v.reshape(-1, 24, 4, 4)
    alternatives = torch.stack([torch.stack([block_average(item, band)
        for band in (1, 2, 4)]) for item in items])
    return alternatives.reshape(*v.shape[:-3], 3, 24, 4, 4)


def execute_command(v, completed_w, addresses, symbols, evaluator_mappings):
    """A completed fork replaces an address from V, never from degraded W."""
    if (v.ndim != 5 or v.shape[1:] != (4, 24, 4, 4) or completed_w.shape != v.shape
            or v.dtype != torch.float32 or v.device.type != 'cpu'
            or not torch.isfinite(v).all()):
        raise ValueError('executor requires [B,4,24,4,4] V and completed W')
    batch = v.shape[0]
    if (addresses.shape != (batch,) or symbols.shape != (batch,)
            or addresses.dtype != torch.long or symbols.dtype != torch.long
            or evaluator_mappings.shape != (batch, 4, 3)
            or ((addresses < 0) | (addresses >= 4)).any()
            or ((symbols < 0) | (symbols >= 3)).any()):
        raise ValueError('invalid executor command or evaluator mapping')
    if (completed_w.dtype != torch.float32 or not torch.isfinite(completed_w).all()
            or completed_w.device.type != 'cpu'):
        raise ValueError('invalid actual completed W')
    result = completed_w.clone()
    for world in range(batch):
        address, symbol = int(addresses[world]), int(symbols[world])
        band = int(evaluator_mappings[world, address, symbol])
        result[world, address] = block_average(v[world, address], band)
    return result


def actual_episode(v, command_addresses, command_symbols, evaluator_mappings):
    """Shared actual history and all three forks; sources never receive this object."""
    if command_addresses.shape != (v.shape[0], 15) or command_symbols.shape != command_addresses.shape:
        raise ValueError('exactly fifteen transitions are required')
    completed = [v.clone()]
    forks = []
    for time in range(15):
        address = command_addresses[:, time]
        alternatives = torch.stack([
            execute_command(v, completed[-1], address,
                            torch.full_like(address, symbol), evaluator_mappings)
            for symbol in range(3)], dim=1)
        forks.append(alternatives)
        row = torch.arange(v.shape[0])
        completed.append(alternatives[row, command_symbols[:, time]].clone())
    return {'actual_completed_w': torch.stack(completed, dim=1),
            'actual_forks': torch.stack(forks, dim=1)}


def donor_prefix(v, episode_schedule, decision_time, patched_addresses):
    """Reachable donor with one mapping changed; conceal W_t in source replay.

    Returned completed states are evaluator data. Source replay must consume only
    transitions 0..t-2 (W1..W(t-1)), not the final donor completion W_t.
    """
    if decision_time not in (12, 13, 14):
        raise ValueError('donors exist only at the three registered query times')
    batch = v.shape[0]
    if (patched_addresses.shape != (batch,) or patched_addresses.dtype != torch.long
            or ((patched_addresses < 0) | (patched_addresses >= 4)).any()):
        raise ValueError('invalid donor address')
    indices = episode_schedule['evaluator_mapping_indices'].clone()
    row = torch.arange(batch)
    indices[row, patched_addresses] = (indices[row, patched_addresses] + 1) % 6
    mappings = torch.tensor(MAPPINGS, dtype=torch.long)[indices]
    completed = [v.clone()]
    for time in range(decision_time):
        completed.append(execute_command(v, completed[-1],
            episode_schedule['command_addresses'][:, time],
            episode_schedule['command_symbols'][:, time], mappings))
    return {'evaluator_mapping_indices': indices, 'evaluator_mappings': mappings,
            'actual_completed_w': torch.stack(completed, dim=1),
            'executor_command_indices': tuple(range(decision_time)),
            'permitted_feedback_command_indices': tuple(range(decision_time - 1)),
            'decision_time': decision_time, 'patched_addresses': patched_addresses.clone()}


def collect_worlds(seed, split, extractor, released_split):
    """Phase runner supplies a verified release; this function cannot release one."""
    split_index = _indices(seed, split)
    if released_split != split:
        raise ValueError('requested split is not released')
    namespace = 310000000 + seed * 1000 + split_index * 100000
    worlds = [render_world(namespace, index) for index in range(WORLDS)]
    pixels = torch.stack([world.pixels for world in worlds])
    with torch.no_grad():
        v = extractor.extract(pixels).detach().clone()
    episode_schedule = schedule(seed, split)
    return {'encoder_seed': seed, 'split': split, 'world_namespace': namespace,
            'world_indices': torch.arange(WORLDS), 'pixels': pixels, 'V': v,
            'evaluator_cells': torch.tensor([world.truth.cells for world in worlds]),
            'render_nuisance': [{'token_shapes': world.token_shapes,
                                'intensities': world.intensities,
                                'noise_sigma': world.noise_sigma} for world in worlds],
            **episode_schedule, **actual_episode(v, episode_schedule['command_addresses'],
                episode_schedule['command_symbols'], episode_schedule['evaluator_mappings'])}
