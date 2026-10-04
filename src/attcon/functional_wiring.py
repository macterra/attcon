"""Matched causal fixture; no experiential labels or learned model claims."""
import itertools
import random

CONDITIONS = ('own_access', 'external_device', 'decoupled')
ORDERS = tuple(itertools.permutations(range(3)))


def fixture(seed, condition, order=(0, 1, 2), initial_only=False):
    if condition not in CONDITIONS or tuple(sorted(order)) != (0, 1, 2):
        raise ValueError('invalid condition or presentation order')
    rng = random.Random(seed)
    categories = [rng.randrange(4) for _ in range(4)]
    quality = [.6 + .35 * rng.random() for _ in range(4)]
    schedule = rng.randrange(4)
    initial = [[.25] * 4 for _ in range(4)]
    controlled = {'own_access': 0, 'external_device': 1, 'decoupled': None}[condition]
    names = {physical: f'n{shown}' for shown, physical in enumerate(order)}

    def rows(buffers):
        values = buffers + [buffers[0]]
        return [{'node': names[p], 'values': values[p]} for p in order]

    payload = {'initial': rows([initial, initial]), 'trials': []}
    if initial_only:
        return payload
    for command in range(4):
        buffers = []
        for buffer in range(2):
            slot = command if buffer == controlled else schedule
            values = [row.copy() for row in initial]
            values[slot] = [(1-quality[slot]) / 4 + quality[slot] * (category == categories[slot])
                            for category in range(4)]
            buffers.append(values)
        payload['trials'].append({'command': f'k{command}', 'nodes': rows(buffers),
                                  'output_node': names[2]})
    return payload


def identify(payload):
    """Only an engineering observer of measured command consequences."""
    trials = payload['trials']
    if len(trials) < 2:
        return 'insufficient_evidence'
    baseline = {r['node']: r['values'] for r in trials[0]['nodes']}
    varying = {r['node'] for trial in trials[1:] for r in trial['nodes']
               if r['values'] != baseline[r['node']]}
    if trials[0]['output_node'] in varying:
        return 'own_access'
    return 'external_device' if varying else 'decoupled'
