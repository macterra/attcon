"""Aggregate acquisition comparisons while retaining all costs and seed failures."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(paths):
    runs = [json.loads(path.read_text()) for path in paths]
    if len(runs) < 3 or len({r['seed'] for r in runs}) != len(runs):
        raise ValueError('requires three unique seeds')
    reference = runs[0]
    for run in runs:
        if run['source_sha256'] != reference['source_sha256']:
            raise ValueError('incomparable source fingerprints')
        if set(run['test']['costs']) != {'0.1', '0.25', '0.4'}:
            raise ValueError('wrong costs')
        for field in ('initial_sha256', 'updates', 'parameters'):
            if len({record[field] for record in run['training'].values()}) != 1:
                raise ValueError('unmatched controller ' + field)
        for value in run['test']['costs'].values():
            intervals = value['state_minus_comparator_intervals']
            gates = {f'gain_over_{name}': interval['mean'] >= .02 for name, interval in intervals.items()}
            gates.update({f'positive_bound_over_{name}': intervals[name]['low'] > 0 for name in ('action', 'confidence')})
            gates['fresh_accuracy'] = value['policies']['state']['conditions']['fresh']['accuracy'] >= .90
            gates['forced_verification_accuracy'] = value['policies']['twice']['accuracy'] >= .90
            if gates != value['gates'] or all(gates.values()) != value['all_gates_pass']:
                raise ValueError('unearned acquisition gate')
        if run['test']['all_gates_pass'] != all(v['all_gates_pass'] for v in run['test']['costs'].values()):
            raise ValueError('inconsistent overall gate')
    def bounds(values):
        return {'min': min(values), 'max': max(values)}
    costs = {}
    for cost, template in reference['test']['costs'].items():
        costs[cost] = {'return_ranges': {name: bounds([r['test']['costs'][cost]['policies'][name]['return'] for r in runs]) for name in template['policies']},
            'state_gain_ranges': {name: bounds([r['test']['costs'][cost]['state_minus_comparator_intervals'][name]['mean'] for r in runs]) for name in template['state_minus_comparator_intervals']},
            'gate_pass_counts': {name: sum(r['test']['costs'][cost]['gates'][name] for r in runs) for name in template['gates']}}
    return {'audit': 'recurrent_acquisition_multiseed', 'seeds': [r['seed'] for r in runs],
        'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'all_seeds_supported': all(r['test']['all_gates_pass'] for r in runs), 'costs': costs,
        'boundary': 'One finite-horizon acquisition task; all registered costs and seeds retained. Recurrent fitted control is trained from environmental answer rewards and transitions. No conscious-experience or Stage 8 claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.artifacts)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_seeds_supported': result['all_seeds_supported'], 'costs': result['costs']}, indent=2))
if __name__ == '__main__':
    main()
