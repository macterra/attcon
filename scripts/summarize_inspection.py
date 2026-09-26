"""Aggregate the registered inspection gate without selecting successful costs."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(paths):
    runs = [json.loads(path.read_text()) for path in paths]
    if len(runs) < 3 or len({run['seed'] for run in runs}) != len(runs):
        raise ValueError('requires at least three unique seeds')
    reference = runs[0]
    for run in runs:
        for field in ('audit', 'source_sha256', 'policy_parameters'):
            if run[field] != reference[field]:
                raise ValueError('incomparable ' + field)
        expected = {'task_viable': run['seen_choice_accuracy'] >= 0.85}
        if set(run['test']['costs']) != {'0.2', '0.4', '0.6'}:
            raise ValueError('wrong cost set')
        for cost, result in run['test']['costs'].items():
            intervals = result['state_minus_baseline_intervals']
            gates = {f'gain_over_{name}': intervals[name]['mean'] >= 0.02 for name in ('action', 'confidence')}
            gates.update({f'positive_bound_over_{name}': intervals[name]['low'] > 0 for name in ('action', 'confidence')})
            gates.update({f'beats_{name}': intervals[name]['mean'] > 0 for name in ('never_inspect', 'always_inspect')})
            gates['gain_over_null_p95'] = result['policies']['state']['return'] - result['null_p95'] >= 0.05
            if gates != result['gates'] or all(gates.values()) != result['all_gates_pass']:
                raise ValueError('inconsistent cost gates')
            expected['cost_' + cost] = all(gates.values())
        if expected != run['gates']:
            raise ValueError('inconsistent overall gates')
    all_gates = {key: all(run['gates'][key] for run in runs) for key in reference['gates']}
    def bounds(values):
        return {'min': min(values), 'max': max(values)}
    return {'audit': 'reward_trained_inspection_multiseed', 'seeds': [run['seed'] for run in runs],
        'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'status': 'replicated_bounded_inspection_advantage' if all(all_gates.values()) else 'inspection_advantage_gates_not_met',
        'all_seed_gates': all_gates,
        'gate_pass_counts': {key: sum(run['gates'][key] for run in runs) for key in all_gates},
        'costs': {cost: {'return_ranges': {name: bounds([run['test']['costs'][cost]['policies'][name]['return'] for run in runs]) for name in ('state', 'action', 'confidence')},
            'state_gain_ranges': {name: bounds([run['test']['costs'][cost]['state_minus_baseline_intervals'][name]['mean'] for run in runs]) for name in ('action', 'confidence', 'always_inspect', 'never_inspect')},
            'gate_pass_counts': {gate: sum(run['test']['costs'][cost]['gates'][gate] for run in runs) for gate in reference['test']['costs'][cost]['gates']}}
            for cost in reference['test']['costs']},
        'boundary': 'All predeclared costs and seeds retained; no test-based policy choice. One synthetic task and one frozen recurrent architecture. Reward-trained external policy does not establish endogenous recurrent control or subjective access.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.artifacts)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'gate_pass_counts': result['gate_pass_counts']}))
if __name__ == '__main__':
    main()
