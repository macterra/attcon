"""Aggregate external verified-information reporting without changing its gates."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(p.read_text()) for p in args.artifacts]
    if len(runs) < 3 or len({r['seed'] for r in runs}) != len(runs):
        raise ValueError('requires three unique seeds')
    reference = runs[0]
    for run in runs:
        if run['source_sha256'] != reference['source_sha256'] or run['parameters'] != reference['parameters'] or len(set(run['parameters'].values())) != 1:
            raise ValueError('incomparable reporters')
        primary = run['reports']['state']['all_stages']
        expected = {'verified_accuracy': primary['verified_value_accuracy'] >= .90,
            'unverified_accuracy': primary['unverified_accuracy'] >= .90,
            'gain_over_action': primary['balanced_accuracy'] - run['reports']['action']['all_stages']['balanced_accuracy'] >= .02,
            'gain_over_null_p95': primary['balanced_accuracy'] - run['null_p95'] >= .10}
        if expected != run['gates'] or all(expected.values()) != run['all_gates_pass'] or len(run['null_balanced_accuracies']) != 10:
            raise ValueError('invalid reporting gates or null count')
    def bounds(values):
        return {'min': min(values), 'max': max(values)}
    result = {'audit': 'acquisition_verified_reporting_multiseed', 'seeds': [r['seed'] for r in runs],
        'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.artifacts},
        'all_seeds_supported': all(r['all_gates_pass'] for r in runs),
        'gate_pass_counts': {name: sum(r['gates'][name] for r in runs) for name in reference['gates']},
        'ranges': {family: {cohort: {metric: bounds([r['reports'][family][cohort][metric] for r in runs]) for metric in ('verified_value_accuracy', 'unverified_accuracy', 'balanced_accuracy')} for cohort in ('all_stages', 'visited_stages')} for family in ('state', 'action')},
        'boundary': 'Accurate decoding of environmental verification is distinct from internal access or conscious reporting. Action-logit comparators remain part of every support decision. Earlier reporting gates and Stage 8 are unchanged.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'gate_pass_counts': result['gate_pass_counts']}))
if __name__ == '__main__':
    main()
