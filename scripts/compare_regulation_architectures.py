"""Describe near-parameter-matched architectures without inferring necessity."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gru', nargs='+', type=Path, required=True)
    parser.add_argument('--rnn', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    def read(paths):
        values = [json.loads(p.read_text()) for p in paths]
        if len({v['settings']['seed'] for v in values}) != len(values):
            raise ValueError('duplicate seeds')
        return {v['settings']['seed']: v for v in values}
    gru, rnn = read(args.gru), read(args.rnn)
    if set(gru) != set(rnn) or len(gru) < 3:
        raise ValueError('requires three paired seeds')
    pairs = {}
    for seed in gru:
        a, b = gru[seed], rnn[seed]
        if a['settings']['architecture'] != 'gru' or b['settings']['architecture'] != 'rnn_matched':
            raise ValueError('wrong architectures')
        for field in ('dataset_sha256', 'delay_assignment_sha256', 'source_sha256', 'thresholds', 'report_probe_parameters'):
            if a[field] != b[field]:
                raise ValueError('mismatched ' + field)
        for field in ('recipe', 'epochs', 'probe_steps', 'null_fits', 'train_groups', 'fit_groups'):
            if a['settings'][field] != b['settings'][field]:
                raise ValueError('mismatched ' + field)
        for field in ('updates', 'recurrent_example_steps'):
            if a['training'][field] != b['training'][field]:
                raise ValueError('mismatched ' + field)
        pairs[str(seed)] = {'controls_valid': True,
            'agent_parameters': {'gru': a['agent_parameters'], 'rnn': b['agent_parameters']},
            'parameter_difference': b['agent_parameters'] - a['agent_parameters'],
            'observed': {'gru': a['observed'], 'rnn': b['observed']},
            'gate_counts': {'gru': sum(a['gates'].values()), 'rnn': sum(b['gates'].values())},
            'delay9_choice_accuracy': {'gru': a['delay_slices']['9']['action']['seen_accuracy'], 'rnn': b['delay_slices']['9']['action']['seen_accuracy']}}
    result = {'audit': 'regulation_architecture_comparison', 'pairs': pairs,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.gru + args.rnn},
        'boundary': 'Near parameter counts, same data, delay assignments, updates, recurrent example steps, and reporter capacity. Different hidden dimensions and optimization landscapes remain. One recipe cannot establish a necessity of gating or a universal architectural limit.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'pairs': len(pairs)}))
if __name__ == '__main__':
    main()
