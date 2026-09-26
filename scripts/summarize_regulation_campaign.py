"""Consolidate the completed delay/reporting/inspection campaign conservatively."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {
        'delay': 'audits/regulation_delay_replication.json',
        'gru': 'audits/regulation_gru_variable_multiseed.json',
        'rnn': 'audits/regulation_rnn_variable_multiseed.json',
        'architectures': 'audits/regulation_architecture_comparison.json',
        'report_interventions': 'audits/regulation_report_interventions.json',
        'inspection': 'audits/inspection_multiseed.json',
        'inspection_interventions': 'audits/inspection_interventions.json',
        'stage8': 'audits/stage8_convergence_current.json',
    }
    data = {key: json.loads((ROOT / path).read_text()) for key, path in paths.items()}
    seeds = {2101, 2111, 2129}
    for name in ('gru', 'rnn', 'inspection'):
        if set(data[name]['seeds']) != seeds:
            raise ValueError('missing registered seeds: ' + name)
    for name, field in (('delay', 'pairs'), ('architectures', 'pairs'), ('report_interventions', 'runs'), ('inspection_interventions', 'runs')):
        if set(map(int, data[name][field])) != seeds:
            raise ValueError('missing registered seeds: ' + name)
    paired_valid = all(pair['controls_valid'] for name in ('delay', 'architectures') for pair in data[name]['pairs'].values())
    interventions_valid = data['report_interventions']['valid_choice_null_contrasts'] and data['inspection_interventions']['all_contrasts_valid']
    if not paired_valid or not interventions_valid:
        raise ValueError('invalid comparison controls')
    result = {'audit': 'regulation_campaign', 'completed_cycles': 6, 'seeds': sorted(seeds),
        'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths.values()},
        'paired_controls_valid': paired_valid, 'choice_preserving_contrasts_valid': interventions_valid,
        'delay9_choice_improves_all_seeds': all(pair['delay_choice_changes']['9'] > 0 for pair in data['delay']['pairs'].values()),
        'paired_report_improvement_lower_bound_positive_all_seeds': all(pair['mixed_report_change_intervals']['paired_accuracy']['low'] > 0 for pair in data['delay']['pairs'].values()),
        'gru_all_seed_gate_count': sum(data['gru']['all_seed_gates'].values()),
        'reporting_supported': all(data['gru']['all_seed_gates'].values()),
        'rnn_task_viable_all_seeds': data['rnn']['all_seed_gates']['seen_choice_accuracy'],
        'inspection_advantage_supported': all(data['inspection']['all_seed_gates'].values()),
        'inspection_state_beats_learned_comparators_all_seeds_and_costs': all(value['state_gain_ranges'][name]['min'] > 0 for value in data['inspection']['costs'].values() for name in ('action', 'confidence')),
        'stage8_recomputed': False,
        'interpretation': 'Variable-delay training improves task robustness and paired reporting. Full reporting gates remain unmet. The nearly parameter-matched RNN recipe is not task-viable. Reports and reward-trained policies have extra state sensitivity at fixed choice logits, but learned action/confidence inspection policies outperform full-state policies. No endogenous recurrent regulation or consciousness claim; Stage 8 unchanged.'}
    path = ROOT / 'audits/regulation_campaign.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
