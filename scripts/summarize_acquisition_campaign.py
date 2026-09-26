"""Consolidate six registered acquisition cycles without upgrading prior claims."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {'environment': 'audits/acquisition_environment.json',
        'control': 'audits/acquisition_multiseed.json', 'interventions': 'audits/acquisition_interventions.json',
        'reporting': 'audits/acquisition_reporting_multiseed.json', 'stress': 'audits/acquisition_stress.json',
        'previous_campaign': 'audits/regulation_campaign.json', 'stage8': 'audits/stage8_convergence_current.json'}
    data = {name: json.loads((ROOT / path).read_text()) for name, path in paths.items()}
    seeds = {2309, 2333, 2351}
    for name in ('control', 'reporting'):
        if set(data[name]['seeds']) != seeds:
            raise ValueError('missing registered seeds')
    for name in ('environment', 'interventions', 'stress'):
        if set(map(int, data[name]['runs'])) != seeds:
            raise ValueError('missing registered seeds')
    if not data['interventions']['all_contrasts_valid'] or not all(run['disjoint_contexts'] for run in data['stress']['runs'].values()):
        raise ValueError('invalid controls')
    def bounds(values):
        return {'min': min(values), 'max': max(values)}
    stress = {}
    for condition in ('baseline', 'long_delay', 'degraded_sensor'):
        stress[condition] = {}
        for cost in ('0.1', '0.25', '0.4'):
            values = [run['conditions'][condition]['evaluation']['costs'][cost] for run in data['stress']['runs'].values()]
            stress[condition][cost] = {'state_return_range': bounds([v['policies']['state']['return'] for v in values]),
                'all_gate_pass_count': sum(v['all_gates_pass'] for v in values),
                'fresh_accuracy_pass_count': sum(v['gates']['fresh_accuracy'] for v in values),
                'forced_verification_pass_count': sum(v['gates']['forced_verification_accuracy'] for v in values)}
    counts = [cost['gate_pass_counts'] for cost in data['control']['costs'].values()]
    result = {'audit': 'active_inspection_campaign', 'completed_cycles': 6, 'seeds': sorted(seeds),
        'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths.values()},
        'adaptive_acquisition_beats_all_fixed_policies_all_costs_seeds': all(c['gain_over_' + name] == 3 for c in counts for name in ('never', 'once', 'twice')),
        'full_state_control_advantage_supported': data['control']['all_seeds_supported'],
        'full_state_report_advantage_supported': data['reporting']['all_seeds_supported'],
        'choice_preserving_contrasts_valid': data['interventions']['all_contrasts_valid'],
        'availability_transplant_inspection_switch_ranges': bounds([run['results']['transplants']['availability']['initial_switch_rate'] for run in data['interventions']['runs'].values()]),
        'initial_reset_return_change_range': bounds([run['results']['reset_return_change'] for run in data['interventions']['runs'].values()]),
        'reserved_stress': stress, 'stage8_recomputed': False,
        'interpretation': 'Task-trained recurrent policies acquire and verify information adaptively, but action-score/confidence controllers explain the benefit. Verified-information reporters are accurate and also matched by action logits. History matters causally; the answer-preserving fitted availability direction causes no policy switches. Longer delays and unannounced sensor degradation reduce reward. No claim of conscious access or Stage 8 support.'}
    (ROOT / 'audits/active_inspection_campaign.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
