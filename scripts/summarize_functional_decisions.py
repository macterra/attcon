"""Descriptive paired analysis of retained decisions; no new trials or gates."""
import csv
import json
from pathlib import Path
import torch
from run_functional_decisions import ROOT, POLICIES, read_trace
from run_functional_controls import SEEDS, digest


def summarize():
    rows = []
    for seed in SEEDS:
        trace = read_trace(ROOT/f'seed{seed}_decisions.pt.gz')
        for condition, context in trace['contexts'].items():
            baseline = context['policies']['model_guided']
            for policy in POLICIES:
                paired = context['policies'][policy]
                count = len(baseline)*len(baseline[0]['query'])
                def differences(left, right):
                    return sum(int((left(b) != right(t)).reshape(len(b['query']), -1).any(-1).sum())
                               for b, t in zip(baseline, paired))
                row = {'seed': seed, 'condition': condition, 'policy': policy,
                    'paired_trial_branches': count,
                    'command_change_count': differences(lambda t:t['command'], lambda t:t['command']),
                    'task_buffer_allocation_change_count': differences(
                        lambda t:t['allocation'][:, 0], lambda t:t['allocation'][:, 0]),
                    'task_buffer_physical_recovery_change_count': differences(
                        lambda t:t['physical_recovery'][:, 0], lambda t:t['physical_recovery'][:, 0]),
                    'task_buffer_modeled_recovery_change_count': differences(
                        lambda t:t['updated_access'][:, 0, 0], lambda t:t['updated_access'][:, 0, 0]),
                    'response_change_count': differences(lambda t:t['answer']['response'],
                                                        lambda t:t['answer']['response']),
                    'physical_threshold_response_change_count': differences(
                        lambda t:t['physical_answer']['response'], lambda t:t['physical_answer']['response']),
                    'raw_category_change_count': differences(lambda t:t['answer']['labels'],
                                                            lambda t:t['answer']['labels'])}
                if condition in (1, -1):
                    assert row['task_buffer_allocation_change_count'] == 0
                    assert row['task_buffer_physical_recovery_change_count'] == 0
                    assert row['physical_threshold_response_change_count'] == 0
                if policy == 'restored_guided':
                    assert all(v == 0 for k,v in row.items() if k.endswith('_count') and k != 'paired_trial_branches')
                rows.append(row)
    return rows


def main():
    torch.set_num_threads(2)
    summary = json.loads((ROOT/'summary.json').read_text())
    for seed in SEEDS:
        assert digest(ROOT/f'seed{seed}_decisions.pt.gz') == summary['trace_sha256'][str(seed)]
    rows = summarize()
    output = ROOT/'paired_policy_changes.csv'
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    rendered = stream.getvalue()
    if output.exists():
        assert output.read_bytes() == rendered.encode(), 'existing paired analysis differs'
    else:
        output.write_bytes(rendered.encode())
    provenance = {'kind': 'post_hoc_descriptive_paired_policy_analysis',
        'source_sha256': {str(Path(__file__).relative_to(Path.cwd())): digest(__file__)},
        'dependency_sha256': {str(ROOT/'summary.json'): digest(ROOT/'summary.json'),
            **{str(ROOT/f'seed{seed}_decisions.pt.gz'): summary['trace_sha256'][str(seed)] for seed in SEEDS}},
        'output_sha256': {str(output): digest(output)}, 'new_trials': 0, 'new_success_gate': None}
    provenance_path = ROOT/'paired_analysis.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text()) == provenance
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'All {len(rows)} paired change rows retained/reproduced; task-buffer physical invariance checked')


if __name__ == '__main__':
    main()
