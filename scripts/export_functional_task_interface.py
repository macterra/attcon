"""Freeze/export/replay neutral task views from already executed Controller traces."""
import argparse
import gzip
import itertools
import json
from pathlib import Path
import subprocess
import torch
from attcon.functional_task_interface import task_record, remap_task, without_task_relation
from attcon.functional_controls import counterfactual_recovery
from attcon.predictive_attention import Forecast, PredictiveAttention
from run_functional_controls import SEEDS, digest
from run_functional_decisions import ROOT as DECISIONS, read_trace

ROOT = Path('audits/functional_task_interface_v1')
INTERFACE = Path('audits/functional_interface_v1')
CONTROL = Path('audits/functional_controls_v1')
ORDERS = tuple(itertools.permutations(range(3)))
ROWS = tuple(range(6))
VARIANTS = ('before_execution', 'after_execution', 'attenuated_readout',
            'restored_readout', 'remapped', 'missing_relation')
SOURCES = ('scripts/export_functional_task_interface.py',
    'src/attcon/functional_task_interface.py', 'tests/test_functional_task_interface.py',
    'docs/FUNCTIONAL_TASK_INTERFACE_PROTOCOL.md', 'docs/FUNCTIONAL_TASK_SOURCE_INVENTORY.md',
    'src/attcon/functional_interface.py', 'src/attcon/functional_model.py',
    'src/attcon/functional_controls.py', 'src/attcon/predictive_attention.py',
    'src/attcon/bound_content.py', 'scripts/run_functional_decisions.py',
    'scripts/run_functional_controls.py', 'scripts/verify_functional_controls.py')


def prepare():
    if (ROOT/'manifest.json').exists():
        return json.loads((ROOT/'manifest.json').read_text())
    dependencies = [DECISIONS/'summary.json', DECISIONS/'manifest.json']
    for seed in SEEDS:
        dependencies.extend((DECISIONS/f'seed{seed}_decisions.pt.gz',
            INTERFACE/f'seed{seed}_states.pt.gz', CONTROL/f'seed{seed}.pt'))
    manifest = {'kind': 'development_export_of_executed_task_inputs', 'seeds': SEEDS,
        'rows': ROWS, 'queries': [0, 1, 2, 3], 'node_orders': ORDERS, 'variants': VARIANTS,
        'policies': ['model_guided', 'random_command', 'swapped_effect_guided', 'restored_guided'],
        'conditions': [0, 1, -1], 'expected_records': 5184,
        'new_language_reports': 0, 'new_task_trials': 0, 'new_training': False,
        'new_success_gate': None, 'no_overwrite_or_retry': True,
        'source_sha256': {p: digest(p) for p in SOURCES},
        'dependency_sha256': {str(p): digest(p) for p in dependencies},
        'torch_version': torch.__version__}
    ROOT.mkdir(parents=True)
    (ROOT/'source_code.json').write_text(json.dumps({p: Path(p).read_text() for p in SOURCES}, indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def check_manifest(manifest):
    import hashlib
    snapshots = json.loads((ROOT/'source_code.json').read_text())
    for p, h in {**manifest['source_sha256'], **manifest['dependency_sha256']}.items():
        assert digest(p) == h, p
    for p, h in manifest['source_sha256'].items():
        assert hashlib.sha256(snapshots[p].encode()).hexdigest() == h, p


def inspect_payload(payload):
    forbidden = {'condition', 'controlled', 'owner', 'physical_owner', 'policy', 'variant',
        'truth', 'correct', 'score', 'gates', 'physical_recovery', 'physical_replay', 'seed'}
    def walk(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value), forbidden.intersection(value)
            for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
    walk(payload)
    task = payload['task_decision']
    assert task['query']['node'] == payload['output_node']
    nodes = {node['node']: node for node in payload['predicted_current']}
    assert 'selection_distribution' not in nodes[payload['output_node']]
    response = task['response']
    if task['command_executed']:
        assert response['status'] in ('answered', 'abstained')
        if payload['observed_history'] is not None:
            assert task['selected_command'] == payload['observed_history'][-1]['command']
        obj = nodes[payload['output_node']]['objects'][int(task['query']['position'][1:])]
        known = obj['identified_color'] is not None and obj['identified_shape'] is not None
        assert (response['status'] == 'answered') == known
        if known:
            assert response['color'] == obj['identified_color']
            assert response['shape'] == obj['identified_shape']
        else:
            assert response['color'] is None and response['shape'] is None
    else:
        assert response == {'status': 'pending', 'color': None, 'shape': None}


@torch.no_grad()
def build():
    samples, index = [], {}
    for seed in SEEDS:
        trace = read_trace(DECISIONS/f'seed{seed}_decisions.pt.gz')
        prior = read_trace(INTERFACE/f'seed{seed}_states.pt.gz')
        model = PredictiveAttention()
        model.load_state_dict(torch.load(CONTROL/f'seed{seed}.pt', weights_only=True)['state_dict'])
        model.eval()
        visual = prior['visual']
        assert torch.equal(visual[:, 0], trace['visual'])
        for condition, context in trace['contexts'].items():
            initial = Forecast(context['initial_allocation'], context['initial_access'], context['initial_effects'])
            observed = context['initial_observations']
            hidden = context['initial_hidden']
            for policy, trials in context['policies'].items():
                planned_state = initial.intervene('effects', initial.effects.flip(-2)) if policy == 'swapped_effect_guided' else initial
                prospective = counterfactual_recovery(model, planned_state, hidden)
                expected = context['swapped_predicted_recovery'] if policy == 'swapped_effect_guided' else context['predicted_recovery']
                assert torch.equal(prospective[:, :, 0], expected)
                for position, trial in enumerate(trials):
                    current = Forecast(trial['updated_allocation'], trial['updated_access'], trial['updated_effects'])
                    altered = current.intervene('access', trial['attenuated_access'])
                    restored = altered.intervene('access', current.access)
                    history = torch.cat((observed, trial['observed'][:, None]), 1)
                    next_hidden = trial['updated_hidden']
                    future = counterfactual_recovery(model, current, next_hidden)
                    # An access-head intervention does not alter effects/hidden, so future
                    # estimates regenerate from the identical transition inputs.
                    altered_future = counterfactual_recovery(model, altered, next_hidden)
                    restored_future = counterfactual_recovery(model, restored, next_hidden)
                    assert torch.equal(future, altered_future) and torch.equal(future, restored_future)
                    for row in ROWS:
                        order = ORDERS[row]
                        query, command = trial['query'], trial['command']
                        before = task_record(visual, planned_state, prospective, observed,
                                             row, query, command, order=order)
                        after = task_record(visual, current, future, history, row, query, command,
                            response=trial['answer']['response'], executed=True, order=order)
                        attenuated = task_record(visual, altered, altered_future, history, row, query, command,
                            response=trial['attenuated_answer']['response'], executed=True, order=order)
                        restored_payload = task_record(visual, restored, restored_future, history, row, query, command,
                            response=trial['restored_answer']['response'], executed=True, order=order)
                        assert restored_payload == after
                        remapped = remap_task(after)
                        inverse_nodes = {'q7':'n0', 'q2':'n1', 'q9':'n2'}
                        inverse_commands = {'m7':'k0', 'm2':'k1', 'm9':'k2', 'm4':'k3'}
                        assert remap_task(remapped, inverse_nodes, inverse_commands) == after
                        removed = without_task_relation(after)
                        assert removed['task_decision'] == after['task_decision']
                        assert removed['predicted_current'] == after['predicted_current']
                        for key in ('observed_history', 'anticipated_history', 'predicted_by_command'):
                            assert removed[key] is None
                        variants = dict(zip(VARIANTS, (before, after, attenuated, restored_payload,
                                                      remapped, removed)))
                        for variant, payload in variants.items():
                            inspect_payload(payload)
                            key = f'{seed}_c{condition}_{policy}_p{position}_r{row}_{variant}'
                            samples.append({'id': key, 'seed': seed, 'condition': condition,
                                'policy': policy, 'position': position, 'row': row, 'order': order,
                                'variant': variant, 'payload': payload})
                            index[(seed, condition, policy, position, row, variant)] = payload
                        if policy == 'restored_guided':
                            for variant, payload in variants.items():
                                assert payload == index[(seed, condition, 'model_guided', position, row, variant)]
        print(f'{seed}: executed-task sources connected to neutral inputs', flush=True)
    examples = []
    choices = [(0, 'before_execution'), (0, 'after_execution'), (0, 'attenuated_readout'),
               (0, 'restored_readout'), (1, 'after_execution'), (-1, 'after_execution')]
    for number, (condition, variant) in enumerate(choices, 1):
        examples.append({'example': f'x{number}', 'payload': index[(2011, condition, 'model_guided', 0, 0, variant)]})
    checks = {'records': len(samples), 'new_language_reports': 0, 'new_task_trials': 0,
        'all_six_node_orders_per_context': True, 'all_four_queries_per_row': True,
        'response_matches_supplied_readout_rule': True, 'executed_commands_match_actual_history': True,
        'response_restoration_payload_exact': True, 'command_restoration_payload_exact': True,
        'remapping_reversible': True, 'no_evaluator_fields_in_payloads': True,
        'missing_relation_preserves_current_task_and_state': True,
        'transient_access_intervention_future_invariance': True}
    assert len(samples) == 5184
    return samples, examples, checks


def encoded(value):
    return (json.dumps(value, separators=(',', ':'))+'\n').encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'export', 'verify'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = prepare(); check_manifest(manifest)
    if args.stage == 'prepare':
        print('Prepared task-input export; no new task execution or language calls')
        return
    attempt = ROOT/'attempt.json'
    if args.stage == 'export':
        if attempt.exists(): raise SystemExit('refusing overwrite or retry of task-input export')
        attempt.write_text(json.dumps({'status': 'reserved', 'new_language_reports': 0})+'\n')
    else:
        assert json.loads(attempt.read_text())['status'] == 'completed'
    samples, examples, checks = build()
    records_path, examples_path = ROOT/'records.json.gz', ROOT/'examples.json'
    compressed = gzip.compress(encoded(samples), mtime=0)
    rendered_examples = json.dumps(examples, indent=2)+'\n'
    if args.stage == 'export':
        records_path.write_bytes(compressed)
        examples_path.write_text(rendered_examples)
    else:
        assert records_path.read_bytes() == compressed
        assert examples_path.read_text() == rendered_examples
    summary = {'status': 'completed', 'kind': manifest['kind'], 'checks': checks,
        'record_archive_sha256': digest(records_path), 'record_archive_bytes': records_path.stat().st_size,
        'examples_sha256': digest(examples_path), 'new_success_gate': None}
    if args.stage == 'export':
        (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        attempt.write_text(json.dumps({'status': 'completed', 'run_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode().strip(), 'new_language_reports': 0})+'\n')
    else:
        assert json.loads((ROOT/'summary.json').read_text()) == summary
        print('All 5,184 payloads and compressed/example bytes reproduced exactly')


if __name__ == '__main__':
    main()
