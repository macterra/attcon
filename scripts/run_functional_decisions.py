"""Freeze, execute and replay explicit category-task Controller branches."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import torch
from attcon.functional_decisions import (
    DecisionWorld, answer_query, execute_decision, joint_confidence, plan_command)
from attcon.functional_model import buffer_contents
from attcon.functional_controls import CONDITIONS, counterfactual_recovery
from attcon.predictive_attention import Forecast, PredictiveAttention
from run_functional_controls import SEEDS, digest, save_trace
from verify_functional_controls import compare as compare_value

ROOT = Path('audits/functional_decisions_v1')
INTERFACE = Path('audits/functional_interface_v1')
CONTROL = Path('audits/functional_controls_v1')
POLICIES = ('model_guided', 'random_command', 'swapped_effect_guided', 'restored_guided')
SOURCES = ('src/attcon/functional_decisions.py', 'scripts/run_functional_decisions.py',
           'tests/test_functional_decisions.py', 'docs/FUNCTIONAL_DECISIONS_PROTOCOL.md',
           'src/attcon/functional_model.py', 'src/attcon/functional_controls.py',
           'src/attcon/predictive_attention.py', 'scripts/run_functional_controls.py',
           'scripts/verify_functional_controls.py')


def compare(expected, actual, path='trace'):
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f'{path}/{index}')
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), path
        for key in expected:
            compare(expected[key], actual[key], f'{path}/{key}')
    else:
        compare_value(expected, actual, path)


def read_trace(path):
    with gzip.open(path, 'rb') as stream:
        return torch.load(io.BytesIO(stream.read()), weights_only=True)


def prepare():
    if (ROOT/'manifest.json').exists():
        return json.loads((ROOT/'manifest.json').read_text())
    dependencies = [INTERFACE/'summary.json', CONTROL/'summary.json',
                    Path('audits/functional_readout_audit_v1/summary.json')]
    for seed in SEEDS:
        dependencies.extend((INTERFACE/f'seed{seed}_states.pt.gz',
                             CONTROL/f'seed{seed}_trace.pt.gz', CONTROL/f'seed{seed}.pt'))
    manifest = {'kind': 'post_hoc_engineering_category_decision_assay',
        'seeds': SEEDS, 'conditions': CONDITIONS, 'policies': POLICIES,
        'episodes_per_condition': 512, 'query_positions': [0, 1, 2, 3],
        'world_steps_per_trial': 1, 'acquisition_quality': .8,
        'random_command_seed_rule': '790000000 + model_seed',
        'random_commands_shared_across_conditions': True,
        'answer_threshold': .6, 'answer_rounding_decimal_places': 5,
        'response_recovery_attenuation': .25,
        'new_language_reports': 0, 'new_training': False, 'new_success_gate': None,
        'no_overwrite_or_retry': True, 'torch_version': torch.__version__,
        'source_sha256': {p: digest(p) for p in SOURCES},
        'dependency_sha256': {str(p): digest(p) for p in dependencies}}
    ROOT.mkdir(parents=True)
    (ROOT/'source_code.json').write_text(json.dumps(
        {p: Path(p).read_text() for p in SOURCES}, indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def check_manifest(manifest):
    for p, h in {**manifest['source_sha256'], **manifest['dependency_sha256']}.items():
        assert digest(p) == h, p
    snapshots = json.loads((ROOT/'source_code.json').read_text())
    import hashlib
    for p, h in manifest['source_sha256'].items():
        assert hashlib.sha256(snapshots[p].encode()).hexdigest() == h, p


def metrics(seed, condition, policy, trials):
    def cat(key):
        return torch.cat([t[key] for t in trials])
    answered = torch.cat([t['answer']['answered'] for t in trials])
    response = torch.cat([t['answer']['response'] for t in trials])
    raw = torch.cat([t['answer']['labels'] for t in trials])
    truth = cat('truth')
    physical_answered = torch.cat([t['physical_answer']['answered'] for t in trials])
    correct = (response == truth).all(-1)
    selected_scores = cat('physical_selected_score')
    oracle_scores = cat('physical_oracle_score')
    return {'seed': seed, 'condition': condition, 'policy': policy,
        'trial_count': len(answered), 'answer_count': int(answered.sum()),
        'abstention_count': int((~answered).sum()),
        'correct_answer_count': int((answered & correct).sum()),
        'incorrect_answer_count': int((answered & ~correct).sum()),
        'answer_fraction': float(answered.double().mean()),
        'accuracy_when_answering': float(correct[answered].double().mean()) if answered.any() else None,
        'raw_argmax_accuracy': float((raw == truth).all(-1).double().mean()),
        'physical_target_acquired_fraction': float(cat('target_acquired').double().mean()),
        'physical_query_recovery_mean': float(cat('physical_query_recovery').double().mean()),
        'physical_threshold_answer_fraction': float(physical_answered.double().mean()),
        'modeled_physical_answer_mismatch_fraction': float((answered != physical_answered).double().mean()),
        'physical_confidence_regret_mean': float((oracle_scores-selected_scores).double().mean()),
        'attenuated_answer_count': int(torch.cat([t['attenuated_answer']['answered'] for t in trials]).sum()),
        'attenuation_preserves_raw_labels': all(torch.equal(t['answer']['labels'],
            t['attenuated_answer']['labels']) for t in trials),
        'response_restoration_exact': all(t['response_restoration_exact'] for t in trials)}


@torch.no_grad()
def build(seed):
    inputs = read_trace(INTERFACE/f'seed{seed}_states.pt.gz')
    world_inputs = read_trace(CONTROL/f'seed{seed}_trace.pt.gz')
    model = PredictiveAttention()
    model.load_state_dict(torch.load(CONTROL/f'seed{seed}.pt', weights_only=True)['state_dict'])
    model.eval()
    visual = inputs['visual'][:, 0].clone()
    generator = torch.Generator().manual_seed(790000000+seed)
    random_commands = torch.randint(4, (512, 4), generator=generator)
    trace = {'visual': visual, 'random_commands': random_commands, 'contexts': {}}
    rows = torch.arange(512)
    records = []
    for condition in CONDITIONS:
        state = inputs['static'][condition]['neutral']
        source = world_inputs['static'][condition]
        current = Forecast(state['allocation'], state['access'], state['effects'])
        hidden = state['hidden']
        sequence, recomputed_hidden = model(source['observations'])
        assert torch.equal(recomputed_hidden, hidden)
        assert torch.equal(sequence.access[:, -1], current.access)
        assert torch.equal(sequence.effects[:, -1], current.effects)
        altered = current.intervene('effects', current.effects.flip(-2))
        restored = altered.intervene('effects', current.effects)
        future = counterfactual_recovery(model, current, hidden)[:, :, 0]
        swapped_future = counterfactual_recovery(model, altered, hidden)[:, :, 0]
        restored_future = counterfactual_recovery(model, restored, hidden)[:, :, 0]
        assert torch.equal(future, state['prospective_recovery'][:, :, 0])
        assert torch.equal(swapped_future, inputs['static'][condition]['model_swap']['prospective_recovery'][:, :, 0])
        assert torch.equal(future, restored_future)
        world = DecisionWorld(source['replay'], source['direction'], source['access'][:, -1, 0],
                              torch.full((512,), condition, dtype=torch.long))
        physical_future = source['physical_recovery'][:, :, 0]
        physical_future_content = buffer_contents(visual[:, None], physical_future)
        context = {'initial_allocation': current.allocation.clone(),
            'initial_access': current.access.clone(), 'initial_effects': current.effects.clone(),
            'initial_hidden': hidden.clone(), 'initial_observations': source['observations'].clone(),
            'initial_world': {'replay': world.replay.clone(), 'direction': world.direction.clone(),
                'recovery': world.recovery.clone(), 'controlled': world.controlled.clone()},
            'predicted_recovery': future, 'swapped_predicted_recovery': swapped_future,
            'restored_predicted_recovery': restored_future,
            'physical_all_command_recovery': physical_future.clone(), 'policies': {}}
        for policy in POLICIES:
            policy_future = swapped_future if policy == 'swapped_effect_guided' else (
                restored_future if policy == 'restored_guided' else future)
            trials = []
            for position in range(4):
                query = torch.full((512,), position, dtype=torch.long)
                planned, scores = plan_command(visual, policy_future, query)
                command = random_commands[:, position] if policy == 'random_command' else planned
                before = answer_query(visual, current.access[:, 0, 0], query)
                updated, next_hidden, executed = execute_decision(model, hidden, world, command)
                answer = answer_query(visual, updated.access[:, 0, 0], query)
                changed_access = updated.access.clone()
                changed_access[:, :, 0] *= .25
                changed = updated.intervene('access', changed_access)
                attenuated = answer_query(visual, changed.access[:, 0, 0], query)
                restored_readout = changed.intervene('access', updated.access)
                response_restored = answer_query(visual, restored_readout.access[:, 0, 0], query)
                compare(answer, response_restored, 'response_restoration')
                assert torch.equal(changed.allocation, updated.allocation)
                assert torch.equal(changed.effects, updated.effects)
                physical_answer = answer_query(visual, executed['physical_recovery'][:, 0], query)
                physical_scores = joint_confidence(physical_future_content)[rows, :, query]
                selected_physical_score = physical_scores[rows, command]
                assert torch.equal(executed['physical_recovery'][:, 0], physical_future[rows, command])
                trial = {'query': query, 'truth': torch.stack(
                    (inputs['colors'][:, position], inputs['shapes'][:, position]), -1),
                    'planning_scores': scores, 'planned_command': planned, **executed,
                    'updated_allocation': updated.allocation, 'updated_access': updated.access,
                    'updated_effects': updated.effects, 'updated_hidden': next_hidden,
                    'before_answer': before, 'answer': answer, 'attenuated_access': changed.access,
                    'attenuated_answer': attenuated, 'restored_answer': response_restored,
                    'response_restoration_exact': True, 'physical_answer': physical_answer,
                    'target_acquired': executed['allocation'][rows, 0, query] == 1,
                    'physical_query_recovery': executed['physical_recovery'][rows, 0, query],
                    'physical_selected_score': selected_physical_score,
                    'physical_oracle_score': physical_scores.max(-1).values}
                trials.append(trial)
            context['policies'][policy] = trials
            records.append(metrics(seed, condition, policy, trials))
        compare(context['policies']['model_guided'], context['policies']['restored_guided'], 'command_restoration')
        assert torch.equal(current.access, context['initial_access'])
        assert torch.equal(hidden, context['initial_hidden'])
        assert torch.equal(world.recovery, context['initial_world']['recovery'])
        trace['contexts'][condition] = context
    return trace, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'run', 'verify'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = prepare()
    check_manifest(manifest)
    if args.stage == 'prepare':
        print('Prepared explicit Controller decision assay; no task trials executed')
        return
    attempt = ROOT/'attempt.json'
    if args.stage == 'run':
        if attempt.exists():
            raise SystemExit('refusing to overwrite or retry decision assay')
        attempt.write_text(json.dumps({'status': 'reserved', 'new_language_reports': 0})+'\n')
    else:
        assert json.loads(attempt.read_text())['status'] == 'completed'
    all_metrics, archives = [], {}
    for seed in SEEDS:
        trace, records = build(seed)
        path = ROOT/f'seed{seed}_decisions.pt.gz'
        if args.stage == 'run':
            save_trace(trace, path)
        else:
            compare(trace, read_trace(path), f'seed{seed}')
        all_metrics.extend(records)
        archives[str(seed)] = digest(path)
        print(f'{seed}: {len(records)} policy/condition contexts '+(
            'executed and retained' if args.stage == 'run' else 'exactly replayed'), flush=True)
    summary = {'status': 'completed', 'kind': manifest['kind'], 'metrics': all_metrics,
        'contexts': len(all_metrics), 'task_trial_branches': sum(r['trial_count'] for r in all_metrics),
        'independent_training_models': len(SEEDS), 'reused_episodes_per_model': 512,
        'new_language_reports': 0, 'new_success_gate': None,
        'command_restoration_exact': True,
        'response_restoration_exact': all(r['response_restoration_exact'] for r in all_metrics),
        'attenuation_preserves_raw_labels': all(r['attenuation_preserves_raw_labels'] for r in all_metrics),
        'trace_sha256': archives}
    if args.stage == 'run':
        (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        with (ROOT/'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_metrics[0]))
            writer.writeheader(); writer.writerows(all_metrics)
        attempt.write_text(json.dumps({'status': 'completed', 'run_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode().strip(), 'new_language_reports': 0})+'\n')
    else:
        compare(summary, json.loads((ROOT/'summary.json').read_text()), 'summary')
        with (ROOT/'metrics.csv').open(newline='') as stream:
            actual_rows = list(csv.DictReader(stream))
        expected_rows = [{k: '' if v is None else str(v) for k, v in r.items()} for r in all_metrics]
        compare(expected_rows, actual_rows, 'csv')
        print('Every physical/model/action/response array, metric and archive hash replayed exactly')


if __name__ == '__main__':
    main()
