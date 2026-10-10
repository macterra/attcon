"""Single-attempt registered synthetic attention-outcome qualification."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from grounded_outcome_data_v1 import collect, remap_data, FrozenSensor, COUNTS, REPEATS, SNAPSHOTS
from grounded_outcome_models_v1 import fit, calibrate, predict, KINDS
from grounded_outcome_assessment_v1 import assess, control
from orientation_transfer_diagnostic_v1 import SEEDS, checkpoint, digest
from verify_isolated_orientation_task_v1 import compare

ROOT = Path('audits/grounded_attention_outcomes_v1')
SOURCES = ('scripts/grounded_attention_outcomes_v1.py', 'scripts/grounded_outcome_data_v1.py',
 'scripts/grounded_outcome_models_v1.py', 'scripts/grounded_outcome_assessment_v1.py',
 'scripts/grounded_attention_world_v1.py', 'scripts/isolated_orientation_task_v1.py',
 'scripts/orientation_transfer_diagnostic_v1.py', 'scripts/verify_isolated_orientation_task_v1.py',
 'src/attcon/bound_content.py', 'docs/GROUNDED_ATTENTION_OUTCOMES_V1_PROTOCOL.md',
 'tests/test_grounded_outcome_data.py', 'tests/test_grounded_outcome_models.py',
 'tests/test_grounded_outcome_assessment.py', 'audits/grounded_attention_outcomes_v1/execution_wrapper.py')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def register():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'manifest.json').exists(), 'manifest already reserved'
    assert (ROOT/'design_review/review.md').exists() and (ROOT/'design_review/metadata.json').exists()
    inputs = [checkpoint(seed) for seed in SEEDS]+[Path(f'audits/sensory_task_adapter_v1/states-{seed}.pt.gz') for seed in SEEDS]
    manifest = {'kind': 'fresh_executed_attention_outcome_qualification', 'counts': COUNTS,
        'snapshots': SNAPSHOTS, 'repeats': REPEATS, 'visual_seeds': SEEDS, 'kinds': KINDS,
        'source_sha256': {p: digest(p) for p in SOURCES},
        'frozen_input_sha256': {str(p): digest(p) for p in inputs},
        'design_review_sha256': {str(p): digest(p) for p in (ROOT/'design_review').glob('*') if p.is_file()},
        'appearance_targets': 0, 'reporter_outputs': 0, 'goal_achieved': False}
    dump(ROOT/'source_snapshots.json', {p: Path(p).read_text() for p in SOURCES})
    dump(ROOT/'manifest.json', manifest)


def check_manifest():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, sha in {**manifest['source_sha256'], **manifest['frozen_input_sha256'], **manifest['design_review_sha256']}.items():
        assert digest(path) == sha, path
    snapshots = json.loads((ROOT/'source_snapshots.json').read_text())
    for path, source in snapshots.items():
        assert hashlib.sha256(source.encode()).hexdigest() == manifest['source_sha256'][path]


class Archive:
    def __init__(self, verify):
        self.verify = verify
        self.files = {}

    def write(self, relative, value):
        path = ROOT/(relative+'.pt.gz')
        if self.verify:
            with gzip.open(path, 'rb') as stream:
                expected = torch.load(stream, weights_only=True)
            compare(value, expected, str(path))
        else:
            assert not path.exists(), path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
                    torch.save(value, stream)
        assert path.stat().st_size < 100000000, path
        self.files[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}


def label_args(data):
    return tuple(data[key] for key in ('sensory', 'history', 'current_labels', 'refresh_labels', 'delivery_labels', 'after_labels'))


def remap_assessment(model, data):
    original = predict(model, data['restored']['sensory'], data['restored']['history'])
    changed = predict(model, data['sensory'], data['history'])
    restored = predict(model, data['restored']['sensory'], data['restored']['history'])
    exact = all(torch.equal(original[key], restored[key]) for key in original)
    correct = changed['delivery'].argmax(-1) == data['delivery_labels']
    return {'top1_by_observations': correct.double().mean((0, 2)).tolist(),
        'final_delivery_top1_pass': float(correct[:, -1].double().mean()) >= .95,
        'restoration_exact': exact, 'intermediate_histories_out_of_distribution': True}, {
        'original': original, 'remapped': changed, 'restored': restored}


def truth_diagnostics(data, worlds, forecasts):
    """Evaluator-only curves; inaccessible latent truth never qualifies inference."""
    result = {}
    for kind in ('current', 'refresh', 'after'):
        p = forecasts[kind]
        observed = data[kind+'_labels'].double()
        if kind != 'current': observed = observed.mean(2)
        result[kind] = {}
        for field, count in (('contrasts', 6), ('labels', 2)):
            values = worlds[field][:, None, :].expand_as(data['current_labels'])
            seen = data['sensory'][..., 65].bool()
            if kind == 'after':
                values = values.unsqueeze(-2).expand_as(p)
                seen = seen.unsqueeze(-2).expand_as(p)
            groups = {}
            for value in range(count):
                groups[str(value)] = {}
                for name, status in (('seen', seen), ('unseen', ~seen)):
                    mask = (values == value) & status
                    amount = int(mask.sum())
                    if not amount:
                        groups[str(value)][name] = {'prediction_count': 0}
                        continue
                    probability, y = p[mask], observed[mask]
                    bins = torch.bucketize(probability.contiguous(), torch.arange(1,10,dtype=torch.float64)/10, right=True)
                    ece = sum(float(((probability-y)*(bins==b)).sum().abs())/amount for b in range(10))
                    groups[str(value)][name] = {'prediction_count': amount,
                        'mean_probability': float(probability.mean()), 'accuracy': float(y.mean()),
                        'signed_residual': float((probability-y).mean()), 'ece': ece,
                        'brier': float((probability.square()-2*probability*y+y).mean())}
            result[kind][field] = groups
    return {'qualification_gates': False, 'future_latent_truth_is_not_available_to_inference': True, 'curves': result}


def execute(verify=False):
    check_manifest()
    archive = Archive(verify)
    models = {}
    for seed in SEEDS:
        print(f'collect {seed}', flush=True)
        sensor = FrozenSensor(seed)
        splits, worlds = [], []
        for split in range(3):
            prefix = f'model-{seed}/split-{split}/'
            data, world = collect(seed, split, sensor, lambda name, value: archive.write(prefix+name, value))
            splits.append(data); worlds.append(world)
        remap = remap_data(seed, splits[2], worlds[2], sensor,
            lambda name, value: archive.write(f'model-{seed}/'+name, value))
        instruments = {}
        for index, kind in enumerate(KINDS):
            print(f'fit {seed} {kind}', flush=True)
            fitted = fit(*label_args(splits[0]), seed=1010000000+seed+index*10000, kind=kind)
            qualified = calibrate(fitted, *label_args(splits[1]))
            forecasts = [predict(qualified, data['sensory'], data['history']) for data in splits]
            assessment = assess(splits[1], splits[2], forecasts[1], forecasts[2], 1020000000+seed, kind)
            controller, controller_archive = control(splits[2], forecasts[2], 1030000000+seed, kind)
            revision, revision_archive = remap_assessment(qualified, remap)
            gates = {**assessment['gates'], **controller.get('gates', {})}
            if kind == 'structured':
                gates.update(remap_final_delivery=revision['final_delivery_top1_pass'],
                             remap_restoration=revision['restoration_exact'])
            summary = {'assessment': assessment, 'controller': controller, 'remap': revision,
                'truth_diagnostics': truth_diagnostics(splits[2], worlds[2], forecasts[2]),
                'gates': gates, 'engineering_gates_pass': all(gates.values()) if kind == 'structured' else None,
                'parameter_count': fitted['parameter_count'], 'loss_trace': fitted['loss_trace']}
            instruments[kind] = summary
            archive.write(f'model-{seed}/instrument-{kind}', {'fitted': fitted, 'calibrated': qualified,
                'forecasts': forecasts, 'controller': controller_archive, 'remap': revision_archive})
        models[str(seed)] = instruments
    results = {'kind': 'synthetic_executed_attention_outcomes', 'models': models,
        'primary_all_engineering_gates_pass': all(m['structured']['engineering_gates_pass'] for m in models.values()),
        'appearance_tested': False, 'consciousness_related_correspondence_established': False,
        'reporter_tested': False, 'controller_is_reporter': False, 'goal_achieved': False,
        'independent_visual_models': len(SEEDS)}
    if verify:
        compare(results, json.loads((ROOT/'results.json').read_text()), 'results')
        assert archive.files == json.loads((ROOT/'completion.json').read_text())['archive_files']
        dump(ROOT/'replay.json', {'status': 'exact_collection_fit_calibration_forecasts_results_replay',
            'manifest_sha256': digest(ROOT/'manifest.json'), 'results_sha256': digest(ROOT/'results.json'),
            'archive_count': len(archive.files), 'all_files_verified': True})
    else:
        dump(ROOT/'results.json', results)
        dump(ROOT/'completion.json', {'archive_files': archive.files, 'results_sha256': digest(ROOT/'results.json'),
            'manifest_sha256': digest(ROOT/'manifest.json'), 'status': 'single_registered_attempt_completed'})
    print(json.dumps({'completed': True, 'verify': verify, 'primary_all_engineering_gates_pass': results['primary_all_engineering_gates_pass']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('register', 'run', 'verify'))
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.mode == 'register': register()
    elif args.mode == 'run':
        check_manifest()
        reservation = ROOT/'attempt.json'
        with reservation.open('x') as stream:
            json.dump({'manifest_sha256': digest(ROOT/'manifest.json'), 'single_attempt': True}, stream)
        execute()
    else: execute(verify=True)
