"""Registered fit/dev/heldout phase runner; registration never imports torch.

Source draft. Actual execution requires a complete independently approved source
closure, committed input registration and immutable committed phase receipts.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1301, 1311, 1321)
PHASES = ('fit', 'dev', 'heldout')
SCHEMA = 'attention_content_model_v1_phase_registration'
REVIEW_SCOPE = 'attention_content_model_v1_complete_preexecution_review'
REQUIRED_SOURCES = (
    'scripts/run_attention_content_model_v1.py',
    'scripts/attention_content_model_v1.py',
    'scripts/attention_content_model_v1_data.py',
    'scripts/attention_content_model_v1_experiment.py',
    'scripts/attention_content_form_primitives_v1.py',
    'scripts/spatial_native_task_v1.py',
    'src/attcon/__init__.py', 'src/attcon/bound_content.py',
    'src/attcon/data.py', 'src/attcon/models.py',
    'src/attcon/predictive_attention.py',
    'docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md',
)
NATIVE_DIR = 'audits/spatial_native_task_v1/run'
NATIVE_REVIEW_DIR = 'audits/spatial_native_task_v1/numerical_corrected_review'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT), 'artifact must be inside repository')
    return path.relative_to(ROOT).as_posix()


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bound_files(commit, hashes):
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
            'preservation commit must be full lowercase hexadecimal')
    for path, expected in hashes.items():
        require(digest(ROOT / path) == expected, f'current binding failed: {path}')
        process = subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{path}'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        require(hashlib.sha256(process.stdout).hexdigest() == expected,
                f'committed binding failed: {path}')


def native_acceptance():
    """JSON/hash-only prerequisite, before any torch import or model load."""
    manifest_path = ROOT / NATIVE_REVIEW_DIR / 'metadata_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    packed = ROOT / manifest['published_path']
    require(digest(packed) == manifest['compressed_sha256'], 'native review transport changed')
    raw = gzip.decompress(packed.read_bytes())
    require(hashlib.sha256(raw).hexdigest() == manifest['raw_sha256'], 'native review raw identity failed')
    review = json.loads(raw)
    require(review['status'] == 'passed_corrected_reference_revalidation'
            and review['check_count'] == 206070 and review['error_count'] == 0
            and review['core_check_inventory_matches_first_review'], 'native prerequisite failed')
    results = json.loads((ROOT / NATIVE_DIR / 'results.json').read_text())
    require(results['all_engineering_feasibility_gates_pass'], 'native feasibility gate failed')
    completion = json.loads((ROOT / NATIVE_DIR / 'completion.json').read_text())
    require(completion['status'] == 'completed', 'native run incomplete')
    hashes = {path: digest(ROOT / path) for path in (
        f'{NATIVE_REVIEW_DIR}/metadata_manifest.json', str(packed.relative_to(ROOT)),
        f'{NATIVE_DIR}/results.json', f'{NATIVE_DIR}/completion.json')}
    for seed in SEEDS:
        path = f'{NATIVE_DIR}/states-{seed}.pt.gz'
        expected = completion['archive_sha256'][path]
        require(digest(ROOT / path) == expected, 'native model input archive changed')
        hashes[path] = expected
    return hashes


def register(args):
    require(args.source_commit and args.review, 'registration needs exact source commit and complete review')
    hashes = {path: digest(ROOT / path) for path in REQUIRED_SOURCES}
    review_path = relative(args.review)
    review = json.loads((ROOT / review_path).read_text())
    require(review['scope'] == REVIEW_SCOPE and review['approved'] is True
            and review['approved_source_sha256'] == hashes, 'complete source review does not bind exact closure')
    hashes[review_path] = digest(ROOT / review_path)
    bound_files(args.source_commit, hashes)
    inputs = native_acceptance()
    registration = {'schema': SCHEMA, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': args.source_commit, 'source_hashes': hashes, 'input_hashes': inputs,
        'output_dir': relative(args.output_dir), 'encoder_seeds': list(SEEDS),
        'worlds_per_encoder_phase': 96, 'parameter_count_per_learned_source': 197328,
        'updates_per_source': 300, 'phases': list(PHASES),
        'source_inputs': ['V', 'issued commands', 'completed feedback through t-1'],
        'forbidden_source_inputs': ['pixels', 'mapping labels', 'bandwidth truth',
            'renderer truth', 'current concealed W_t', 'future fork outcomes', 'world identifiers'],
        'phase_receipts_required_before_release': True, 'reporter_calls': 0,
        'no_fit_retry_or_adaptation': True, 'development_only': True,
        'source_semantics_qualified': False, 'presentation_qualified': False}
    write_exclusive(args.registration, registration)
    print('Registered reviewed input bytes; no torch import, rendering or fit.')


def receipt(base, phase, commit, registration_sha256):
    path = base / phase / 'receipt.json'
    value = json.loads(path.read_text())
    require(value['status'] == 'completed' and value['phase'] == phase, f'{phase} phase incomplete')
    require(value['registration_sha256'] == registration_sha256,
            f'{phase} receipt belongs to a different input registration')
    hashes = {relative(path): digest(path), **value['artifact_sha256']}
    bound_files(commit, hashes)
    return value


def load_native(seed):
    """Reconstruct exact frozen modules from the accepted native archive only."""
    import torch
    from attcon.bound_content import VisualEncoder
    from spatial_native_task_v1 import ChannelStandardizer, FrozenSpatialExtractor, LocalOccupancyHead
    with gzip.open(ROOT / NATIVE_DIR / f'states-{seed}.pt.gz', 'rb') as stream:
        trace = torch.load(stream, map_location='cpu', weights_only=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoder = VisualEncoder().float().eval()
        orientation = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).float().eval()
        head = LocalOccupancyHead().float().eval()
    encoder.load_state_dict(trace['frozen_source_weights']['encoder'])
    orientation.load_state_dict(trace['frozen_source_weights']['orientation_head'])
    head.load_state_dict(trace['native_head_fit']['weights'])
    orientation_coordinates = trace['frozen_orientation_standardizer']
    extractor = FrozenSpatialExtractor(encoder, orientation, orientation_coordinates['mean'], orientation_coordinates['scale'])
    coordinates = trace['native_head_fit']['fit_only_standardizer']
    return extractor, head, ChannelStandardizer(tuple(coordinates['mean']), tuple(coordinates['scale']))


def save_tensor_archive(path, value):
    import torch
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            torch.save(value, stream)


def enforce_software_checks(checks):
    """Hard invariants are separate from descriptive prediction measurements."""
    import torch
    leaves = 0
    def visit(value):
        nonlocal leaves
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, bool):
            leaves += 1
            require(value, 'failed engine software invariant')
        elif isinstance(value, torch.Tensor) and value.dtype == torch.bool and value.numel():
            leaves += value.numel()
            require(bool(value.all()), 'failed engine per-opportunity invariant')
        else:
            raise ValueError('software checks must be nonempty bool values/tensors')
    visit(checks)
    require(leaves > 0, 'engine returned no software invariants')
    return leaves


def execute_phase(args):
    require(args.registration_commit, 'phase requires preserved input registration')
    registration_path = relative(args.registration)
    registration = json.loads((ROOT / registration_path).read_text())
    require(registration['schema'] == SCHEMA, 'wrong registration schema')
    bound_files(args.registration_commit, {registration_path: digest(ROOT / registration_path),
        **registration['source_hashes'], **registration['input_hashes']})
    require(native_acceptance() == registration['input_hashes'], 'native inputs changed after registration')
    base = ROOT / registration['output_dir']
    fit_receipt = None
    if args.phase in ('dev', 'heldout'):
        require(args.fit_commit, 'dev/heldout requires preserved immutable fit receipt')
        fit_receipt = receipt(base, 'fit', args.fit_commit, digest(ROOT / registration_path))
    if args.phase == 'heldout':
        require(args.dev_commit and args.assessment, 'heldout requires preserved dev receipt and assessment')
        dev_receipt = receipt(base, 'dev', args.dev_commit, digest(ROOT / registration_path))
        assessment_path = relative(args.assessment)
        bound_files(args.dev_commit, {assessment_path: digest(ROOT / assessment_path)})
        assessment = json.loads((ROOT / assessment_path).read_text())
        require(assessment['hard_invariants_pass'] is True
            and assessment['dev_receipt_sha256'] == digest(base / 'dev' / 'receipt.json')
            and assessment['fitted_artifact_sha256'] == fit_receipt['artifact_sha256'], 'dev acceptance mismatch')
    phase_dir = base / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    # The exclusive claim precedes torch/model imports or any released generation.
    write_exclusive(phase_dir / 'claim.json', {'phase': args.phase,
        'registration_sha256': digest(ROOT / registration_path), 'registration_commit': args.registration_commit,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'retry_allowed': False})
    partial = {'phase': args.phase, 'seed': None}
    try:
        import torch
        from attention_content_model_v1 import FitNormalizer, build_learned_sources
        from attention_content_model_v1_data import collect_worlds
        from attention_content_model_v1_experiment import fit_sources, evaluate_sources
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        outputs = {}
        with (phase_dir / 'first_run.log').open('x') as log:
            def emit(value):
                print(json.dumps(value, allow_nan=False), file=log, flush=True)
                print(json.dumps(value, allow_nan=False), flush=True)
            for seed in SEEDS:
                partial = {'phase': args.phase, 'seed': seed}
                extractor, head, coordinates = load_native(seed)
                episode = collect_worlds(seed, args.phase, extractor, released_split=args.phase)
                partial['episode'] = episode
                if args.phase == 'fit':
                    fitted = fit_sources(seed, episode, log=emit)
                    fitted = {key: value for key, value in fitted.items() if key != 'models'}
                    fitted['normalizer'] = asdict(fitted['normalizer'])
                    bundle = {'episode': episode, 'fit': fitted}
                else:
                    with gzip.open(base / 'fit' / f'states-{seed}.pt.gz', 'rb') as stream:
                        fitted = torch.load(stream, map_location='cpu', weights_only=False)['fit']
                    normalizer = FitNormalizer(**{key: tuple(value) for key, value in fitted['normalizer'].items()})
                    attention, ordinary, alias = build_learned_sources(normalizer, seed)
                    models = {'A': attention, 'ordinary': ordinary}
                    for source, model in models.items():
                        model.load_state_dict(fitted['checkpoints'][source])
                        model.eval()
                    with torch.no_grad():
                        evaluation = evaluate_sources(models, episode, head, coordinates, extractor, args.phase, log=emit)
                    bundle = {'episode': episode, 'evaluation': evaluation,
                        'fitted_artifact_sha256': fit_receipt['artifact_sha256']}
                partial = bundle
                output = phase_dir / f'states-{seed}.pt.gz'
                save_tensor_archive(output, bundle)
                if args.phase != 'fit':
                    enforce_software_checks(evaluation['software_checks'])
                outputs[relative(output)] = digest(output)
                emit({'seed': seed, 'phase': args.phase, 'archive_sha256': digest(output)})
        outputs[relative(phase_dir / 'first_run.log')] = digest(phase_dir / 'first_run.log')
        write_exclusive(phase_dir / 'receipt.json', {'status': 'completed', 'phase': args.phase,
            'registration_sha256': digest(ROOT / registration_path), 'artifact_sha256': outputs,
            'fit_receipt_sha256': digest(base / 'fit' / 'receipt.json') if fit_receipt else None,
            'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'retry_performed': False,
            'no_reporter_calls': True, 'semantic_qualification_asserted': False})
    except BaseException:
        save_tensor_archive(phase_dir / 'first_failure.pt.gz', partial)
        write_exclusive(phase_dir / 'failure.json', {'status': 'failed_first_phase_execution',
            'phase': args.phase, 'traceback': traceback.format_exc(), 'retry_allowed': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--register', action='store_true')
    modes.add_argument('--phase', choices=PHASES)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--source-commit')
    parser.add_argument('--review', type=Path)
    parser.add_argument('--registration-commit')
    parser.add_argument('--fit-commit')
    parser.add_argument('--dev-commit')
    parser.add_argument('--assessment', type=Path)
    args = parser.parse_args()
    if args.register:
        require(args.output_dir, 'registration requires fixed output directory')
        register(args)
    else:
        sys.path.insert(0, str(ROOT / 'src'))
        execute_phase(args)


if __name__ == '__main__':
    main()
