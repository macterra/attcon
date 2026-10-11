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
import math
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
    'scripts/verify_attention_content_model_v1.py',
    'scripts/attention_content_form_primitives_v1.py',
    'scripts/spatial_native_task_v1.py',
    'src/attcon/__init__.py', 'src/attcon/bound_content.py',
    'src/attcon/data.py', 'src/attcon/models.py',
    'src/attcon/predictive_attention.py',
    'docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md',
    'docs/ATTENTION_CONTENT_MODEL_V1_PERMISSION_SCHEMA.md',
)
NATIVE_DIR = 'audits/spatial_native_task_v1/run'
NATIVE_REVIEW_DIR = 'audits/spatial_native_task_v1/numerical_corrected_review'
NATIVE_ACCEPTANCE_COMMIT = 'c84d3dabc0e3ca8b2d9898b8554b032556c0acff'


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
        require(relative(path) == path, 'noncanonical bound file path')
        require(digest(ROOT / path) == expected, f'current binding failed: {path}')
        process = subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{path}'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        require(hashlib.sha256(process.stdout).hexdigest() == expected,
                f'committed binding failed: {path}')


def ancestor(older, newer):
    require(all(len(value) == 40 and all(c in '0123456789abcdef' for c in value)
                for value in (older, newer)), 'ancestry needs full commit IDs')
    result = subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', older, newer],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode == 0, 'phase commit ancestry failed')


def native_acceptance():
    """JSON/hash-only prerequisite, before any torch import or model load."""
    manifest_path = ROOT / NATIVE_REVIEW_DIR / 'metadata_manifest.json'
    # A new self-consistent green JSON cannot replace the accepted first result.
    accepted_paths = (f'{NATIVE_REVIEW_DIR}/metadata_manifest.json',
                      f'{NATIVE_REVIEW_DIR}/metadata.json.gz')
    bound_files(NATIVE_ACCEPTANCE_COMMIT,
                {path: digest(ROOT / path) for path in accepted_paths})
    manifest = json.loads(manifest_path.read_text())
    packed = ROOT / manifest['published_path']
    require(digest(packed) == manifest['compressed_sha256'], 'native review transport changed')
    raw = gzip.decompress(packed.read_bytes())
    require(hashlib.sha256(raw).hexdigest() == manifest['raw_sha256'], 'native review raw identity failed')
    review = json.loads(raw)
    require(review['status'] == 'passed_corrected_reference_revalidation'
            and review['check_count'] == 206070 and review['error_count'] == 0
            and review['core_check_inventory_matches_first_review'], 'native prerequisite failed')
    require(review['reconstructed_engineering_feasibility_pass'] is True
            and len(review['checks']) == 206070 and all(row['passed'] for row in review['checks']),
            'native check inventory or reconstructed feasibility failed')
    actual_bindings = {row['name'].removeprefix('actual_results_bytes/'): row
                      for row in review['checks'] if row['name'].startswith('actual_results_bytes/')}
    required_actual = tuple(f'{NATIVE_DIR}/states-{seed}.pt.gz' for seed in SEEDS) + (
        f'{NATIVE_DIR}/results.json', f'{NATIVE_DIR}/completion.json')
    for path in required_actual:
        row = actual_bindings.get(path)
        require(row is not None and row['expected_sha256'] == row['actual_sha256'] == digest(ROOT / path),
                f'native passing evidence belongs to different bytes: {path}')
    provenance = review['corrected_reference_provenance']
    for row in provenance['bindings']:
        bound_files(row['commit'], {row['path']: row['expected_sha256']})
    require(digest(ROOT / 'scripts/verify_spatial_native_task_v1_corrected.py') == review['verifier_sha256'],
            'accepted native verifier changed')
    results = json.loads((ROOT / NATIVE_DIR / 'results.json').read_text())
    require(results['all_engineering_feasibility_gates_pass'], 'native feasibility gate failed')
    completion = json.loads((ROOT / NATIVE_DIR / 'completion.json').read_text())
    require(completion['status'] == 'completed', 'native run incomplete')
    require(completion['results_sha256'] == digest(ROOT / NATIVE_DIR / 'results.json'),
            'native completion result binding failed')
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
        'source_commit': args.source_commit, 'review_path': review_path,
        'source_hashes': hashes, 'input_hashes': inputs,
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


def validate_registration(registration):
    require(registration['schema'] == SCHEMA, 'wrong registration schema')
    review_path = registration['review_path']
    require(relative(review_path) == review_path and review_path not in REQUIRED_SOURCES,
            'invalid complete-review path')
    require(set(registration['source_hashes']) == set(REQUIRED_SOURCES) | {review_path},
            'registration omits or changes reviewed source closure')
    review = json.loads((ROOT / review_path).read_text())
    require(review['scope'] == REVIEW_SCOPE and review['approved'] is True
            and review['approved_source_sha256'] == {
                path: registration['source_hashes'][path] for path in REQUIRED_SOURCES},
            'registration does not bind a complete exact-source approval')
    expected = {'encoder_seeds': list(SEEDS), 'worlds_per_encoder_phase': 96,
        'parameter_count_per_learned_source': 197328, 'updates_per_source': 300,
        'phases': list(PHASES), 'phase_receipts_required_before_release': True,
        'reporter_calls': 0, 'no_fit_retry_or_adaptation': True,
        'development_only': True, 'source_semantics_qualified': False,
        'presentation_qualified': False,
        'source_inputs': ['V', 'issued commands', 'completed feedback through t-1'],
        'forbidden_source_inputs': ['pixels', 'mapping labels', 'bandwidth truth',
            'renderer truth', 'current concealed W_t', 'future fork outcomes', 'world identifiers']}
    require(all(registration[key] == value for key, value in expected.items()),
            'registration changes the fixed design or reporting boundary')
    require(relative(registration['output_dir']) == registration['output_dir'],
            'noncanonical phase output directory')
    bound_files(registration['source_commit'], registration['source_hashes'])


def receipt(base, phase, commit, registration_sha256):
    path = base / phase / 'receipt.json'
    value = json.loads(path.read_text())
    require(value['status'] == 'completed' and value['phase'] == phase, f'{phase} phase incomplete')
    require(value['registration_sha256'] == registration_sha256,
            f'{phase} receipt belongs to a different input registration')
    required = {relative(base / phase / f'states-{seed}.pt.gz') for seed in SEEDS}
    required.add(relative(base / phase / 'first_run.log'))
    require(set(value['artifact_sha256']) == required, 'incomplete phase artifact inventory')
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


def validate_fit_result(fitted, episode, seed):
    """Validate returned fit inventory, not an unobserved optimizer trajectory."""
    import torch
    from attention_content_model_v1 import FitNormalizer, build_learned_sources
    required = {'models', 'normalizer', 'initial_checkpoints', 'checkpoints',
        'optimizer_terminal_states', 'training_log', 'fit_visibility',
        'completed_updates_per_source', 'visibility_reconstruction'}
    require(required.issubset(fitted), 'incomplete fitted artifact schema')
    require(episode['split'] == 'fit' and episode['encoder_seed'] == seed, 'wrong fit provenance')
    require(fitted['completed_updates_per_source'] == 300, 'wrong fixed fit update count')
    normalizer = fitted['normalizer']
    require(normalizer == FitNormalizer.fit(episode['V'], split='fit'), 'normalizer is not fit-only V coordinates')
    initial_a, initial_ordinary, _alias = build_learned_sources(normalizer, seed)
    expected_initial = {'A': initial_a, 'ordinary': initial_ordinary}
    for key in ('models', 'initial_checkpoints', 'checkpoints', 'optimizer_terminal_states', 'fit_visibility'):
        require(set(fitted[key]) == {'A', 'ordinary'}, f'wrong fitted source inventory: {key}')
    for source, model in fitted['models'].items():
        require(model.source_key == source and model.normalizer == normalizer,
                'fitted model source/coordinates mismatch')
        require(sum(parameter.numel() for parameter in model.parameters()) == 197328,
                'wrong learned parameter count')
        actual = model.state_dict()
        initial = expected_initial[source].state_dict()
        require(set(actual) == set(fitted['checkpoints'][source]) == set(fitted['initial_checkpoints'][source]),
                'incomplete checkpoint parameter inventory')
        for key, value in actual.items():
            require(value.dtype == torch.float32 and value.device.type == 'cpu' and torch.isfinite(value).all(),
                    'nonfinite or wrong-dtype fitted weights')
            require(torch.equal(value, fitted['checkpoints'][source][key]), 'saved checkpoint differs from fitted model')
            require(torch.equal(initial[key], fitted['initial_checkpoints'][source][key]),
                    'initial checkpoint does not match fixed source seed')
        optimizer = fitted['optimizer_terminal_states'][source]
        require(len(optimizer['param_groups']) == 1, 'wrong Adam parameter-group inventory')
        group = optimizer['param_groups'][0]
        expected = {'lr': .003, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': 0,
                    'amsgrad': False, 'foreach': False}
        require(all(group[key] == value for key, value in expected.items()), 'wrong fixed optimizer settings')
        require(set(optimizer['state']) == set(group['params']) and len(group['params']) == len(actual),
                'incomplete optimizer terminal states')
        for state in optimizer['state'].values():
            require(float(state['step']) == 300, 'Adam terminal step is not 300')
            require(all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor)),
                    'nonfinite optimizer terminal state')
        records = [row for row in fitted['training_log'] if row['source'] == source]
        require(len(records) == 300 and [row['update_index'] for row in records] == list(range(300))
                and [row['completed_updates'] for row in records] == list(range(1, 301)), 'incomplete ordered training log')
        require(all(math.isfinite(row[key]) for row in records
                    for key in ('loss', 'current_loss', 'next_loss', 'preclip_gradient_norm')), 'nonfinite fit log')
        calls = fitted['fit_visibility'][source]
        require(len(calls) == 15, 'incomplete fit observation recipe')
        for time, call in enumerate(calls):
            require(call['kind'] == 'primary' and call['decision_time'] == time
                    and call['last_revealed_at'] == call['source_revealed_through'] == max(0, time - 1)
                    and call['update_command_indices'] == list(range(max(0, time - 1)))
                    and call['current_completion_visible'] == (time == 0)
                    and call['fork_outcomes_visible'] is False, 'invalid fit observation boundary')
    require(len(fitted['training_log']) == 600, 'extra/missing source update records')
    recipe = fitted['visibility_reconstruction']
    require(recipe['updates_per_source'] == 300 and recipe['reset_state_each_update'] is True
            and recipe['decision_indices'] == list(range(15)) and recipe['revealed_at'] == list(range(1, 15)),
            'invalid fit visibility repetition recipe')


def validate_evaluation_inventory(evaluation, seed, phase):
    """Freeze named gates and opportunities before accepting an engine result."""
    import torch
    from collections import Counter
    metadata = evaluation['metadata']
    require(metadata['encoder_seed'] == seed and metadata['split'] == phase
            and metadata['worlds'] == 96 and metadata['source_indices'] == ['A', 'ordinary', 'analytic'],
            'evaluation source/split/world inventory mismatch')
    shared = {'sham_selected_exact_v': (96, 3), 'sham_nonselected_unchanged': (96, 3),
        'same_w_driver_exact': (96, 3), 'same_w_driver_fields_exact': (96, 3),
        'same_w_driver_action_exact': (96, 3), 'actual_restoration_exact': (96, 3),
        'actual_restoration_fields_exact': (96, 3), 'actual_restoration_action_exact': (96, 3),
        'actual_address_fields_exact': (96, 3), 'actual_address_action_exact': (96, 3),
        'actual_family_reconstruction_exact': (96, 60),
        'actual_compaction_reconstruction_exact': (96, 61)}
    baseline = {name: (96, 3) for name in ('selected_policy_fork_exact', 'selected_policy_native_action_exact',
        'confidence_choice_exact', 'address_current_equivariance', 'address_next_equivariance')}
    baseline.update(initial_current_exact_v=(96, 1), current_nonselected_copy_exact=(96, 14),
        next_noncandidate_inheritance_exact=(96, 15), observation_history_exact=(96, 15),
        last_revealed_snapshot_exact=(96, 15), prediction_inputs_unchanged_exact=(96, 15),
        prediction_state_unchanged_exact=(96, 15))
    for schedule in ('current', 'prospective'):
        for value in ('current', 'next', 'policy', 'policy_fields', 'state'):
            baseline[f'restore_{schedule}_{value}_exact'] = (96, 3)
        for value in ('state', 'current', 'next'):
            baseline[f'patch_{schedule}_nonselected_{value}_exact'] = (96, 3)
        for value in ('selected_state_transfer', 'nonselected_decoder', 'next_inheritance'):
            baseline[f'patch_{schedule}_{value}_exact'] = (96, 3)
    checks = evaluation['software_checks']
    require(set(checks) == {'shared', 'sources'} and set(checks['sources']) == {'A', 'ordinary', 'analytic'},
            'wrong hard-check source inventory')
    def shapes(values, expected):
        require(set(values) == set(expected), 'missing or undeclared hard invariant names')
        for name, shape in expected.items():
            value = values[name]
            require(isinstance(value, torch.Tensor) and value.shape == shape
                    and value.dtype == torch.bool and value.device.type == 'cpu', 'wrong hard-invariant opportunity shape')
    shapes(checks['shared'], shared)
    for source in ('A', 'ordinary', 'analytic'):
        expected = dict(baseline)
        if source != 'analytic':
            expected['fitted_weights_unchanged'] = (1,)
        if source == 'A':
            expected.update(alias_current_exact=(96, 15), alias_next_exact=(96, 15))
        shapes(checks['sources'][source], expected)
        calls = metadata['source_call_visibility'][source]
        expected_calls = {'primary': 15, 'address_remapping': 3, 'donor_current': 3,
                          'donor_prospective': 3, 'restored_current': 3, 'restored_prospective': 3}
        if source == 'A':
            expected_calls['same_weight_alias_duplicate'] = 15
        require(Counter(call['kind'] for call in calls) == expected_calls, 'incomplete source-call opportunity inventory')
        for call in calls:
            time = call['decision_time']
            require(0 <= time <= 14 and call['last_revealed_at'] == call['source_revealed_through'] == max(0, time - 1)
                    and call['update_command_indices'] == list(range(max(0, time - 1)))
                    and call['current_completion_visible'] == (time == 0)
                    and call['fork_outcomes_visible'] is False, 'invalid evaluation observation boundary')
            require(call['completed'] is True and call['source_key'] == source
                    and call['state_hashes'] == call['post_call_state_hashes']
                    and all(call[key] == value for key, value in call['post_call_input_hashes'].items()),
                    'source call did not complete or mutated inputs/state')
        updates = metadata['source_update_visibility'][source]
        require(Counter(call['kind'] for call in updates) == {'primary_feedback': 14, 'donor_replay': 72},
                'incomplete actual update-call inventory')
        for index, call in enumerate(updates):
            time = call['completed_at']
            require(call['call_index'] == index and call['completed'] is True and call['source_key'] == source
                    and 1 <= time <= 14 and call['command_index'] == time - 1
                    and call['scored_decision_at'] == time and call['pre_revealed_through'] == time - 1
                    and call['post_revealed_through'] == time
                    and call['pre_update_command_indices'] == list(range(time - 1))
                    and call['post_update_command_indices'] == list(range(time)),
                    'invalid actual feedback-update boundary')
        require(len(metadata['source_decoder_visibility'][source]) == 27,
                'incomplete separately counted decoder-probe inventory')
        require(metadata['source_call_inventory'][source] == {'prediction_calls': len(calls),
            'primary_update_calls': 14, 'donor_update_calls': 72, 'address_local_decoder_probe_calls': 27},
            'declared source-call counts disagree with actual records')
    require(metadata['denominators'] == {'worlds_shared': 96, 'common_transitions_shared': 1440,
        'actual_forks_shared': 4320, 'current_selected_per_source': 1344, 'next_selected_per_source': 4320,
        'full_next_item_records_per_source': 17280, 'query_actual_actions_shared': 864,
        'executed_policy_duplicates_per_source': 288, 'donor_pairs_per_source_per_schedule': 288,
        'donor_transitions_shared': 7488, 'donor_feedback_events_per_source': 6912,
        'donor_policy_fork_references_per_source': 576}, 'changed source/shared denominators')
    require(metadata['software_invariants_pass'] is True, 'engine reports failed hard invariants')
    enforce_software_checks(checks)


def validate_compaction(compact, evaluation):
    import torch
    checks = compact['actual_reconstruction_exact']
    require(compact['actual_compaction_status'] == 'exact_reconstruction_bound'
            and checks.shape == (96, 61) and checks.dtype == torch.bool
            and checks.device.type == 'cpu' and bool(checks.all())
            and torch.equal(checks, evaluation['software_checks']['shared']['actual_compaction_reconstruction_exact']),
            'compacted actual state does not match exact engine reconstruction')


def execute_phase(args):
    require(args.registration_commit, 'phase requires preserved input registration')
    registration_path = relative(args.registration)
    registration = json.loads((ROOT / registration_path).read_text())
    registration_sha256 = digest(ROOT / registration_path)
    validate_registration(registration)
    ancestor(registration['source_commit'], args.registration_commit)
    bound_files(args.registration_commit, {registration_path: registration_sha256,
        **registration['source_hashes'], **registration['input_hashes']})
    require(native_acceptance() == registration['input_hashes'], 'native inputs changed after registration')
    base = ROOT / registration['output_dir']
    fit_receipt = None
    if args.phase in ('dev', 'heldout'):
        require(args.fit_commit, 'dev/heldout requires preserved immutable fit receipt')
        ancestor(args.registration_commit, args.fit_commit)
        fit_receipt = receipt(base, 'fit', args.fit_commit, digest(ROOT / registration_path))
    if args.phase == 'heldout':
        require(args.dev_commit and args.assessment, 'heldout requires preserved dev receipt and assessment')
        ancestor(args.fit_commit, args.dev_commit)
        dev_receipt = receipt(base, 'dev', args.dev_commit, digest(ROOT / registration_path))
        require(dev_receipt['fit_receipt_sha256'] == digest(base / 'fit' / 'receipt.json'),
                'dev used a different immutable fit receipt')
        assessment_path = relative(args.assessment)
        bound_files(args.dev_commit, {assessment_path: digest(ROOT / assessment_path)})
        assessment = json.loads((ROOT / assessment_path).read_text())
        require(assessment['schema'] == 'attention_content_model_v1_dev_acceptance'
            and assessment['hard_invariants_pass'] is True
            and assessment['dev_receipt_sha256'] == digest(base / 'dev' / 'receipt.json')
            and assessment['fitted_artifact_sha256'] == fit_receipt['artifact_sha256'], 'dev acceptance mismatch')
    phase_dir = base / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    # The exclusive claim precedes torch/model imports or any released generation.
    write_exclusive(phase_dir / 'claim.json', {'phase': args.phase,
        'registration_sha256': digest(ROOT / registration_path), 'registration_commit': args.registration_commit,
        'fit_commit': args.fit_commit, 'dev_commit': args.dev_commit,
        'fit_receipt_sha256': digest(base / 'fit' / 'receipt.json') if fit_receipt else None,
        'dev_receipt_sha256': digest(base / 'dev' / 'receipt.json') if args.phase == 'heldout' else None,
        'dev_assessment_sha256': digest(args.assessment) if args.phase == 'heldout' else None,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'retry_allowed': False})
    partial = {'phase': args.phase, 'seed': None}
    active_seed = None
    try:
        import torch
        from attention_content_model_v1 import FitNormalizer, build_learned_sources
        from attention_content_model_v1_data import collect_worlds
        from attention_content_model_v1_experiment import fit_sources, evaluate_sources, compact_episode
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        outputs = {}
        with (phase_dir / 'first_run.log').open('x') as log:
            def emit(value):
                print(json.dumps(value, allow_nan=False), file=log, flush=True)
                print(json.dumps(value, allow_nan=False), flush=True)
            for seed in SEEDS:
                active_seed = seed
                partial = {'phase': args.phase, 'seed': seed}
                extractor, head, coordinates = load_native(seed)
                episode = collect_worlds(seed, args.phase, extractor, released_split=args.phase)
                partial['episode'] = episode
                if args.phase == 'fit':
                    fitted = fit_sources(seed, episode, log=emit, partial=partial)
                    partial['returned_fit'] = fitted
                    validate_fit_result(fitted, episode, seed)
                    with torch.no_grad():
                        evaluation = evaluate_sources(fitted['models'], episode, head, coordinates,
                                                      extractor, 'fit', log=emit, partial=partial)
                    fitted = {key: value for key, value in fitted.items() if key != 'models'}
                    fitted['normalizer'] = asdict(fitted['normalizer'])
                    bundle = {'episode': compact_episode(episode), 'fit': fitted, 'evaluation': evaluation}
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
                        evaluation = evaluate_sources(models, episode, head, coordinates, extractor, args.phase,
                                                      log=emit, partial=partial)
                    bundle = {'episode': compact_episode(episode), 'evaluation': evaluation,
                        'fitted_artifact_sha256': fit_receipt['artifact_sha256']}
                partial = bundle
                output = phase_dir / f'states-{seed}.pt.gz'
                save_tensor_archive(output, bundle)
                validate_evaluation_inventory(evaluation, seed, args.phase)
                validate_compaction(bundle['episode'], evaluation)
                outputs[relative(output)] = digest(output)
                emit({'seed': seed, 'phase': args.phase, 'archive_sha256': digest(output)})
        outputs[relative(phase_dir / 'first_run.log')] = digest(phase_dir / 'first_run.log')
        write_exclusive(phase_dir / 'receipt.json', {'status': 'completed', 'phase': args.phase,
            'registration_sha256': digest(ROOT / registration_path), 'artifact_sha256': outputs,
            'fit_receipt_sha256': digest(base / 'fit' / 'receipt.json') if fit_receipt else None,
            'registration_commit': args.registration_commit, 'fit_commit': args.fit_commit,
            'dev_commit': args.dev_commit,
            'dev_assessment_sha256': digest(args.assessment) if args.phase == 'heldout' else None,
            'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'retry_performed': False,
            'no_reporter_calls': True, 'semantic_qualification_asserted': False})
    except BaseException:
        original_traceback = traceback.format_exc()
        write_exclusive(phase_dir / 'failure.json', {'status': 'failed_first_phase_execution',
            'phase': args.phase, 'seed': active_seed, 'traceback': original_traceback,
            'registration_sha256': registration_sha256, 'retry_allowed': False})
        try:
            save_tensor_archive(phase_dir / 'first_failure.pt.gz', partial)
        except BaseException:
            write_exclusive(phase_dir / 'retention_failure.json', {'status': 'secondary_retention_failure',
                'original_failure_preserved': True, 'traceback': traceback.format_exc(), 'retry_allowed': False})
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
