"""Single-attempt registered spatial task DEVELOPMENT runner.

--help and compilation do not import torch, load checkpoints, or generate worlds.
Registration hashes bytes only. Execution is separately authorized by a committed
registration and an independently approved, exact source review receipt.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1301, 1311, 1321)
FIT_WORLDS, DIAGNOSTIC_WORLDS, ITEMS, STEPS = 128, 64, 4, 500
BANDS = (1, 2, 4)
ATOL, RTOL = 1e-6, 1e-5
SCHEMA = 'spatial_native_task_v1_development_registration'
REVIEW_SCOPE = 'spatial_native_task_v1_preexecution_source_review'
REQUIRED_SOURCES = (
    'scripts/run_spatial_native_task_v1.py', 'scripts/spatial_native_task_v1.py',
    'scripts/attention_content_form_primitives_v1.py',
    'scripts/grounded_outcome_data_v1.py', 'scripts/grounded_attention_world_v1.py',
    'scripts/orientation_transfer_diagnostic_v1.py',
    'scripts/isolated_orientation_task_v1.py',
    'src/attcon/__init__.py', 'src/attcon/data.py', 'src/attcon/models.py',
    'src/attcon/bound_content.py', 'src/attcon/predictive_attention.py',
    'docs/SPATIAL_NATIVE_TASK_V1_PROTOCOL.md',
    'audits/spatial_native_task_v1/runner_software/design.md',
    'audits/spatial_native_task_v1/runner_software/first_run.log',
    'audits/spatial_native_task_v1/runner_software/validation.json',
)
CHECKPOINTS = tuple(path for seed in SEEDS for path in (
    f'audits/bound_content/confirmation_v2/seed{seed}.pt',
    f'audits/sensory_task_adapter_v1/states-{seed}.pt.gz'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def relative(path):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    require(resolved.is_relative_to(ROOT), 'bound file must reside under repository root')
    return resolved.relative_to(ROOT).as_posix()


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def git(checkout, *args):
    result = subprocess.run(['git', '-C', str(checkout), *args], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def preserved(checkout, commit, hashes):
    require(re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit or '') is not None,
            'preservation commit must be a full lowercase Git object ID')
    require(git(checkout, 'rev-parse', '--verify', commit + '^{commit}').decode().strip() == commit,
            'preservation commit is not a full commit object')
    for path, expected in hashes.items():
        actual = git(checkout, 'show', f'{commit}:{path}')
        require(sha_bytes(actual) == expected, f'committed preservation mismatch: {path}')


def check_hashes(hashes):
    for path, expected in hashes.items():
        require(relative(path) == path, f'noncanonical bound path: {path}')
        require(digest(ROOT / path) == expected, f'current file hash mismatch: {path}')


def review_receipt(path, source_hashes):
    value = json.loads((ROOT / path).read_text())
    require(value.get('source_review_status') == 'approved_for_development_only',
            'review must approve development only')
    require(value.get('status') == 'approved' and value.get('receipt_scope') == REVIEW_SCOPE,
            'review status or receipt scope does not approve this source execution')
    require(value.get('independent') is True and isinstance(value.get('reviewer'), str)
            and value['reviewer'].strip(), 'independent reviewer identity is required')
    require(value.get('approved_sha256') == source_hashes,
            'review approved_sha256 must exactly equal the registered source hash mapping')
    return value


def envelope():
    return {
        'seeds': list(SEEDS), 'fit_worlds_per_seed': FIT_WORLDS,
        'diagnostic_worlds_per_seed': DIAGNOSTIC_WORLDS, 'items_per_world': ITEMS,
        'operator_bandwidths': list(BANDS), 'world_namespace': '210000000 + seed*1000 + split*100000',
        'native_head_init_namespace': '220000000 + seed',
        'raw_head_init_namespace': '221000000 + seed',
        'head_parameters': {'native24': 899, 'raw3': 227},
        'full_batch_adam_steps': STEPS, 'adam': {'lr': .003, 'betas': [.9, .999],
            'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False, 'foreach': False},
        'dtype': 'float32', 'device': 'cpu', 'threads': 4,
        'standardizer': 'fit_only_channel_mean_population_std; scale<1e-6 replaced with one',
        'bootstrap_draws': 256, 'bootstrap_seed': '230000000 + seed',
        'engineering_gates_per_seed': {'T1_fine': .50, 'T1_coarse': .75, 'raw_fine': .95},
        'semantic_qualification_gate': None, 'single_attempt': True,
        'fine_tie_rule': 'exact maximum, first row-major cell',
        'coarse_unknown_absolute_tolerance': 1e-6,
        'role_field_meaning': 'forced conditional location field, one known task role of each kind; not presence/confidence',
        'field_change_rule': 'maximum absolute role-field difference > 1e-6; exact-change also retained',
        'address_permutation_old_to_new': [1, 2, 3, 0],
        'synthetic_site_permutation_new_to_old': [15] + list(range(15)),
        'permutation_numerics': {'atol': ATOL, 'rtol': RTOL},
        'expected_counts': {'primary_item_operator_opportunities': 2304,
            'primary_role_queries': 4608, 'sham_T1_role_queries': 1536,
            'matched_exogenous_T2_role_queries': 1536, 'restore_after_T2_role_queries': 1536,
            'address_permutation_role_queries': 4608, 'synthetic_site_permutation_role_queries': 4608},
        'interpretation': 'development task/content semantics only; no attention model A, Reporter, presentation qualification, or qualia',
    }


def register(args):
    require(args.review and args.source_commit, '--register requires --review and --source-commit')
    require(args.preservation_checkout, '--register requires --preservation-checkout')
    require(args.execution and args.output_dir, '--register requires fixed --execution and --output-dir declarations')
    registration_path, execution_path, output_path = (relative(p) for p in
        (args.registration, args.execution, args.output_dir))
    require(registration_path != execution_path, 'registration and execution paths must differ')
    require(not (ROOT / execution_path).exists() and not (ROOT / output_path).exists(),
            'declared execution/output paths already exist')
    require(not (ROOT / execution_path).is_relative_to(ROOT / output_path)
            and not (ROOT / registration_path).is_relative_to(ROOT / output_path),
            'registration and execution claims must be outside the new output directory')
    sources = tuple(sorted(set(REQUIRED_SOURCES) | {relative(p) for p in args.source}))
    source_hashes = {p: digest(ROOT / p) for p in sources}
    review_path = relative(args.review)
    require(review_path not in source_hashes, 'review receipt cannot approve its own hash')
    review_receipt(review_path, source_hashes)
    review_hashes = {review_path: digest(ROOT / review_path)}
    preserved(args.preservation_checkout, args.source_commit, {**source_hashes, **review_hashes})
    checkpoint_hashes = {p: digest(ROOT / p) for p in CHECKPOINTS}
    value = {'schema': SCHEMA, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'envelope': envelope(), 'source_sha256': source_hashes,
        'review_sha256': review_hashes, 'checkpoint_sha256': checkpoint_hashes,
        'source_preservation_commit': args.source_commit,
        'execution_path': execution_path, 'output_dir': output_path,
        'source_review_status': 'approved_for_development_only',
        'goal_achieved': False, 'development_execution_started': False}
    exclusive_json(ROOT / relative(args.registration), value)
    print('Registered exact independently approved DEVELOPMENT bytes; no checkpoint load, world generation or fit.')


def validate_registration(args):
    registration_path = relative(args.registration)
    registration = json.loads((ROOT / registration_path).read_text())
    require(registration.get('schema') == SCHEMA and registration.get('envelope') == envelope(),
            'registration schema or fixed envelope mismatch')
    require(registration.get('source_review_status') == 'approved_for_development_only'
            and registration.get('development_execution_started') is False,
            'registration is not a preexecution development registration')
    require(relative(args.execution) == registration.get('execution_path')
            and relative(args.output_dir) == registration.get('output_dir'),
            'execution/output paths must exactly match committed registration; no alternate retry paths')
    source_hashes = registration['source_sha256']
    require(set(REQUIRED_SOURCES).issubset(source_hashes), 'missing required source bindings')
    checkpoint_hashes = registration['checkpoint_sha256']
    require(set(checkpoint_hashes) == set(CHECKPOINTS), 'checkpoint inventory mismatch')
    review_hashes = registration['review_sha256']
    require(len(review_hashes) == 1, 'exactly one independent review receipt is required')
    check_hashes({**source_hashes, **review_hashes, **checkpoint_hashes})
    review_receipt(next(iter(review_hashes)), source_hashes)
    require(args.preservation_checkout and args.registration_commit,
            '--run requires --preservation-checkout and --registration-commit')
    preserved(args.preservation_checkout, registration['source_preservation_commit'],
              {**source_hashes, **review_hashes})
    preserved(args.preservation_checkout, args.registration_commit,
              {**source_hashes, **review_hashes, registration_path: digest(ROOT / registration_path)})
    require(subprocess.run(['git', '-C', str(args.preservation_checkout), 'merge-base',
            '--is-ancestor', registration['source_preservation_commit'], args.registration_commit],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0,
            'registration commit must descend from preserved reviewed source commit')
    return registration


def finite_tree(value, path='archive'):
    if isinstance(value, torch.Tensor):
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), f'nonfinite {path}')
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f'{path}/{key}')
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            finite_tree(item, f'{path}/{i}')
    elif isinstance(value, float):
        require(__import__('math').isfinite(value), f'nonfinite {path}')


def world_set(seed, split, count):
    namespace = 210000000 + seed * 1000 + split * 100000
    worlds = [task.render_world(namespace, index) for index in range(count)]
    return {'pixels': torch.stack([w.pixels for w in worlds]),
        'evaluator_cells': torch.tensor([w.truth.cells for w in worlds], dtype=torch.long),
        'occupancy_targets': torch.stack([w.truth.occupancy() for w in worlds]),
        'render_nuisance': [{'token_shapes': w.token_shapes, 'intensities': w.intensities,
                            'noise_sigma': w.noise_sigma} for w in worlds],
        'world_namespace': namespace, 'world_indices': torch.arange(count)}


def fit_head(seed, name, fit_maps, targets, log):
    standardizer = task.ChannelStandardizer.fit(fit_maps)
    init_seed = 220000000 + seed + (1000000 if name == 'raw3' else 0)
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        head = task.LocalOccupancyHead(fit_maps.shape[-3]).float().cpu()
    require(sum(p.numel() for p in head.parameters()) == (227 if name == 'raw3' else 899),
            'head parameter count mismatch')
    initial_weights = {k: v.detach().clone() for k, v in head.state_dict().items()}
    inputs = standardizer.apply(fit_maps)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003, betas=(.9, .999),
        eps=1e-8, weight_decay=0, amsgrad=False, foreach=False)
    losses = []
    for update in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = task.occupancy_loss(head(inputs), targets)
        require(bool(torch.isfinite(loss)), 'nonfinite fit loss')
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
        log({'kind': 'fit_loss', 'seed': seed, 'head': name,
             'update': update + 1, 'preupdate_loss': losses[-1]})
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        final_logits = head(inputs)
        final_loss = float(task.occupancy_loss(final_logits, targets))
    log({'kind': 'fit_finished', 'seed': seed, 'head': name,
         'steps': STEPS, 'final_loss': final_loss})
    return head, standardizer, {'initialization_seed': init_seed,
        'fit_only_standardizer': asdict(standardizer), 'initial_weights': initial_weights,
        'weights': {k: v.detach().clone() for k, v in head.state_dict().items()},
        'losses_preupdate': torch.tensor(losses), 'final_fit_loss': final_loss,
        'final_fit_logits': final_logits.detach().clone()}


def initial_state(maps, orientation_logits):
    return task.AddressedState(tuple(task.AddressedItem.from_tensors(address, maps[item],
        maps[item], True, 'role_pair', float(orientation_logits[item]), 0, 0)
        for item, address in enumerate(task.ADDRESSES)), 0)


def tensor_stats(maps):
    rows = maps.double().flatten(-2)
    return {'channel_mean': rows.mean(-1), 'channel_norm': rows.square().sum(-1).sqrt(),
            'channel_population_std': rows.std(-1, correction=0)}


def source_controls(extractor, maps):
    controls = extractor.controls(maps)
    controls['orientation_standardized_hidden'] = (
        controls['orientation_hidden'] - extractor.mean64) / extractor.scale64
    coordinate = controls['orientation_logits'].sigmoid()
    controls['orientation_uncalibrated_sigmoid_coordinates'] = torch.stack((1 - coordinate, coordinate), -1)
    return controls


def field_stats(fields):
    flat = fields.double().flatten(-2)
    ordered = flat.sort(-1, descending=True).values
    return {'entropy': -(flat * flat.clamp_min(1e-300).log()).sum(-1),
            'maximum': ordered[..., 0], 'margin': ordered[..., 0] - ordered[..., 1],
            'expected_column': (fields.double() * torch.arange(4, dtype=torch.float64)).sum((-2, -1))}


def evidence_record(controller, snapshot, address):
    evidence = snapshot.item(address)
    actions = [controller.target(snapshot, address, role) for role in task.ROLES]
    coarse = controller.coarse_order(snapshot, address)
    return {'logits': evidence.local_logits.tensor(), 'role_fields': evidence.role_fields.tensor(),
        'actions': [asdict(a) for a in actions], 'coarse': asdict(coarse),
        'snapshot_time': snapshot.time, 'queried_address': address}


def same_evidence(left, right, exact=False):
    for key in ('logits', 'role_fields'):
        comparison = torch.equal(left[key], right[key]) if exact else torch.allclose(
            left[key], right[key], atol=ATOL, rtol=RTOL)
        require(comparison, f'native evidence parity failed: {key}')
    require(left['actions'] == right['actions'] and left['coarse'] == right['coarse'],
            'native action/coarse parity failed')


def stack_records(records, targets, cells):
    # Ordered [world,item], all actions originate in actual NativeSnapshots.
    logits = torch.stack([record['logits'] for record in records]).reshape(
        DIAGNOSTIC_WORLDS, ITEMS, 3, 4, 4)
    fields = torch.stack([record['role_fields'] for record in records]).reshape(
        DIAGNOSTIC_WORLDS, ITEMS, 2, 4, 4)
    selected = torch.tensor([[[a['row'], a['column']] for a in record['actions']]
                             for record in records]).reshape(DIAGNOSTIC_WORLDS, ITEMS, 2, 2)
    ties = torch.tensor([[len(a['tied_cells']) for a in record['actions']]
                        for record in records]).reshape(DIAGNOSTIC_WORLDS, ITEMS, 2)
    orders = [record['coarse']['order'] for record in records]
    order_codes = torch.tensor([{'red_left_of_blue': -1, 'unknown': 0,
                                'blue_left_of_red': 1}[order] for order in orders]).reshape(
                                    DIAGNOSTIC_WORLDS, ITEMS)
    true_codes = torch.where(cells[..., 0, 1] < cells[..., 1, 1], -1, 1)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1),
        reduction='none').mean((-2, -1)).reshape(DIAGNOSTIC_WORLDS, ITEMS)
    return {'local_logits': logits, 'role_fields': fields, 'selected_cells': selected,
        'tie_counts': ties, 'coarse_codes': order_codes, 'evaluator_coarse_codes': true_codes,
        'occupancy_loss': loss, 'fine_correct': (selected == cells).all(-1),
        'coarse_correct': order_codes == true_codes, 'field_stats': field_stats(fields),
        'native_records': records}


def evaluate_native(seed, diagnostic, maps, extractor, controller, partial):
    original_controls = source_controls(extractor, maps)
    actual_maps = {band: torch.zeros_like(maps) for band in BANDS}
    primary = {band: [] for band in BANDS}
    control_records = {key: [] for key in ('sham_T1', 'matched_exogenous_T2', 'restore_after_T2')}
    permutation_records = {key: {band: [] for band in BANDS}
                           for key in ('address_permutation', 'synthetic_site_permutation')}
    operator_metadata = {band: [] for band in BANDS}
    synthetic_maps = {band: torch.zeros_like(maps) for band in BANDS}
    isolation_checks = []
    site_permutation = torch.tensor(envelope()['synthetic_site_permutation_new_to_old'])
    address_permutation = envelope()['address_permutation_old_to_new']
    partial.update(primary_native_records=primary, duplicate_native_records=control_records,
        permutation_native_records=permutation_records, actual_operator_maps=actual_maps,
        synthetic_maps=synthetic_maps, operator_metadata=operator_metadata,
        input_query_isolation_checks=isolation_checks, original_native_controls=original_controls)
    # Batch outputs are an independent computation check, never the action source.
    with torch.no_grad():
        for band in BANDS:
            for world in range(DIAGNOSTIC_WORLDS):
                for item in range(ITEMS):
                    actual_maps[band][world, item] = task.block_average(maps[world, item], band)
        batch_logits = {b: controller.head(controller.standardizer.apply(w))
                        for b, w in actual_maps.items()}
        batch_fields = {b: task.role_fields(logits) for b, logits in batch_logits.items()}
    for world in range(DIAGNOSTIC_WORLDS):
        saved = initial_state(maps[world], original_controls['orientation_logits'][world])
        for item, address in enumerate(task.ADDRESSES):
            baseline = None
            for band in BANDS:
                pending = task.PendingOperator(address, band, 0, 1)
                completed = task.execute_endogenous(saved, pending)
                snapshot = controller.consume(completed)
                record = evidence_record(controller, snapshot, address)
                record.update(seed=seed, world_index=world, item_index=item, bandwidth=band,
                              opportunity='primary_endogenous')
                require(all(x == y for x, y in zip(saved.items, completed.items) if x.address != address),
                        'operator changed a nonselected item')
                require(torch.equal(completed.item(address).actual_w.tensor(), actual_maps[band][world, item]),
                        'completed actual W differs from finite primitive')
                require(torch.allclose(record['logits'], batch_logits[band][world, item], atol=ATOL, rtol=RTOL)
                    and torch.allclose(record['role_fields'], batch_fields[band][world, item], atol=ATOL, rtol=RTOL),
                    'actual controller consumption differs from batched computation')
                primary[band].append(record)
                operator_metadata[band].append({'world_index': world, 'item_index': item,
                    'pending': asdict(pending), 'completed_history': [asdict(e) for e in completed.history],
                    'actual_completed_at': completed.item(address).w_completed_at,
                    'other_items_unchanged': all(x == y for x, y in zip(saved.items, completed.items)
                                                 if x.address != address)})
                if band == 1:
                    baseline = record
                    require(completed.item(address).actual_w == saved.item(address).actual_w,
                            'T1 is not an exact saved-state copy')
                    sham_record = evidence_record(controller, controller.consume(saved), address)
                    same_evidence(record, sham_record, exact=True)
                    sham_record.update(seed=seed, world_index=world, item_index=item, bandwidth=1,
                                       opportunity='sham_saved_initial')
                    control_records['sham_T1'].append(sham_record)
                elif band == 2:
                    exogenous = task.execute_exogenous(saved, pending)
                    require(all(x.actual_w == y.actual_w for x, y in zip(completed.items, exogenous.items)),
                            'matched driver W differs')
                    external = evidence_record(controller, controller.consume(exogenous), address)
                    same_evidence(record, external, exact=True)
                    external['completed_history'] = [asdict(e) for e in exogenous.history]
                    external.update(seed=seed, world_index=world, item_index=item, bandwidth=2,
                                    opportunity='matched_exogenous_T2')
                    control_records['matched_exogenous_T2'].append(external)
                    # Completed T2 has occurred; the exact saved snapshot is the restoration source.
                    restored = task.restore_saved(saved)
                    require(restored == saved and restored != completed, 'saved-state restoration failed')
                    restoration = evidence_record(controller, controller.consume(restored), address)
                    same_evidence(baseline, restoration, exact=True)
                    restoration['restored_after'] = [asdict(e) for e in completed.history]
                    restoration['restored_history'] = []
                    restoration.update(seed=seed, world_index=world, item_index=item, bandwidth=2,
                                       opportunity='restore_saved_after_T2')
                    control_records['restore_after_T2'].append(restoration)
                else:
                    require(bool((record['role_fields'] == 1 / 16).all()), 'T4 field not exactly uniform')
                    require(all(len(a['tied_cells']) == 16 for a in record['actions'])
                            and record['coarse']['order'] == 'unknown', 'T4 ties/coarse invariant failed')
                renamed = task.AddressedState(tuple(sorted((replace(x,
                    address=task.ADDRESSES[address_permutation[j]]) for j, x in enumerate(completed.items)),
                    key=lambda x: x.address)), completed.time, tuple(replace(e,
                    address=task.ADDRESSES[address_permutation[task.ADDRESSES.index(e.address)]])
                    for e in completed.history))
                renamed_address = task.ADDRESSES[address_permutation[item]]
                permuted = evidence_record(controller, controller.consume(renamed), renamed_address)
                require(torch.allclose(permuted['logits'], record['logits'], atol=ATOL, rtol=RTOL)
                        and torch.allclose(permuted['role_fields'], record['role_fields'], atol=ATOL, rtol=RTOL),
                        'outer-address permutation evidence failed')
                require(all((a['row'], a['column'], a['tied_cells']) == (b['row'], b['column'], b['tied_cells'])
                            for a, b in zip(permuted['actions'], record['actions']))
                        and permuted['coarse']['order'] == record['coarse']['order'],
                        'outer-address permutation query consistency failed')
                permuted.update(world_index=world, item_index=item, bandwidth=band,
                    original_address=address, remapped_completed_history=[asdict(e) for e in renamed.history])
                permutation_records['address_permutation'][band].append(permuted)
                synth_items = tuple(replace(x, actual_w=task.SpatialBuffer.from_tensor(
                    x.actual_w.tensor().flatten(-2)[..., site_permutation].reshape(24, 4, 4)))
                    for x in completed.items)
                synthetic_state = task.AddressedState(synth_items, completed.time, completed.history)
                synthetic_maps[band][world, item] = synthetic_state.item(address).actual_w.tensor()
                synthetic = evidence_record(controller, controller.consume(synthetic_state), address)
                expected = record['role_fields'].flatten(-2)[..., site_permutation].reshape(2, 4, 4)
                expected_logits = record['logits'].flatten(-2)[..., site_permutation].reshape(3, 4, 4)
                require(torch.allclose(synthetic['role_fields'], expected, atol=ATOL, rtol=RTOL)
                        and torch.allclose(synthetic['logits'], expected_logits, atol=ATOL, rtol=RTOL),
                        'synthetic site-permutation logits/field equivariance failed')
                for stat in ('entropy', 'maximum', 'margin'):
                    require(torch.allclose(field_stats(synthetic['role_fields'])[stat],
                        field_stats(record['role_fields'])[stat], atol=ATOL, rtol=RTOL),
                        f'synthetic site-permutation {stat} invariance failed')
                unique_checks = []
                for role_index, action in enumerate(record['actions']):
                    unique = len(action['tied_cells']) == 1
                    if unique:
                        old_site = action['row'] * 4 + action['column']
                        expected_site = int((site_permutation == old_site).nonzero()[0, 0])
                        selected = synthetic['actions'][role_index]
                        require(selected['row'] * 4 + selected['column'] == expected_site,
                                'unique-maximum site-permutation action equivariance failed')
                    unique_checks.append(unique)
                synthetic.update(world_index=world, item_index=item, bandwidth=band,
                    unique_maximum_action_checks=unique_checks,
                    actual_W_changed=not torch.equal(synthetic_maps[band][world, item],
                                                    actual_maps[band][world, item]))
                permutation_records['synthetic_site_permutation'][band].append(synthetic)
                # Counterfactual evaluator truth is kept outside Controller input.
                evaluator_truth = diagnostic['evaluator_cells'][world, item].clone()
                counterfactual_truth = evaluator_truth.flip(0)
                isolated_items = tuple(replace(x, identity='counterfactual_retained_record',
                    raw_logit=x.raw_logit + 1000., source_v=task.SpatialBuffer.from_tensor(
                        torch.zeros_like(x.source_v.tensor())), actual_w=(x.actual_w if x.address == address else
                    task.SpatialBuffer.from_tensor(torch.zeros_like(x.actual_w.tensor()))))
                    for x in completed.items)
                diagnostic['evaluator_cells'][world, item] = counterfactual_truth
                try:
                    isolated = controller.consume(task.AddressedState(isolated_items, completed.time, completed.history))
                finally:
                    diagnostic['evaluator_cells'][world, item] = evaluator_truth
                isolated_evidence = isolated.item(address)
                require(torch.equal(isolated_evidence.local_logits.tensor(), record['logits'])
                        and torch.equal(isolated_evidence.role_fields.tensor(), record['role_fields']),
                        'retained metadata/nonqueried input isolation failed')
                isolation_checks.append({'world_index': world, 'item_index': item, 'bandwidth': band,
                    'queried_address': address, 'identity_replacement': 'counterfactual_retained_record',
                    'raw_logit_delta': 1000., 'nonqueried_actual_W_replacement': 'all zeros',
                    'all_source_V_replacement': 'all zeros', 'counterfactual_evaluator_truth': counterfactual_truth,
                    'isolated_local_logits': isolated_evidence.local_logits.tensor(),
                    'isolated_role_fields': isolated_evidence.role_fields.tensor(),
                    'native_logits_fields_exactly_equal': True,
                    'evaluator_truth_present_in_native_input': False})
    targets, cells = diagnostic['occupancy_targets'], diagnostic['evaluator_cells']
    results = {f'T{band}': stack_records(records, targets, cells) for band, records in primary.items()}
    for band in BANDS:
        result = results[f'T{band}']
        result['actual_W'] = actual_maps[band]
        result['actual_W_stats'] = tensor_stats(actual_maps[band])
        result['actual_W_stat_changes_from_original'] = {k: v - tensor_stats(maps)[k]
                         for k, v in result['actual_W_stats'].items()}
        result['native_controls'] = source_controls(extractor, actual_maps[band])
        result['control_changes_from_original'] = {k: value - original_controls[k]
                            for k, value in result['native_controls'].items()}
        result['operator_metadata'] = operator_metadata[band]
        if band == 2:
            require(bool((result['tie_counts'] >= 4).all()), 'T2 must have at least four exact tied maximum sites')
    original_stats = tensor_stats(maps)
    controls = {key: stack_records(records, targets, cells) for key, records in control_records.items()}
    for key, by_band in permutation_records.items():
        controls[key] = {f'T{band}': stack_records(records, targets, cells)
                         for band, records in by_band.items()}
    for band in BANDS:
        synth_stats = tensor_stats(synthetic_maps[band])
        for stat in ('channel_mean', 'channel_norm', 'channel_population_std'):
            require(torch.allclose(tensor_stats(actual_maps[band])[stat], synth_stats[stat], atol=ATOL, rtol=RTOL),
                    f'synthetic site-permutation map {stat} invariance failed')
        synth_controls = source_controls(extractor, synthetic_maps[band])
        controls['synthetic_site_permutation'][f'T{band}'].update(actual_W=synthetic_maps[band],
            actual_W_stats=synth_stats, native_controls=synth_controls,
            control_changes_from_original={k: v - original_controls[k] for k, v in synth_controls.items()},
            provenance='synthetic within-map permutation of completed W, not naturally reached attention state',
            new_to_old_site_permutation=site_permutation, fine_action_equivariance_on_ties_asserted=False)
        controls['address_permutation'][f'T{band}'].update(
            original_item_actual_W=actual_maps[band],
            old_to_new_address_indices=torch.tensor(address_permutation))
    counts = {key: 2 * len(records) for key, records in control_records.items()}
    counts.update({key: 2 * sum(len(records) for records in by_band.values())
                   for key, by_band in permutation_records.items()})
    require(sum(2 * len(records) for records in primary.values()) == 1536, 'per-seed primary query count')
    require(all(counts[key] == 512 for key in control_records)
            and all(counts[key] == 1536 for key in permutation_records), 'per-seed control query count')
    return {'primary': results, 'controls': controls, 'original_native_controls': original_controls,
        'original_V_stats': original_stats, 'batched_operator_logits': batch_logits,
        'batched_operator_role_fields': batch_fields,
        'input_query_isolation_checks': isolation_checks,
        'query_counts': {'primary': 1536, **counts},
        'initial_state_contract': {'identity': 'role_pair', 'observed': True,
            'delivered_at': 0, 'W_completed_at': 0, 'snapshot_time': 0,
            'raw_logit_source': 'preserved original frozen orientation logit',
            'all_initial_actual_W_equal_V': True}}


def raw_assessment(head, standardizer, maps, targets, cells):
    with torch.no_grad():
        logits = head(standardizer.apply(maps))
        fields = task.role_fields(logits)
    selected_flat = fields.flatten(-2).argmax(-1)
    selected = torch.stack((selected_flat // 4, selected_flat % 4), -1)
    expected = field_stats(fields)['expected_column']
    difference = expected[..., 0] - expected[..., 1]
    coarse = torch.where(difference.abs() <= 1e-6, 0, torch.where(difference < 0, -1, 1))
    true_coarse = torch.where(cells[..., 0, 1] < cells[..., 1, 1], -1, 1)
    return {'actual_raw_grid': maps, 'local_logits': logits, 'role_fields': fields,
        'selected_cells': selected, 'fine_correct': (selected == cells).all(-1),
        'tie_counts': (fields.flatten(-2) == fields.flatten(-2).max(-1, keepdim=True).values).sum(-1),
        'coarse_codes': coarse, 'coarse_correct': coarse == true_coarse,
        'occupancy_loss': torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1),
            reduction='none').mean((-2, -1)).reshape(DIAGNOSTIC_WORLDS, ITEMS),
        'field_stats': field_stats(fields), 'provenance': 'upstream raw-pixel block-mean feasibility alternative'}


def aggregate(value, bootstrap_indices):
    value = value.double()
    per_world = value.reshape(DIAGNOSTIC_WORLDS, -1).mean(-1)
    interval = torch.quantile(per_world[bootstrap_indices].mean(-1),
                             torch.tensor([.025, .975], dtype=torch.float64))
    return {'unconditional_mean': float(per_world.mean()), 'per_world': per_world.tolist(),
            'conditional_world_bootstrap_95': interval.tolist(),
            'world_denominator': DIAGNOSTIC_WORLDS}


def summarize(seed, native, raw):
    indices = torch.randint(DIAGNOSTIC_WORLDS, (256, DIAGNOSTIC_WORLDS),
                            generator=torch.Generator().manual_seed(230000000 + seed))
    summaries = {}
    for key, data in {**native['primary'], 'raw_grid': raw}.items():
        measurements = {'fine_target_accuracy': data['fine_correct'],
            'coarse_order_accuracy': data['coarse_correct'], 'occupancy_loss': data['occupancy_loss'],
            'tie_frequency': data['tie_counts'] > 1, 'mean_tie_count': data['tie_counts'],
            'coarse_unknown_frequency': data['coarse_codes'] == 0, **data['field_stats']}
        summaries[key] = {name: aggregate(value, indices) for name, value in measurements.items()}
        summaries[key]['by_role'] = {role: {
            'fine_target_accuracy': aggregate(data['fine_correct'][..., i], indices),
            'tie_frequency': aggregate(data['tie_counts'][..., i] > 1, indices),
            'mean_tie_count': aggregate(data['tie_counts'][..., i], indices),
            **{name: aggregate(value[..., i], indices) for name, value in data['field_stats'].items()}}
            for i, role in enumerate(task.ROLES)}
    t1, t2 = native['primary']['T1'], native['primary']['T2']
    changed = (t2['role_fields'] - t1['role_fields']).abs().flatten(-3).max(-1).values > ATOL
    exact_changed = (t2['role_fields'] != t1['role_fields']).flatten(-3).any(-1)
    retained = t1['coarse_correct'] & t2['coarse_correct']
    gates = {'T1_fine_at_least_0_50': summaries['T1']['fine_target_accuracy']['unconditional_mean'] >= .50,
        'T1_coarse_at_least_0_75': summaries['T1']['coarse_order_accuracy']['unconditional_mean'] >= .75,
        'raw_fine_at_least_0_95': summaries['raw_grid']['fine_target_accuracy']['unconditional_mean'] >= .95}
    # A constant 2x2 block and positional tie rule can target only its upper-left site.
    cells = t2['selected_cells'].new_tensor([[0, 0], [0, 2], [2, 0], [2, 2]])
    truth = native['evaluator_cells']
    structural = (truth[..., None, :] == cells).all(-1).any(-1)
    require(not bool((t2['fine_correct'] & ~structural).any()), 'T2 exceeded structural fine ceiling')
    require(float(structural.double().mean()) == .25, 'declared balanced T2 structural ceiling is not .25')
    require(summaries['T4']['fine_target_accuracy']['unconditional_mean'] == .0625
            and summaries['T4']['coarse_unknown_frequency']['unconditional_mean'] == 1.,
            'balanced T4 policy consequence failed')
    return {'seed': seed, 'unconditional_primary': summaries,
        'T2_changed_and_correctly_retained_coarse_unconditional': aggregate(changed & retained, indices),
        'T2_exact_field_change_unconditional': aggregate(exact_changed, indices),
        'T2_structural_fine_ceiling': aggregate(structural, indices),
        'imposed_policy_consequences': {'T2_fine_ceiling': .25, 'T4_fine_accuracy': .0625,
            'T4_coarse_unknown_frequency': 1., 'T2_top_two_site_margin': 0.},
        'engineering_gates': gates, 'engineering_feasibility_pass': all(gates.values()),
        'query_counts': native['query_counts'], 'semantic_qualification': False,
        'goal_achieved': False}, indices


def frozen_weights(extractor):
    return {name: {k: v.detach().clone() for k, v in module.state_dict().items()}
            for name, module in (('encoder', extractor.encoder),
                                 ('orientation_head', extractor.orientation_head))}


def run_seed(seed, log, archive_failure):
    # This is the first development checkpoint load, after the execution claim.
    from grounded_outcome_data_v1 import FrozenSensor
    trace = {'seed': seed, 'development_only': True}
    try:
        sensor = FrozenSensor(seed)
        extractor = task.FrozenSpatialExtractor(sensor.encoder, sensor.readout,
                                                sensor.mean.float(), sensor.scale.float())
        before = frozen_weights(extractor)
        trace.update(frozen_source_weights=before,
            frozen_orientation_standardizer={'mean': extractor.mean64, 'scale': extractor.scale64})
        fit = world_set(seed, 0, FIT_WORLDS)
        trace['fit_worlds'] = fit
        diagnostic = world_set(seed, 1, DIAGNOSTIC_WORLDS)
        trace['diagnostic_worlds'] = diagnostic
        fit_maps, diagnostic_maps = extractor.extract(fit['pixels']), extractor.extract(diagnostic['pixels'])
        fit_raw = fit['pixels'].reshape(FIT_WORLDS, ITEMS, 3, 4, 4, 4, 4).mean((-3, -1))
        diagnostic_raw = diagnostic['pixels'].reshape(DIAGNOSTIC_WORLDS, ITEMS, 3, 4, 4, 4, 4).mean((-3, -1))
        trace.update(fit_V=fit_maps, diagnostic_V=diagnostic_maps, fit_raw_grid=fit_raw,
                     diagnostic_raw_grid=diagnostic_raw, fit_native_controls=source_controls(extractor, fit_maps))
        head, standardizer, native_fit = fit_head(seed, 'native24', fit_maps, fit['occupancy_targets'], log)
        trace['native_head_fit'] = native_fit
        raw_head, raw_standardizer, raw_fit = fit_head(seed, 'raw3', fit_raw, fit['occupancy_targets'], log)
        trace['raw_head_fit'] = raw_fit
        controller = task.NativeController(head, standardizer)
        trace['native_execution_partial'] = {}
        native = evaluate_native(seed, diagnostic, diagnostic_maps, extractor, controller,
                                 trace['native_execution_partial'])
        native['evaluator_cells'] = diagnostic['evaluator_cells']
        trace['native_execution'] = native
        del trace['native_execution_partial']
        raw = raw_assessment(raw_head, raw_standardizer, diagnostic_raw,
                             diagnostic['occupancy_targets'], diagnostic['evaluator_cells'])
        trace['raw_grid_diagnostic'] = raw
        after = frozen_weights(extractor)
        trace['frozen_source_weights_after'] = after
        require(all(torch.equal(value, after[name][key]) for name, weights in before.items()
                    for key, value in weights.items()), 'frozen source weights changed')
        trace['frozen_weight_equality_checks'] = {'encoder': True, 'orientation_head': True}
        summary, indices = summarize(seed, native, raw)
        trace.update(bootstrap_indices=indices, software_invariants_pass=True)
        finite_tree(trace)
        return trace, summary
    except BaseException:
        trace.update(status='failed_single_attempt', traceback=traceback.format_exc(), retry_allowed=False)
        archive_failure(trace)
        raise


def run(args):
    require(args.execution and args.output_dir, '--run requires --execution and --output-dir')
    registration = validate_registration(args)
    started = time.monotonic()
    execution_path = ROOT / relative(args.execution)
    output_dir = ROOT / relative(args.output_dir)
    require(not output_dir.exists(), 'output directory already exists; refusing overwrite/retry')
    require(execution_path != ROOT / relative(args.registration), 'execution and registration paths must differ')
    # A durable, exclusive claim is written before directory creation/imports/loads/worlds.
    exclusive_json(execution_path, {'status': 'claimed', 'single_attempt': True,
        'registration_sha256': digest(ROOT / relative(args.registration)),
        'registration_commit': args.registration_commit,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'output_dir': relative(output_dir), 'goal_achieved': False})
    log_stream = None
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        log_stream = (output_dir / 'first_execution.log').open('x', encoding='utf-8', buffering=1)
        def log(value):
            line = json.dumps(value, allow_nan=False)
            log_stream.write(line + '\n')
            log_stream.flush()
            print(line, flush=True)
        # Runtime imports happen only after reservation and log setup.
        global torch, task
        sys.path.insert(0, str(ROOT / 'src'))
        sys.path.insert(0, str(ROOT / 'scripts'))
        import torch
        import spatial_native_task_v1 as task
        require(Path(task.__file__).resolve() == ROOT / 'scripts/spatial_native_task_v1.py',
                'task module resolved outside bound source')
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        os.chdir(ROOT)
        log({'kind': 'execution_started', 'torch_version': str(torch.__version__),
            'python_version': sys.version, 'threads': torch.get_num_threads(),
            'development_only': True, 'registration_sha256': digest(ROOT / relative(args.registration))})
        models, archives = {}, {}
        def archive_failure(trace):
            path = output_dir / f"failed-states-{trace['seed']}.pt.gz"
            with path.open('xb') as raw_stream:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw_stream, mtime=0) as compressed:
                    torch.save(trace, compressed)
            log({'kind': 'failed_seed_archive', 'seed': trace['seed'],
                 'path': relative(path), 'sha256': digest(path), 'retry_allowed': False})
        for seed in SEEDS:
            check_hashes({**registration['source_sha256'], **registration['review_sha256'],
                          **registration['checkpoint_sha256']})
            trace, summary = run_seed(seed, log, archive_failure)
            archive = output_dir / f'states-{seed}.pt.gz'
            with archive.open('xb') as raw_stream:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw_stream, mtime=0) as compressed:
                    torch.save(trace, compressed)
            archives[relative(archive)] = digest(archive)
            models[str(seed)] = summary
            log({'kind': 'seed_completed', 'seed': seed,
                 'engineering_feasibility_pass': summary['engineering_feasibility_pass'],
                 'archive_sha256': archives[relative(archive)]})
            del trace
        counts = {key: sum(model['query_counts'][key] for model in models.values())
                  for key in next(iter(models.values()))['query_counts']}
        require(counts['primary'] == 4608 and all(counts[k] == 1536
                for k in ('sham_T1', 'matched_exogenous_T2', 'restore_after_T2'))
                and all(counts[k] == 4608 for k in ('address_permutation', 'synthetic_site_permutation')),
                'full development query counts mismatch')
        results = {'kind': 'spatial_native_task_v1_development', 'models': models,
            'all_engineering_feasibility_gates_pass': all(m['engineering_feasibility_pass'] for m in models.values()),
            'all_software_invariants_pass': True, 'query_counts': counts,
            'primary_item_operator_opportunities': 2304, 'all_outcomes_retained': True,
            'item_opportunity_counts': {key: value // 2 for key, value in counts.items()},
            'raw_grid_feasibility_role_queries': 1536,
            'goal_achieved': False, 'presentation_qualified': False, 'attention_model_A_tested': False,
            'new_reports': 0, 'development_only': True}
        exclusive_json(output_dir / 'results.json', results)
        log({'kind': 'execution_completed', 'all_engineering_feasibility_gates_pass':
             results['all_engineering_feasibility_gates_pass'], 'elapsed_seconds': time.monotonic() - started})
        log_stream.flush()
        os.fsync(log_stream.fileno())
        exclusive_json(output_dir / 'completion.json', {'status': 'completed',
            'archive_sha256': archives, 'results_sha256': digest(output_dir / 'results.json'),
            'first_execution_log_sha256': digest(output_dir / 'first_execution.log'),
            'registration_sha256': digest(ROOT / relative(args.registration)),
            'torch_version': str(torch.__version__), 'python_version': sys.version,
            'cpu_threads': torch.get_num_threads(), 'elapsed_seconds': time.monotonic() - started,
            'all_frozen_encoder_orientation_weights_unchanged': True,
            'goal_achieved': False})
    except BaseException:
        failure = {'status': 'failed_single_attempt', 'traceback': traceback.format_exc(),
                   'elapsed_seconds': time.monotonic() - started,
                   'goal_achieved': False, 'retry_allowed': False}
        if log_stream is not None:
            log_stream.write(json.dumps(failure) + '\n')
            log_stream.flush()
            os.fsync(log_stream.fileno())
        if output_dir.is_dir():
            exclusive_json(output_dir / 'failure.json', failure)
        raise
    finally:
        if log_stream is not None:
            log_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true', help='hash and register exact reviewed source; no execution')
    mode.add_argument('--run', action='store_true', help='claim and execute the single committed development registration')
    parser.add_argument('--registration', required=True, help='repository-relative registration JSON, created exclusively')
    parser.add_argument('--review', help='independent approved exact-hash JSON receipt for registration')
    parser.add_argument('--source', action='append', default=[], help='additional reviewed source/artifact path binding')
    parser.add_argument('--source-commit', help='full preserved reviewed source commit for --register')
    parser.add_argument('--preservation-checkout', help='read-only path to checkout containing preservation commits')
    parser.add_argument('--registration-commit', help='full committed preexecution registration commit for --run')
    parser.add_argument('--execution', help='fixed exclusive claim JSON path declared at registration and used for execution')
    parser.add_argument('--output-dir', help='fixed new archive directory declared at registration and used for execution')
    args = parser.parse_args()
    if args.register:
        require(not args.registration_commit, '--registration-commit cannot accompany --register')
        register(args)
    else:
        require(not args.review and not args.source and not args.source_commit,
                'registration-only arguments cannot accompany --run')
        run(args)


if __name__ == '__main__':
    main()
