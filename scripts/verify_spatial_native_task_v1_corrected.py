"""Corrected-reference revalidation of preserved results, not a study retry.

Do not execute until this exact corrected reference and its design are committed
and authorized. The original failed verifier/review and arithmetic diagnosis stay
preserved. The same core checks, gates and tolerances reconstruct the same first
results. Production Controller replay remains separately named input isolation.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
SEEDS = (1301, 1311, 1321)
BANDS = (1, 2, 4)
ATOL, RTOL = 1e-6, 1e-5
ROLES = ('red', 'blue')
ADDRESSES = ('l0', 'l1', 'l2', 'l3')
SITE_ROLL = (15,) + tuple(range(15))
ADDRESS_ROLL = (1, 2, 3, 0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks, self.errors = [], []

    def check(self, name, condition, **evidence):
        passed = bool(condition)
        self.checks.append({'name': name, 'passed': passed, **evidence})
        if not passed:
            self.errors.append(name)

    def compare(self, name, actual, expected, exact=False, atol=ATOL, rtol=RTOL):
        actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
        same_shape = actual.shape == expected.shape
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        if same_shape and finite:
            delta = (actual.double() - expected.double()).abs()
            within = delta <= atol + rtol * expected.double().abs()
            passed = (actual.dtype == expected.dtype and torch.equal(actual, expected)) if exact else bool(within.all())
            maximum = float(delta.max()) if delta.numel() else 0.
            different = int((actual != expected).sum())
            violations = int((~within).sum())
        else:
            passed, maximum, different, violations = False, None, None, None
        self.check(name, passed, actual_shape=list(actual.shape), expected_shape=list(expected.shape),
                   actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype), finite=finite,
                   exact=exact, atol=0. if exact else atol, rtol=0. if exact else rtol,
                   maximum_absolute_difference=maximum, unequal_elements=different,
                   tolerance_violations=violations, compared_elements=actual.numel())

    def tree(self, name, actual, expected):
        if isinstance(expected, dict):
            self.check(name + '/keys', set(actual) == set(expected), actual=sorted(actual), expected=sorted(expected))
            for key in expected:
                if key in actual:
                    self.tree(name + '/' + key, actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.compare(name, actual, expected)
        elif isinstance(expected, bool) or isinstance(expected, str) or expected is None:
            self.check(name, actual == expected, actual=actual, expected=expected)
        else:
            self.compare(name, actual, expected)


def block_reference(v, band):
    """Explicit disjoint float64 site sums, then return archived input dtype."""
    if band == 1:
        return v.clone()
    result = torch.empty_like(v)
    for row in range(0, 4, band):
        for column in range(0, 4, band):
            value = (v[..., row:row + band, column:column + band].double() / (band * band)).sum((-2, -1))
            result[..., row:row + band, column:column + band] = value.to(v.dtype)[..., None, None]
    return result


def raw_reference(pixels):
    grid = torch.empty(*pixels.shape[:-3], 3, 4, 4, dtype=torch.float32)
    for row in range(4):
        for col in range(4):
            # Independent indexing; float32 matches the specified raw-grid mean.
            grid[..., row, col] = pixels[..., row*4:row*4+4, col*4:col*4+4].mean((-2, -1))
    return grid


def standardizer_reference(maps):
    channels = maps.shape[-3]
    samples = maps.transpose(-3, -1).reshape(-1, channels)
    mean = samples.mean(0)
    # Explicit population variance, avoiding the production std primitive.
    scale = ((samples - mean).square().mean(0)).sqrt()
    scale = torch.where(scale < 1e-6, 1., scale)
    return {'mean': mean.tolist(), 'scale': scale.tolist()}


def head_reference(maps, fit, itemwise=False):
    params, stats = fit['weights'], fit['fit_only_standardizer']
    x = (maps - torch.tensor(stats['mean'], dtype=torch.float32)[:, None, None]) / torch.tensor(
        stats['scale'], dtype=torch.float32)[:, None, None]
    x = x.movedim(-3, -1)
    # Keep explicit independent matmul and separate bias equations. Native
    # action paths consume one [4,4,24] item; batched fitting/raw paths retain
    # their original input shape. The preserved diagnosis isolates this choice.
    def equations(value):
        hidden = (value @ params['net.0.weight'].T + params['net.0.bias']).clamp_min(0)
        return hidden @ params['net.2.weight'].T + params['net.2.bias']
    if itemwise:
        leading = x.shape[:-3]
        items = x.reshape(-1, *x.shape[-3:])
        output = torch.stack([equations(item) for item in items])
        return output.reshape(*leading, 4, 4, 3).movedim(-1, -3)
    return equations(x).movedim(-1, -3)


def fields_reference(logits):
    score = (logits[..., 1:, :, :].double() - logits[..., :1, :, :].double()).flatten(-2)
    unnormalized = (score - score.max(-1, keepdim=True).values).exp()
    field = unnormalized / unnormalized.sum(-1, keepdim=True)
    # Identical score sites give exact 1/16 without a threshold or presence test.
    return field.reshape(*logits.shape[:-3], 2, 4, 4).to(logits.dtype)


def ce_reference(logits, targets):
    value = logits.double()
    maximum = value.max(-3, keepdim=True).values
    log_denominator = maximum.squeeze(-3) + (value - maximum).exp().sum(-3).log()
    chosen = value.gather(-3, targets.unsqueeze(-3)).squeeze(-3)
    return (log_denominator - chosen).mean((-2, -1))


def field_statistics(fields):
    flat = fields.double().flatten(-2)
    maximum = flat.max(-1).values
    # Sort only to measure second-site margin; ties remain exact field comparisons.
    second = flat.sort(-1).values[..., -2]
    entropy = torch.where(flat > 0, -flat * flat.clamp_min(1e-300).log(), 0.).sum(-1)
    expected = sum(fields[..., :, :, c].double().sum(-1) * c for c in range(4))
    return {'entropy': entropy, 'maximum': maximum, 'margin': maximum - second,
            'expected_column': expected}


def assessment(logits, targets, cells, fields=None):
    fields = fields_reference(logits) if fields is None else fields
    flat = fields.flatten(-2)
    indices = flat.argmax(-1)
    selected = torch.stack((indices // 4, indices % 4), -1)
    ties = (flat == flat.max(-1, keepdim=True).values).sum(-1)
    stats = field_statistics(fields)
    delta = stats['expected_column'][..., 0] - stats['expected_column'][..., 1]
    coarse = torch.where(delta.abs() <= 1e-6, 0, torch.where(delta < 0, -1, 1))
    truth = torch.where(cells[..., 0, 1] < cells[..., 1, 1], -1, 1)
    return {'local_logits': logits, 'role_fields': fields, 'selected_cells': selected,
            'tie_counts': ties, 'coarse_codes': coarse, 'evaluator_coarse_codes': truth,
            'occupancy_loss': ce_reference(logits, targets),
            'fine_correct': (selected == cells).all(-1), 'coarse_correct': coarse == truth,
            'field_stats': stats}


def map_statistics(maps):
    value = maps.double().flatten(-2)
    mean = value.mean(-1)
    return {'channel_mean': mean, 'channel_norm': value.square().sum(-1).sqrt(),
            'channel_population_std': ((value - mean[..., None]).square().mean(-1)).sqrt()}


def source_reference(pixels, weights):
    """Frozen encoder equations reconstructed from archived state dict, no loader."""
    x = pixels.reshape(-1, 3, 16, 16)
    for layer in (0, 3):
        x = torch.nn.functional.conv2d(x, weights[f'net.{layer}.weight'], weights[f'net.{layer}.bias'], padding=1)
        x = torch.nn.functional.max_pool2d(x.clamp_min(0), 2)
    return x.reshape(*pixels.shape[:-3], 24, 4, 4)


def controls_reference(maps, trace):
    encoder = trace['frozen_source_weights']['encoder']
    orientation = trace['frozen_source_weights']['orientation_head']
    x = maps.reshape(-1, 384)
    # Independent equations use the registered flattened float32 F.linear
    # dispatch. No production suffix, extractor or control helper is called.
    hidden = torch.nn.functional.linear(x, encoder['net.7.weight'], encoder['net.7.bias']).clamp_min(0)
    logits = torch.nn.functional.linear(hidden, encoder['net.9.weight'], encoder['net.9.bias'])
    stats = trace['frozen_orientation_standardizer']
    standardized = (hidden - stats['mean']) / stats['scale']
    ori_hidden = torch.nn.functional.linear(standardized, orientation['0.weight'], orientation['0.bias']).clamp_min(0)
    ori = torch.nn.functional.linear(ori_hidden, orientation['2.weight'], orientation['2.bias'])[:, 0]
    leading = maps.shape[:-3]
    coordinate = ori.sigmoid()
    return {'category_logits': logits.reshape(*leading, 8),
            'category_probabilities': torch.cat((logits[:, :4].softmax(-1), logits[:, 4:].softmax(-1)), -1).reshape(*leading, 8),
            'orientation_hidden': hidden.reshape(*leading, 64), 'orientation_logits': ori.reshape(leading),
            'orientation_standardized_hidden': standardized.reshape(*leading, 64),
            'orientation_uncalibrated_sigmoid_coordinates': torch.stack((1 - coordinate, coordinate), -1).reshape(*leading, 2)}


def aggregate_reference(value, indices):
    worlds = value.double().reshape(64, -1).mean(-1)
    means = worlds[indices].mean(-1).sort().values
    # Linear quantile interpolation independently replaces torch.quantile.
    interval = []
    for probability in (.025, .975):
        position = probability * 255
        low, high = math.floor(position), math.ceil(position)
        interval.append(float(means[low] * (high - position) + means[high] * (position - low))
                        if low != high else float(means[low]))
    return {'unconditional_mean': float(worlds.mean()), 'per_world': worlds.tolist(),
            'conditional_world_bootstrap_95': interval, 'world_denominator': 64}


def summarize_reference(seed, data, indices, cells):
    out = {}
    for key, row in data.items():
        metrics = {'fine_target_accuracy': row['fine_correct'], 'coarse_order_accuracy': row['coarse_correct'],
                   'occupancy_loss': row['occupancy_loss'], 'tie_frequency': row['tie_counts'] > 1,
                   'mean_tie_count': row['tie_counts'], 'coarse_unknown_frequency': row['coarse_codes'] == 0,
                   **row['field_stats']}
        out[key] = {name: aggregate_reference(value, indices) for name, value in metrics.items()}
        out[key]['by_role'] = {role: {name: aggregate_reference(value[..., i], indices) for name, value in {
            'fine_target_accuracy': row['fine_correct'], 'tie_frequency': row['tie_counts'] > 1,
            'mean_tie_count': row['tie_counts'], **row['field_stats']}.items()} for i, role in enumerate(ROLES)}
    changed = (data['T2']['role_fields'] - data['T1']['role_fields']).abs().flatten(-3).max(-1).values > ATOL
    exact_changed = (data['T2']['role_fields'] != data['T1']['role_fields']).flatten(-3).any(-1)
    retained = data['T1']['coarse_correct'] & data['T2']['coarse_correct']
    structural = ((cells[..., 0] % 2 == 0) & (cells[..., 1] % 2 == 0))
    gates = {'T1_fine_at_least_0_50': out['T1']['fine_target_accuracy']['unconditional_mean'] >= .5,
             'T1_coarse_at_least_0_75': out['T1']['coarse_order_accuracy']['unconditional_mean'] >= .75,
             'raw_fine_at_least_0_95': out['raw_grid']['fine_target_accuracy']['unconditional_mean'] >= .95}
    return {'seed': seed, 'unconditional_primary': out,
            'T2_changed_and_correctly_retained_coarse_unconditional': aggregate_reference(changed & retained, indices),
            'T2_exact_field_change_unconditional': aggregate_reference(exact_changed, indices),
            'T2_structural_fine_ceiling': aggregate_reference(structural, indices),
            'imposed_policy_consequences': {'T2_fine_ceiling': .25, 'T4_fine_accuracy': .0625,
                                          'T4_coarse_unknown_frequency': 1., 'T2_top_two_site_margin': 0.},
            'engineering_gates': gates, 'engineering_feasibility_pass': all(gates.values()),
            'query_counts': {'primary': 1536, 'sham_T1': 512, 'matched_exogenous_T2': 512,
                             'restore_after_T2': 512, 'address_permutation': 1536,
                             'synthetic_site_permutation': 1536}, 'semantic_qualification': False, 'goal_achieved': False}


def layout_reference(count):
    """Reconstruct archived evaluator layout only; no rendering/world sampling."""
    cells = torch.empty(count, 4, 2, 2, dtype=torch.long)
    for world in range(count):
        for item in range(4):
            code = (world % 128) ^ ((item * 32) | (item % 2))
            side = code & 1
            cells[world, item, 0] = torch.tensor(((code >> 2) & 3, ((code >> 1) & 1) + 2 * side))
            cells[world, item, 1] = torch.tensor(((((code >> 3) ^ (code >> 5)) & 1) + 2 * ((code >> 4) & 1),
                (((code >> 1) ^ (code >> 4) ^ (code >> 6)) & 1) + 2 * (1 - side)))
    return cells


def occupancy_reference(cells):
    targets = torch.zeros(*cells.shape[:-2], 4, 4, dtype=torch.long)
    for world in range(cells.shape[0]):
        for item in range(4):
            for role in range(2):
                row, col = cells[world, item, role]
                targets[world, item, row, col] = role + 1
    return targets


def verify_records(audit, name, row, expected, band=None, remapped=False, synthetic=False):
    records = row['native_records']
    audit.check(name + '/record_count', len(records) == 256, actual=len(records), expected=256)
    selected, ties, columns = [], [], []
    for ordinal, record in enumerate(records):
        world, item = divmod(ordinal, 4)
        address = ADDRESSES[ADDRESS_ROLL[item] if remapped else item]
        prefix = f'{name}/record/{world}/{item}'
        audit.check(prefix + '/query', record['queried_address'] == address, actual=record['queried_address'], expected=address)
        audit.compare(prefix + '/logits', record['logits'], expected['local_logits'][world, item])
        audit.compare(prefix + '/fields', record['role_fields'], expected['role_fields'][world, item])
        # Exact policy is checked against the actual emitted field, so floating
        # reconstruction tolerance cannot erase ties or change recorded actions.
        field = record['role_fields'].flatten(-2)
        actions = record['actions']
        audit.check(prefix + '/roles', len(actions) == 2 and tuple(a['role'] for a in actions) == ROLES)
        selected.append([[a['row'], a['column']] for a in actions])
        ties.append([len(a['tied_cells']) for a in actions])
        for role, action in enumerate(actions):
            exact_sites = (field[role] == field[role].max()).nonzero().flatten().tolist()
            expected_ties = tuple((s // 4, s % 4) for s in exact_sites)
            audit.check(prefix + '/action/' + ROLES[role], action['address'] == address
                and (action['row'], action['column']) == expected_ties[0]
                and tuple(tuple(x) for x in action['tied_cells']) == expected_ties,
                emitted=[action['row'], action['column']], exact_tied_cells=expected_ties)
        col = field_statistics(record['role_fields'])['expected_column']
        columns.append(record['coarse']['expected_columns'])
        delta = float(col[0] - col[1])
        order = 'unknown' if abs(delta) <= 1e-6 else ('red_left_of_blue' if delta < 0 else 'blue_left_of_red')
        audit.check(prefix + '/coarse', record['coarse']['address'] == address and record['coarse']['order'] == order,
                    emitted=record['coarse']['order'], reconstructed=order)
        if band is not None:
            audit.check(prefix + '/time', record['snapshot_time'] == 1, actual=record['snapshot_time'], expected=1)
        if remapped or synthetic:
            audit.check(prefix + '/opportunity_key', (record['world_index'], record['item_index'], record['bandwidth']) == (world, item, band))
        if remapped:
            event = {'address': address, 'bandwidth': band, 'issued_at': 0, 'completed_at': 1, 'driver': 'endogenous'}
            audit.check(prefix + '/address_provenance', record['original_address'] == ADDRESSES[item]
                        and record['remapped_completed_history'] == [event], remapped_history=record['remapped_completed_history'])
    audit.compare(name + '/records_vs_stacked_selected', torch.tensor(selected).reshape(64, 4, 2, 2), row['selected_cells'], exact=True)
    audit.compare(name + '/records_vs_stacked_ties', torch.tensor(ties).reshape(64, 4, 2), row['tie_counts'], exact=True)
    audit.compare(name + '/record_centroids', torch.tensor(columns, dtype=torch.float64).reshape(64, 4, 2), expected['field_stats']['expected_column'])


def verify_assessment(audit, name, actual, expected):
    for key in ('local_logits', 'role_fields'):
        audit.check(name + '/' + key + '/float32', actual[key].dtype == torch.float32, actual_dtype=str(actual[key].dtype))
    for key in expected:
        if key not in actual:
            # Raw-grid assessor does not duplicate evaluator coarse labels.
            audit.check(name + '/' + key + '/present', key == 'evaluator_coarse_codes')
        elif key == 'field_stats':
            for stat, value in expected[key].items():
                audit.compare(name + '/field_stats/' + stat, actual[key][stat], value)
        else:
            audit.compare(name + '/' + key, actual[key], expected[key], exact=key in (
                'selected_cells', 'tie_counts', 'coarse_codes', 'evaluator_coarse_codes', 'fine_correct', 'coarse_correct'))


def hash_bindings(audit, name, hashes):
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve()
        valid = path.is_relative_to(ROOT) and path.is_file()
        actual = digest(path) if valid else None
        audit.check(name + '/' + relative, valid and actual == expected, actual_sha256=actual, expected_sha256=expected)


def committed_hashes(audit, checkout, commit, hashes, name):
    """Read-only Git at the explicitly supplied preservation checkout."""
    for path, expected in hashes.items():
        process = subprocess.Popen(['git', '-C', str(checkout), 'show', f'{commit}:{path}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        h = hashlib.sha256()
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b''):
            h.update(chunk)
        error = process.stderr.read().decode(errors='replace')
        status = process.wait()
        audit.check(name + '/' + path, status == 0 and h.hexdigest() == expected,
                    commit=commit, expected_sha256=expected, committed_sha256=h.hexdigest(), git_exit=status, git_error=error)


def finite_tree(audit, value, name):
    if isinstance(value, torch.Tensor):
        audit.check(name + '/finite', bool(torch.isfinite(value).all()), shape=list(value.shape), dtype=str(value.dtype))
        audit.check(name + '/cpu_detached', value.device.type == 'cpu' and not value.requires_grad)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(audit, item, name + '/' + str(key))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            finite_tree(audit, item, name + '/' + str(i))
    elif isinstance(value, float):
        audit.check(name + '/finite', math.isfinite(value))


def verify_checkpoint_values(audit, seed, trace):
    # This is post-preservation read-only loading of registered checkpoints,
    # never FrozenSensor instantiation or any optimizer operation.
    encoder = torch.load(ROOT / f'audits/bound_content/confirmation_v2/seed{seed}.pt', weights_only=True, map_location='cpu')['state_dict']
    with gzip.open(ROOT / f'audits/sensory_task_adapter_v1/states-{seed}.pt.gz', 'rb') as stream:
        orientation = torch.load(stream, weights_only=True, map_location='cpu')['instruments']['visual_hidden']['readout']
    for name, expected in (('encoder', encoder), ('orientation_head', orientation['state_dict'])):
        before = trace['frozen_source_weights'][name]
        after = trace['frozen_source_weights_after'][name]
        audit.check(f'{seed}/frozen/{name}/keys', set(before) == set(expected) == set(after))
        for key in expected:
            audit.compare(f'{seed}/checkpoint/{name}/{key}', before[key], expected[key], exact=True)
            audit.compare(f'{seed}/frozen_after/{name}/{key}', after[key], before[key], exact=True)
    for key in ('mean', 'scale'):
        audit.compare(f'{seed}/orientation_standardizer/{key}', trace['frozen_orientation_standardizer'][key], orientation[key], exact=True)


def replay_input_boundary(audit, seed, trace):
    """Production-consumer reenactment, explicitly separate from reference math."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    import spatial_native_task_v1 as task
    audit.check(f'{seed}/controller/bound_source', Path(task.__file__).resolve() == ROOT / 'scripts/spatial_native_task_v1.py')
    fit = trace['native_head_fit']
    with torch.random.fork_rng():
        head = task.LocalOccupancyHead(24)
    head.load_state_dict(fit['weights'])
    head.eval().requires_grad_(False)
    controller = task.NativeController(head, task.ChannelStandardizer(
        tuple(fit['fit_only_standardizer']['mean']), tuple(fit['fit_only_standardizer']['scale'])))
    audit.check(f'{seed}/controller/no_truth_argument', tuple(inspect.signature(controller.consume).parameters) == ('state',))
    actual_isolation = trace['native_execution']['input_query_isolation_checks']
    audit.check(f'{seed}/isolation/schedule_count', len(actual_isolation) == 768, actual=len(actual_isolation), expected=768)
    for world in range(64):
        for item, address in enumerate(ADDRESSES):
            for band_index, band in enumerate(BANDS):
                name = f'{seed}/controller_replay/{world}/{item}/T{band}'
                row = trace['native_execution']['primary'][f'T{band}']
                # Completed selected W; untouched nonselected W is saved V.
                items = tuple(task.AddressedItem.from_tensors(a, trace['diagnostic_V'][world, j],
                    row['actual_W'][world, j] if j == item else trace['diagnostic_V'][world, j],
                    True, 'role_pair', float(trace['native_execution']['original_native_controls']['orientation_logits'][world, j]),
                    0, 1 if j == item else 0) for j, a in enumerate(ADDRESSES))
                state = task.AddressedState(items, 1)
                base = controller.consume(state)
                isolated_items = tuple(replace(x, source_v=task.SpatialBuffer.from_tensor(torch.zeros(24, 4, 4)),
                    identity='counterfactual_retained_record', raw_logit=x.raw_logit + 1000.,
                    actual_w=x.actual_w if x.address == address else task.SpatialBuffer.from_tensor(torch.zeros(24, 4, 4))) for x in items)
                decoy = controller.consume(task.AddressedState(isolated_items, 1))
                left, right = base.item(address), decoy.item(address)
                audit.compare(name + '/logits_vs_archived', left.local_logits.tensor(), row['local_logits'][world, item], exact=True)
                audit.compare(name + '/fields_vs_archived', left.role_fields.tensor(), row['role_fields'][world, item], exact=True)
                audit.compare(name + '/decoy_logits', right.local_logits.tensor(), left.local_logits.tensor(), exact=True)
                audit.compare(name + '/decoy_fields', right.role_fields.tensor(), left.role_fields.tensor(), exact=True)
                archived = actual_isolation[(world * 4 + item) * 3 + band_index]
                audit.check(name + '/isolation_key', (archived['world_index'], archived['item_index'], archived['bandwidth'], archived['queried_address']) == (world, item, band, address))
                # Retained actual decoy outputs are checked if supplied. The live
                # reenactment above supplies values, never trusts pass booleans.
                for key, tensor in (('isolated_local_logits', right.local_logits.tensor()), ('isolated_role_fields', right.role_fields.tensor())):
                    audit.compare(name + '/archived_' + key, archived[key], tensor, exact=True)
                audit.compare(name + '/counterfactual_truth', archived['counterfactual_evaluator_truth'],
                              trace['diagnostic_worlds']['evaluator_cells'][world, item].flip(0), exact=True)
                audit.check(name + '/decoy_contract', archived['identity_replacement'] == 'counterfactual_retained_record'
                    and archived['raw_logit_delta'] == 1000. and archived['nonqueried_actual_W_replacement'] == 'all zeros'
                    and archived['all_source_V_replacement'] == 'all zeros')
                for role in ROLES:
                    a, b = asdict(controller.target(base, address, role)), asdict(controller.target(decoy, address, role))
                    audit.check(name + '/decoy_action/' + role, a == b, base=a, decoy=b)
                    original_action = row['native_records'][world * 4 + item]['actions'][ROLES.index(role)]
                    audit.check(name + '/action_vs_archived/' + role, a == original_action, reconstructed=a, archived=original_action)
                audit.check(name + '/decoy_coarse', asdict(controller.coarse_order(base, address)) == asdict(controller.coarse_order(decoy, address)))
                audit.check(name + '/coarse_vs_archived', asdict(controller.coarse_order(base, address)) == row['native_records'][world * 4 + item]['coarse'])
    refused = False
    try:
        controller.consume(state, trace['diagnostic_worlds']['evaluator_cells'])
    except TypeError:
        refused = True
    audit.check(f'{seed}/controller/truth_argument_refused', refused)


def verify_seed(audit, seed, trace, summary, log):
    prefix = str(seed)
    finite_tree(audit, trace, prefix + '/archive')
    audit.check(prefix + '/seed', trace['seed'] == seed and trace['development_only'] is True)
    verify_checkpoint_values(audit, seed, trace)
    for split, count in (('fit', 128), ('diagnostic', 64)):
        world = trace[split + '_worlds']
        namespace = 210000000 + seed * 1000 + (100000 if split == 'diagnostic' else 0)
        audit.check(prefix + '/' + split + '/namespace', world['world_namespace'] == namespace, actual=world['world_namespace'], expected=namespace)
        audit.compare(prefix + '/' + split + '/indices', world['world_indices'], torch.arange(count), exact=True)
        audit.check(prefix + '/' + split + '/pixels_shape', world['pixels'].shape == (count, 4, 3, 16, 16) and world['pixels'].dtype == torch.float32)
        audit.check(prefix + '/' + split + '/pixel_range', bool(((world['pixels'] >= 0) & (world['pixels'] <= 1)).all()))
        cells = layout_reference(count)
        audit.compare(prefix + '/' + split + '/cells', world['evaluator_cells'], cells, exact=True)
        audit.compare(prefix + '/' + split + '/targets', world['occupancy_targets'], occupancy_reference(cells), exact=True)
        for role in range(2):
            hist = torch.bincount((cells[..., role, 0] * 4 + cells[..., role, 1]).flatten(), minlength=16)
            audit.compare(prefix + '/' + split + '/role_balance/' + ROLES[role], hist, torch.full((16,), count // 4), exact=True)
        audit.compare(prefix + '/' + split + '/V', trace[split + '_V'], source_reference(world['pixels'], trace['frozen_source_weights']['encoder']))
        audit.compare(prefix + '/' + split + '/raw_grid', trace[split + '_raw_grid'], raw_reference(world['pixels']))
        for key in (split + '_V', split + '_raw_grid'):
            audit.check(prefix + '/' + key + '/float32', trace[key].dtype == torch.float32, actual_dtype=str(trace[key].dtype))
        for label, key in (('native24', split + '_V'), ('raw3', split + '_raw_grid')):
            if split != 'fit':
                continue
            fit = trace['native_head_fit' if label == 'native24' else 'raw_head_fit']
            expected_seed = 220000000 + seed + (1000000 if label == 'raw3' else 0)
            audit.check(prefix + '/' + label + '/init_seed', fit['initialization_seed'] == expected_seed)
            with torch.random.fork_rng():
                torch.manual_seed(expected_seed)
                first = torch.nn.Linear(24 if label == 'native24' else 3, 32)
                last = torch.nn.Linear(32, 3)
            for name, value in {'net.0.weight': first.weight, 'net.0.bias': first.bias,
                                'net.2.weight': last.weight, 'net.2.bias': last.bias}.items():
                audit.compare(prefix + '/' + label + '/initial_weights/' + name, fit['initial_weights'][name], value.detach(), exact=True)
            stats = standardizer_reference(trace[key])
            audit.tree(prefix + '/' + label + '/fit_only_standardizer', fit['fit_only_standardizer'], stats)
            parameters = sum(v.numel() for v in fit['weights'].values())
            audit.check(prefix + '/' + label + '/parameter_count', parameters == (899 if label == 'native24' else 227), actual=parameters)
            audit.check(prefix + '/' + label + '/updates', fit['losses_preupdate'].shape == (500,))
            entries = [e for e in log if e.get('kind') == 'fit_loss' and e.get('seed') == seed and e.get('head') == label]
            audit.check(prefix + '/' + label + '/logged_updates', [e['update'] for e in entries] == list(range(1, 501)), actual_count=len(entries))
            audit.compare(prefix + '/' + label + '/loss_log', fit['losses_preupdate'], [e['preupdate_loss'] for e in entries], exact=True)
            logits = head_reference(trace[key], fit)
            audit.compare(prefix + '/' + label + '/final_fit_logits', fit['final_fit_logits'], logits)
            audit.compare(prefix + '/' + label + '/final_fit_CE', fit['final_fit_loss'], ce_reference(logits, world['occupancy_targets']).mean())
            initial = {**fit, 'weights': fit['initial_weights']}
            audit.compare(prefix + '/' + label + '/initial_CE', fit['losses_preupdate'][0], ce_reference(head_reference(trace[key], initial), world['occupancy_targets']).mean())
            completed = [e for e in log if e.get('kind') == 'fit_finished' and e.get('seed') == seed and e.get('head') == label]
            audit.check(prefix + '/' + label + '/finished_count', len(completed) == 1 and completed[0]['steps'] == 500)
            if completed:
                audit.compare(prefix + '/' + label + '/finished_loss', completed[0]['final_loss'], fit['final_fit_loss'])
    native = trace['native_execution']
    cells = trace['diagnostic_worlds']['evaluator_cells']
    targets = trace['diagnostic_worlds']['occupancy_targets']
    audit.compare(prefix + '/native/cells', native['evaluator_cells'], cells, exact=True)
    baseline_controls = controls_reference(trace['diagnostic_V'], trace)
    for key, value in baseline_controls.items():
        audit.compare(prefix + '/original_controls/' + key, native['original_native_controls'][key], value)
    for key, value in controls_reference(trace['fit_V'], trace).items():
        audit.compare(prefix + '/fit_controls/' + key, trace['fit_native_controls'][key], value)
    for key, value in map_statistics(trace['diagnostic_V']).items():
        audit.compare(prefix + '/V_stats/' + key, native['original_V_stats'][key], value)
    reconstructed = {}
    for band in BANDS:
        name = prefix + '/T' + str(band)
        row = native['primary'][f'T{band}']
        actual_w = block_reference(trace['diagnostic_V'], band)
        audit.compare(name + '/actual_W', row['actual_W'], actual_w, exact=True)
        expected = assessment(head_reference(actual_w, trace['native_head_fit'], itemwise=True), targets, cells)
        verify_assessment(audit, name, row, expected)
        # Reconstruct exact action/summary semantics again from archived emitted
        # fields after numerically verifying those fields against the reference H.
        emitted = assessment(row['local_logits'], targets, cells, row['role_fields'])
        verify_assessment(audit, name + '/emitted_field_policy', row, emitted)
        reconstructed[f'T{band}'] = emitted
        verify_records(audit, name, row, expected, band)
        audit.compare(name + '/batched_logits', native['batched_operator_logits'][band], expected['local_logits'])
        audit.compare(name + '/batched_fields', native['batched_operator_role_fields'][band], expected['role_fields'])
        for key, value in map_statistics(actual_w).items():
            audit.compare(name + '/stats/' + key, row['actual_W_stats'][key], value)
            audit.compare(name + '/stats_delta/' + key, row['actual_W_stat_changes_from_original'][key],
                          value - map_statistics(trace['diagnostic_V'])[key])
        control_values = controls_reference(actual_w, trace)
        for key, value in control_values.items():
            audit.compare(name + '/original_path/' + key, row['native_controls'][key], value)
            audit.compare(name + '/original_path_delta/' + key, row['control_changes_from_original'][key], value - baseline_controls[key])
        metadata = row['operator_metadata']
        audit.check(name + '/operator_count', len(metadata) == 256, actual=len(metadata), expected=256)
        for ordinal, value in enumerate(metadata):
            world, item = divmod(ordinal, 4)
            expected_pending = {'address': ADDRESSES[item], 'bandwidth': band, 'issued_at': 0, 'prospective_at': 1}
            expected_history = [{'address': ADDRESSES[item], 'bandwidth': band, 'issued_at': 0, 'completed_at': 1, 'driver': 'endogenous'}]
            audit.check(name + f'/operator/{ordinal}', value['world_index'] == world and value['item_index'] == item
                and value['pending'] == expected_pending and value['completed_history'] == expected_history
                and value['actual_completed_at'] == 1, pending=value['pending'], history=value['completed_history'])
        for control, permuted in (('address_permutation', False), ('synthetic_site_permutation', True)):
            control_name = prefix + '/' + control + f'/T{band}'
            retained = native['controls'][control][f'T{band}']
            permuted_w = actual_w.flatten(-2)[..., list(SITE_ROLL)].reshape_as(actual_w) if permuted else actual_w
            control_expected = assessment(head_reference(permuted_w, trace['native_head_fit'], itemwise=True), targets, cells)
            verify_assessment(audit, control_name, retained, control_expected)
            verify_records(audit, control_name, retained, control_expected, band, remapped=not permuted, synthetic=permuted)
            if permuted:
                audit.compare(control_name + '/actual_W', retained['actual_W'], permuted_w, exact=True)
                audit.compare(control_name + '/site_indices', retained['new_to_old_site_permutation'], SITE_ROLL, exact=True)
                audit.compare(control_name + '/carried_logits', retained['local_logits'], row['local_logits'].flatten(-2)[..., list(SITE_ROLL)].reshape_as(row['local_logits']))
                audit.compare(control_name + '/carried_fields', retained['role_fields'], row['role_fields'].flatten(-2)[..., list(SITE_ROLL)].reshape_as(row['role_fields']))
                for stat, value in map_statistics(permuted_w).items():
                    audit.compare(control_name + '/stats/' + stat, retained['actual_W_stats'][stat], value)
                    audit.compare(control_name + '/invariant/' + stat, value, map_statistics(actual_w)[stat])
                for stat in ('entropy', 'maximum', 'margin'):
                    audit.compare(control_name + '/invariant/' + stat, retained['field_stats'][stat], row['field_stats'][stat])
                for ordinal, record in enumerate(retained['native_records']):
                    world, item = divmod(ordinal, 4)
                    unique = row['tie_counts'][world, item] == 1
                    audit.compare(control_name + f'/unique_mask/{ordinal}', record['unique_maximum_action_checks'], unique, exact=True)
                    before_site = row['selected_cells'][world, item, :, 0] * 4 + row['selected_cells'][world, item, :, 1]
                    after_site = retained['selected_cells'][world, item, :, 0] * 4 + retained['selected_cells'][world, item, :, 1]
                    audit.compare(control_name + f'/unique_actions/{ordinal}', after_site[unique], ((before_site + 1) % 16)[unique], exact=True)
                    audit.check(control_name + f'/change/{ordinal}', record['actual_W_changed'] == (not torch.equal(permuted_w[world, item], actual_w[world, item])))
                for key, value in controls_reference(permuted_w, trace).items():
                    audit.compare(control_name + '/original_path/' + key, retained['native_controls'][key], value)
                    audit.compare(control_name + '/original_path_delta/' + key, retained['control_changes_from_original'][key], value - baseline_controls[key])
                audit.check(control_name + '/synthetic_provenance', 'synthetic' in retained['provenance'] and retained['fine_action_equivariance_on_ties_asserted'] is False)
            else:
                audit.compare(control_name + '/same_item_W', retained['original_item_actual_W'], actual_w, exact=True)
                audit.compare(control_name + '/address_indices', retained['old_to_new_address_indices'], ADDRESS_ROLL, exact=True)
    for key, band in (('sham_T1', 1), ('matched_exogenous_T2', 2), ('restore_after_T2', 1)):
        row = native['controls'][key]
        expected = reconstructed[f'T{band}']
        verify_assessment(audit, prefix + '/' + key, row, expected)
        verify_records(audit, prefix + '/' + key, row, expected)
        primary = native['primary'][f'T{band}']
        for field in ('local_logits', 'role_fields', 'selected_cells', 'tie_counts', 'coarse_codes'):
            audit.compare(prefix + '/' + key + '/exact_parity/' + field, row[field], primary[field], exact=True)
        for ordinal, record in enumerate(row['native_records']):
            _, item = divmod(ordinal, 4)
            event = {'address': ADDRESSES[item], 'bandwidth': 2, 'issued_at': 0, 'completed_at': 1, 'driver': 'exogenous' if key == 'matched_exogenous_T2' else 'endogenous'}
            if key == 'matched_exogenous_T2':
                audit.check(prefix + '/' + key + f'/provenance/{ordinal}', record['completed_history'] == [event], history=record['completed_history'])
            if key == 'restore_after_T2':
                audit.check(prefix + '/' + key + f'/provenance/{ordinal}', record['restored_after'] == [event] and record['restored_history'] == [] and record['snapshot_time'] == 0)
    raw = trace['raw_grid_diagnostic']
    raw_expected = assessment(head_reference(trace['diagnostic_raw_grid'], trace['raw_head_fit']), targets, cells)
    verify_assessment(audit, prefix + '/raw', raw, raw_expected)
    audit.compare(prefix + '/raw/input', raw['actual_raw_grid'], trace['diagnostic_raw_grid'], exact=True)
    reconstructed['raw_grid'] = assessment(raw['local_logits'], targets, cells, raw['role_fields'])
    indices = torch.randint(64, (256, 64), generator=torch.Generator().manual_seed(230000000 + seed))
    audit.compare(prefix + '/bootstrap/indices', trace['bootstrap_indices'], indices, exact=True)
    expected_summary = summarize_reference(seed, reconstructed, indices, cells)
    audit.tree(prefix + '/summary', summary, expected_summary)
    audit.tree(prefix + '/query_counts', native['query_counts'], expected_summary['query_counts'])
    audit.check(prefix + '/T2/exact_repeat_ties', bool((reconstructed['T2']['tie_counts'] >= 4).all()), minimum=int(reconstructed['T2']['tie_counts'].min()))
    audit.compare(prefix + '/T2/zero_margin', reconstructed['T2']['field_stats']['margin'], torch.zeros(64, 4, 2, dtype=torch.float64), exact=True)
    audit.compare(prefix + '/T4/uniform_field', reconstructed['T4']['role_fields'], torch.full((64, 4, 2, 4, 4), 1/16), exact=True)
    replay_input_boundary(audit, seed, trace)
    return expected_summary


def run(args, audit):
    registration = json.loads(args.registration.read_text())
    execution = json.loads(args.execution.read_text())
    output = args.output_dir
    completion = json.loads((output / 'completion.json').read_text())
    results = json.loads((output / 'results.json').read_text())
    log = [json.loads(line) for line in (output / 'first_execution.log').read_text().splitlines()]
    rel = lambda p: p.resolve().relative_to(ROOT).as_posix()
    audit.check('receipts/schema', registration['schema'] == 'spatial_native_task_v1_development_registration')
    audit.check('receipts/preexecution', registration['development_execution_started'] is False and registration['source_review_status'] == 'approved_for_development_only')
    audit.check('receipts/exclusive_claim', execution['status'] == 'claimed' and execution['single_attempt'] is True)
    audit.check('receipts/declared_paths', registration['output_dir'] == rel(output) == execution['output_dir'] and registration['execution_path'] == rel(args.execution))
    for name, receipt in (('execution', execution), ('completion', completion)):
        audit.check('receipts/' + name + '/registration', receipt['registration_sha256'] == digest(args.registration), expected_sha256=digest(args.registration), actual_sha256=receipt['registration_sha256'])
    audit.check('receipts/completion', completion['status'] == 'completed')
    envelope = registration['envelope']
    for key, expected in {'seeds': list(SEEDS), 'fit_worlds_per_seed': 128, 'diagnostic_worlds_per_seed': 64,
        'items_per_world': 4, 'operator_bandwidths': list(BANDS), 'full_batch_adam_steps': 500,
        'dtype': 'float32', 'device': 'cpu', 'threads': 4, 'bootstrap_draws': 256,
        'single_attempt': True, 'semantic_qualification_gate': None}.items():
        audit.check('envelope/' + key, envelope[key] == expected, actual=envelope[key], expected=expected)
    audit.check('envelope/adam', envelope['adam'] == {'lr': .003, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False, 'foreach': False})
    bindings = {**registration['source_sha256'], **registration['review_sha256'], **registration['checkpoint_sha256']}
    hash_bindings(audit, 'current_bound_bytes', bindings)
    for path in registration['review_sha256']:
        receipt = json.loads((ROOT / path).read_text())
        audit.check('review_receipt/exact_source_binding', receipt['approved_sha256'] == registration['source_sha256'] and receipt['status'] == 'approved'
                    and receipt['source_review_status'] == 'approved_for_development_only' and receipt['independent'] is True
                    and receipt['receipt_scope'] == 'spatial_native_task_v1_preexecution_source_review'
                    and isinstance(receipt['reviewer'], str) and bool(receipt['reviewer'].strip()))
    checkpoint_paths = {f'audits/bound_content/confirmation_v2/seed{seed}.pt' for seed in SEEDS} | {
        f'audits/sensory_task_adapter_v1/states-{seed}.pt.gz' for seed in SEEDS}
    audit.check('receipts/checkpoint_inventory', set(registration['checkpoint_sha256']) == checkpoint_paths,
                actual=sorted(registration['checkpoint_sha256']), expected=sorted(checkpoint_paths))
    for child, ancestor, name in ((execution['registration_commit'], registration['source_preservation_commit'], 'registration_after_source'),
                                 (args.results_commit, execution['registration_commit'], 'results_after_registration')):
        result = subprocess.run(['git', '-C', str(args.preservation_checkout), 'merge-base', '--is-ancestor', ancestor, child],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audit.check('preserved_commit_order/' + name, result.returncode == 0, ancestor=ancestor, descendant=child,
                    git_exit=result.returncode, git_error=result.stderr.decode(errors='replace'))
    committed_hashes(audit, args.preservation_checkout, registration['source_preservation_commit'],
                     {**registration['source_sha256'], **registration['review_sha256']}, 'preserved_reviewed_source')
    committed_hashes(audit, args.preservation_checkout, execution['registration_commit'],
                     {**registration['source_sha256'], **registration['review_sha256'], rel(args.registration): digest(args.registration)}, 'preserved_registration')
    result_hashes = {**completion['archive_sha256'], rel(output / 'results.json'): completion['results_sha256'],
        rel(output / 'first_execution.log'): completion['first_execution_log_sha256'],
        rel(output / 'completion.json'): digest(output / 'completion.json'), rel(args.execution): digest(args.execution)}
    hash_bindings(audit, 'actual_results_bytes', result_hashes)
    committed_hashes(audit, args.preservation_checkout, args.results_commit, result_hashes, 'preserved_first_results')
    expected_archives = {rel(output / f'states-{seed}.pt.gz') for seed in SEEDS}
    audit.check('archives/exact_inventory', set(completion['archive_sha256']) == expected_archives, actual=sorted(completion['archive_sha256']), expected=sorted(expected_archives))
    audit.check('archives/no_failure_or_retry_files', not (output / 'failure.json').exists() and not list(output.glob('failed-states-*')))
    starts = [e for e in log if e.get('kind') == 'execution_started']
    ends = [e for e in log if e.get('kind') == 'execution_completed']
    completed_seeds = [e for e in log if e.get('kind') == 'seed_completed']
    audit.check('execution/one_recorded_attempt', len(starts) == len(ends) == 1 and [e['seed'] for e in completed_seeds] == list(SEEDS), recorded_start_count=len(starts), recorded_end_count=len(ends))
    audit.check('execution/update_count', sum(e.get('kind') == 'fit_loss' for e in log) == 3000, recorded_updates=sum(e.get('kind') == 'fit_loss' for e in log))
    if starts:
        audit.check('execution/started_receipt', starts[0]['threads'] == 4 and starts[0]['development_only'] is True
                    and starts[0]['registration_sha256'] == digest(args.registration))
    audit.check('execution/no_recorded_failure', not any(e.get('status') == 'failed_single_attempt'
                or e.get('kind') == 'failed_seed_archive' for e in log))
    # Hash violations do not silently load or run unbound production source.
    if any(not c['passed'] for c in audit.checks if c['name'].startswith(('current_bound_bytes/', 'actual_results_bytes/', 'preserved_'))):
        raise ValueError('bound or preserved byte verification failed; refusing archive/Controller execution')
    models = {}
    for seed in SEEDS:
        with gzip.open(output / f'states-{seed}.pt.gz', 'rb') as stream:
            trace = torch.load(stream, weights_only=True, map_location='cpu')
        models[str(seed)] = verify_seed(audit, seed, trace, results['models'][str(seed)], log)
        print(json.dumps({'seed': seed, 'reconstructed_engineering_gates': models[str(seed)]['engineering_gates'],
                          'checks_so_far': len(audit.checks), 'errors_so_far': len(audit.errors)}), flush=True)
    counts = {key: sum(m['query_counts'][key] for m in models.values()) for key in models['1301']['query_counts']}
    audit.tree('results/full_ledger', results['query_counts'], counts)
    for key, expected in {'primary_item_operator_opportunities': 2304, 'all_engineering_feasibility_gates_pass': all(m['engineering_feasibility_pass'] for m in models.values()),
        'goal_achieved': False, 'presentation_qualified': False, 'attention_model_A_tested': False, 'new_reports': 0, 'development_only': True}.items():
        audit.check('results/' + key, results[key] == expected, actual=results[key], expected=expected)
    invariant_errors = list(audit.errors)
    audit.check('results/all_software_invariants_pass', results['all_software_invariants_pass'] is True and not invariant_errors,
                reported=results['all_software_invariants_pass'], reconstruction_errors_before_verdict=len(invariant_errors))
    return {'models': models, 'reconstructed_query_counts': counts,
            'reconstructed_primary_item_operator_opportunities': 2304,
            'reconstructed_engineering_feasibility_pass': all(m['engineering_feasibility_pass'] for m in models.values())}


def corrected_reference_provenance(args):
    """Read-only bindings before torch import; supplemental to unchanged checks."""
    required_commits = {
        'results_commit': '614e74ae74ccf1d553f6fc6e0ec318f057a94ed8',
        'first_review_commit': '432146ae8f1420b13bacdbb8f00518570d34b95f',
        'diagnosis_commit': '0f8cf0952d47dca65198a0c46505e3eb9ca3936f',
    }
    for name, expected in required_commits.items():
        if getattr(args, name) != expected:
            raise ValueError(f'wrong preserved {name}')
    if len(args.reference_commit) != 40 or any(c not in '0123456789abcdef' for c in args.reference_commit):
        raise ValueError('reference commit must be a full lowercase hexadecimal commit')
    bindings = []
    def bound(relative, commit, expected=None):
        path = ROOT / relative
        actual = digest(path)
        expected = actual if expected is None else expected
        process = subprocess.Popen(['git', '-C', str(args.preservation_checkout), 'show', f'{commit}:{relative}'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        hasher = hashlib.sha256()
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b''):
            hasher.update(chunk)
        error = process.stderr.read().decode(errors='replace')
        status = process.wait()
        row = {'path': relative, 'commit': commit, 'expected_sha256': expected,
            'actual_sha256': actual, 'committed_sha256': hasher.hexdigest(), 'git_exit': status, 'git_error': error}
        bindings.append(row)
        if status or actual != expected or hasher.hexdigest() != expected:
            raise ValueError(f'corrected-reference preservation binding failed: {relative}')
    for relative in ('scripts/verify_spatial_native_task_v1_corrected.py',
                     'audits/spatial_native_task_v1/numerical_corrected_review/design.md'):
        bound(relative, args.reference_commit)
    for child, ancestor in ((args.first_review_commit, args.results_commit),
                            (args.diagnosis_commit, args.first_review_commit),
                            (args.reference_commit, args.diagnosis_commit)):
        result = subprocess.run(['git', '-C', str(args.preservation_checkout), 'merge-base', '--is-ancestor', ancestor, child],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise ValueError(f'corrected-reference commit order failed: {ancestor} -> {child}')
    review_dir = 'audits/spatial_native_task_v1/numerical_review/'
    bound(review_dir + 'metadata_manifest.json', args.first_review_commit)
    manifest = json.loads((ROOT / (review_dir + 'metadata_manifest.json')).read_text())
    bound(manifest['published_path'], args.first_review_commit, manifest['compressed_sha256'])
    raw = gzip.decompress((ROOT / manifest['published_path']).read_bytes())
    first_raw_hash = hashlib.sha256(raw).hexdigest()
    if first_raw_hash != manifest['raw_sha256'] or first_raw_hash != '6010154b9502daef7f223ac3da4ac9c7854c4c9bc15436b8d987d9a5c93dc338':
        raise ValueError('first failed metadata is not byte-identical')
    first = json.loads(raw)
    if first['status'] != 'failed_first_verification' or first['check_count'] != 206070 or first['error_count'] != 106:
        raise ValueError('unexpected preserved first numerical findings')
    for relative in (review_dir + 'design.md', review_dir + 'review.md', review_dir + 'first_run.log'):
        bound(relative, args.first_review_commit)
    bound('scripts/verify_spatial_native_task_v1.py', args.first_review_commit, first['verifier_sha256'])
    diagnosis_dir = 'audits/spatial_native_task_v1/numeric_diagnosis/'
    diagnosis_metadata_hash = 'f89788f885e7b3b2f477624dbc657147e6f6460edc9b8c0e4f590b40a842a3e1'
    bound(diagnosis_dir + 'metadata.json', args.diagnosis_commit, diagnosis_metadata_hash)
    diagnosis = json.loads((ROOT / (diagnosis_dir + 'metadata.json')).read_text())
    if (diagnosis['status'] != 'diagnosis_complete_all_first_failures_explained'
        or diagnosis['explained_count'] != 106 or diagnosis['unresolved']
        or diagnosis['first_review_commit'] != args.first_review_commit
        or diagnosis['results_commit'] != args.results_commit
        or diagnosis['first_findings']['raw_sha256'] != first_raw_hash
        or len(diagnosis['failed_check_diagnoses']) != 106
        or not all(row['explained'] for row in diagnosis['failed_check_diagnoses'])
        or sorted(row['name'] for row in diagnosis['failed_check_diagnoses']) != sorted(first['errors'])):
        raise ValueError('diagnosis does not bind and account for all first failures')
    for relative in ('scripts/diagnose_spatial_native_numeric_reference_v1.py', diagnosis_dir + 'design.md',
                     diagnosis_dir + 'review.md', diagnosis_dir + 'first_run.log'):
        bound(relative, args.diagnosis_commit)
    provenance = {'reference_commit': args.reference_commit, 'first_review_commit': args.first_review_commit,
        'diagnosis_commit': args.diagnosis_commit, 'original_results_commit': args.results_commit,
        'first_failed_raw_metadata_sha256': first_raw_hash, 'diagnosis_metadata_sha256': diagnosis_metadata_hash,
        'original_first_status': first['status'], 'original_first_check_count': first['check_count'],
        'original_first_error_count': first['error_count'], 'diagnosis_status': diagnosis['status'],
        'bindings': bindings, 'supplemental_to_unchanged_core_check_inventory': True}
    return provenance, [row['name'] for row in first['checks']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preservation-checkout', type=Path, required=True)
    parser.add_argument('--results-commit', required=True, help='commit preserving first actual run/results before this verification')
    parser.add_argument('--reference-commit', required=True, help='full commit preserving this corrected program and design before execution')
    parser.add_argument('--first-review-commit', required=True, help='preserved original failed verification, never relabelled')
    parser.add_argument('--diagnosis-commit', required=True, help='preserved arithmetic diagnosis explaining the first failures')
    parser.add_argument('--report', type=Path, required=True, help='new exclusive JSON report; never overwrite first findings')
    parser.add_argument('--ack-first-results-preserved', action='store_true', required=True)
    args = parser.parse_args()
    # Exclusive audit result reservation occurs before torch/archive import.
    with args.report.open('x', encoding='utf-8') as stream:
        audit = Audit()
        report = {'schema': 'spatial_native_task_v1_corrected_reference_numerical_review',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'verifier_sha256': digest(__file__),
            'reviewer': '/root/spatial_native_numerical_execution_review', 'separate_task_allocation': True,
            'independent_model_family': False, 'shared_model_and_inherited_context': True,
            'human_endorsement': False, 'reference_arithmetic_corrected': True, 'study_rerun': False,
            'original_first_verification_status': 'failed_first_verification', 'original_first_findings_unchanged': True,
            'reference_method_changes': ['Native/action/synthetic/address head reconstruction uses itemwise independent matmul plus separate bias.',
                'Fit initial/final logits and CE and raw head reconstruction preserve batched matmul plus separate bias.',
                'Original control reconstruction uses independent flattened F.linear equations at all four archived linear stages.',
                'Float64 explicit exponent/sum role normalization, all core checks, exact policies, tolerances and gates are unchanged.'],
            'no_fit_replay': True, 'no_world_generation': True, 'development_only': True,
            'task_content_semantics_only': True, 'presentation_qualified': False, 'goal_achieved': False,
            'interpretation_classification': 'development native role/task-content semantics; no presentation/qualia qualification',
            'limitations': ['Final weights/logits and recorded 500-update loss traces do not prove optimizer trajectory without replaying fits.',
                'Before/after/checkpoint equality establishes retained frozen weights, not absence of transient in-memory changes.',
                'Single committed claim/log inventory and exclusive source paths support one recorded attempt; they cannot rule out unrecorded external work.',
                'Intervals are conditional world bootstraps on shared deterministic layout support, not model-family/human/generalization uncertainty.',
                'Controller reenactment uses production consumer only for input isolation; task math and summary reconstruction use independent reference equations.',
                'Corrected reference arithmetic addresses preserved CPU dispatch/shape discrepancies; the original failed first verification remains preserved.']}
        try:
            provenance, original_check_inventory = corrected_reference_provenance(args)
            report['corrected_reference_provenance'] = provenance
            global torch
            import torch
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            with torch.no_grad():
                report.update(run(args, audit))
            inventory_matches = [row['name'] for row in audit.checks] == original_check_inventory
            report['core_check_inventory_matches_first_review'] = inventory_matches
            if not inventory_matches:
                raise ValueError('corrected reference changed the original core check names/count/order')
        except BaseException:
            report['exception'] = traceback.format_exc()
            audit.errors.append('verification_exception')
        report.update(status='passed_corrected_reference_revalidation' if not audit.errors else 'failed_first_corrected_reference_verification',
                      error_count=len(audit.errors), errors=audit.errors, checks=audit.checks,
                      check_count=len(audit.checks), study_retry_or_repair_performed=False)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        print(json.dumps({key: report[key] for key in ('status', 'check_count', 'error_count', 'errors')}), flush=True)
    if audit.errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
