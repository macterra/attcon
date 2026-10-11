"""Draft neutral views of frozen grounded source archives; no Reporter or writer.

Loading archives and constructing records are separate operations. This module
has no CLI, registration, production inference, decoder fitting or output writer.
A future reviewed/frozen exporter must authorize record construction explicitly.
"""
import copy
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

PHYSICAL_ROOT = Path('audits/grounded_physical_information_v1')
SEEDS = (1301, 1311, 1321)
SELECTED_WORLD_INDICES = tuple(range(16))
VARIANTS = ('natural', 'sham', 'memory_readdressed', 'delivery_patched',
            'memory_restored', 'delivery_restored')
FORECAST_KEYS = ('delivery', 'current', 'refresh', 'after', 'risk_hidden')
LOCATION_NAMES = ('l0', 'l1', 'l2', 'l3')
COMMAND_NAMES = ('k0', 'k1', 'k2', 'k3')
SOURCE_ATOL, SOURCE_RTOL, DECODE_ROUNDOFF = 1e-6, 1e-5, 1e-12
SOURCE_SEMANTICS = {'stored_content': 'Current memory rows describe represented evidence. They may differ from past observation placement.', 'history': 'Native input stores only each command row’s latest observed delivery or an unknown-zero row; it is not a full chronological history and does not certify current memory placement.', 'forecasts': 'Supplied represented estimates, including possible export overrides; not observed future events.', 'logits': 'Uncalibrated orientation decision coordinates; positive selects vertical, otherwise horizontal.', 'success': 'Estimated correctness of a fixed orientation decision; not an observed outcome.', 'after_composition': 'delivery*refresh+(1-delivery)*current at each command/location.', 'external_readout': 'Linear estimates are retained without clipping and may lie outside physical ranges; no per-item guarantees.'}


@dataclass
class SourceBundle:
    """Whitelisted source-only tensors; no truth, pixels, target or label fields."""
    visual_seed: int
    original_sensory: torch.Tensor
    native_history: torch.Tensor
    commands: torch.Tensor
    delivered: torch.Tensor
    last_step: torch.Tensor
    sensory_by_variant: dict
    forecasts_by_variant: dict
    physical_by_variant: dict
    provenance: dict
    physical_error_summary: dict
    risk_decoder_parameters: dict
    decode_permutation_error_bounds: dict


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _forecast(source):
    # Explicit extraction is deliberate: never forward the whole archive object.
    return {key: source[key].detach().cpu().clone() for key in FORECAST_KEYS}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_bundle(visual_seed, root=PHYSICAL_ROOT):
    """Future read-only loading; consumes no old per-world grounded-study results.

    All natural/intervention records are from the physical study's evaluation
    worlds; model identity/weights provenance points to its frozen grounded source.
    Mixed archives can contain evaluator data, but only named whitelisted keys are
    extracted. No selected world is accessed or exported by importing this module.
    """
    _require(visual_seed in SEEDS, 'unknown frozen model')
    root = Path(root)
    manifest_path, completion_path = root/'manifest.json', root/'completion.json'
    manifest = json.loads(manifest_path.read_text())
    _require(manifest['permutation_source_tolerance'] == {'atol': SOURCE_ATOL, 'rtol': SOURCE_RTOL}
             and manifest['linear_decoder_roundoff_floor'] == DECODE_ROUNDOFF,
             'registered source numerical rules differ')
    completion = json.loads(completion_path.read_text())
    verification = json.loads((root/'verification.json').read_text())
    _require(completion['status'] == 'single_registered_attempt_completed', 'study incomplete')
    _require(digest(manifest_path) == completion['manifest_sha256'], 'manifest hash mismatch')
    _require(digest(root/'results.json') == completion['results_sha256'], 'results hash mismatch')
    _require(verification['status'] == 'exact_collection_features_decoder_selection_interventions_results_replay'
             and verification['all_files_verified'], 'exact replay receipt required')
    _require(verification['manifest_sha256'] == completion['manifest_sha256']
             and verification['results_sha256'] == completion['results_sha256'], 'replay provenance mismatch')
    read_hashes = {}

    def read(relative):
        path = root/f'model-{visual_seed}'/(relative+'.pt.gz')
        expected = completion['archive_files'][str(path)]
        _require(digest(path) == expected['sha256'], f'archive hash mismatch: {path}')
        read_hashes[str(path)] = expected['sha256']
        with gzip.open(path, 'rb') as stream:
            return torch.load(stream, weights_only=True)

    snapshot = read('split-2/snapshot')
    actual = read('split-2/decoded')
    frozen_forecasts = read('split-2/forecasts')['structured']
    evaluation_physical = read('evaluation_predictions')['risk_hidden']['physical']
    trace = read('interventions')
    decoder = read('decoders')['risk_hidden']
    _require(decoder['layout'] == 'shared_pooled_seen_rows' and decoder['decoder_count'] == 1
             and decoder['feature_dimensions'] == 32,
             'only shared local risk decoder permitted')
    _require(torch.equal(snapshot['sensory'], trace['original_sensory']), 'snapshot/intervention memory mismatch')
    for key in FORECAST_KEYS:
        _require(torch.equal(frozen_forecasts[key], trace['original'][key]), f'forecast archive mismatch: {key}')
    _require(torch.equal(evaluation_physical, trace['original_local_decoded']['risk_hidden']['physical']),
             'readout archive mismatch')
    # Only receipt/model IDs and selected aggregate error coordinates leave JSON.
    results = json.loads((root/'results.json').read_text())
    pooled = results['models'][str(visual_seed)]['readouts']['risk_hidden']['assessment']['all_seen']['pooled']
    error_summary = {key: copy.deepcopy(pooled[key]) for key in ('mae', 'rmse')}
    error_summary['scope'] = 'all seen evaluation rows; conditional fixed-model world-bootstrap intervals; not per-item error bounds'
    model_path = f'audits/grounded_attention_outcomes_v1/model-{visual_seed}/instrument-structured.pt.gz'
    _require(model_path in manifest['frozen_input_sha256'], 'grounded source identity missing')
    _require(digest(model_path) == manifest['frozen_input_sha256'][model_path], 'grounded model input hash mismatch')
    restored_patch = _forecast(trace['patched_forecasts'])
    restored_patch['delivery'] = trace['restored_patch_delivery'].clone()
    restored_patch['after'] = trace['restored_patch_after'].clone()
    sensory = {'natural': snapshot['sensory'].clone(), 'sham': snapshot['sensory'].clone(),
               'memory_readdressed': trace['readdressed_sensory'].clone(),
               'delivery_patched': snapshot['sensory'].clone(),
               'memory_restored': trace['restored_sensory'].clone(),
               'delivery_restored': snapshot['sensory'].clone()}
    forecasts = {'natural': _forecast(frozen_forecasts), 'sham': _forecast(frozen_forecasts),
                 'memory_readdressed': _forecast(trace['readdressed']),
                 'delivery_patched': _forecast(trace['patched_forecasts']),
                 'memory_restored': _forecast(trace['restored']), 'delivery_restored': restored_patch}
    physical = {'natural': evaluation_physical.clone(), 'sham': evaluation_physical.clone(),
                'memory_readdressed': trace['readdressed_local_decoded']['risk_hidden']['physical'].clone(),
                'delivery_patched': trace['patch_local_decoded']['risk_hidden']['physical'].clone(),
                'memory_restored': trace['restored_local_decoded']['risk_hidden']['physical'].clone(),
                'delivery_restored': trace['patch_local_decoded']['risk_hidden']['physical'].clone()}
    bundle = SourceBundle(visual_seed, snapshot['sensory'].clone(), snapshot['history'].clone(), actual['commands'].clone(),
                          actual['delivered'].clone(), snapshot['last_step'].clone(), sensory,
                          forecasts, physical,
                          {'manifest_sha256': completion['manifest_sha256'], 'archive_sha256': read_hashes,
                           'grounded_model_archive': model_path,
                           'grounded_model_sha256': manifest['frozen_input_sha256'][model_path],
                           'risk_decoder_layout': decoder['layout'],
                           'risk_decoder_feature_dimensions': decoder['feature_dimensions']}, error_summary,
                          {key: decoder['fits'][0][key].detach().cpu().clone()
                           for key in ('feature_scale', 'weights', 'target_scale')},
                          {key: trace['decode_permutation_error_bounds']['risk_hidden'][key].detach().cpu().clone()
                           for key in ('absolute_feature_difference', 'absolute_output_difference', 'propagated_bound')})
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    """Validate source fidelity independently of any prose or world truth."""
    original = bundle.original_sensory
    n = len(original)
    _require(original.shape == (n, 1, 4, 66) and n > 0, 'bad memory shape')
    _require(bundle.native_history.shape == (n, 1, 4, 4), 'bad native route-history shape')
    native = bundle.native_history[:, 0]
    _require(bool(((native == 0) | (native == 1)).all()) and bool((native.sum(-1) <= 1).all()),
             'native history rows must be unknown-zero or one-hot')
    _require(bundle.commands.shape == (n, 8) and bundle.delivered.shape == (n, 8), 'eight actual events required')
    _require(bundle.last_step.shape == (n, 4), 'bad last-delivery indices')
    for value in (bundle.commands, bundle.delivered, bundle.last_step):
        _require(value.dtype == torch.long, 'event indices must be integer')
    for value in (bundle.commands, bundle.delivered):
        _require(bool(((value >= 0) & (value < 4)).all()), 'event address out of range')
    _require(set(bundle.sensory_by_variant) == set(VARIANTS), 'incomplete memory views')
    _require(set(bundle.forecasts_by_variant) == set(VARIANTS), 'incomplete forecast views')
    _require(set(bundle.physical_by_variant) == set(VARIANTS), 'incomplete readout views')
    _require(torch.equal(bundle.sensory_by_variant['natural'], original), 'natural memory mismatch')
    for row in range(n):
        for command in range(4):
            events = torch.where(bundle.commands[row] == command)[0]
            expected = torch.zeros(4, dtype=native.dtype)
            if len(events):
                expected[int(bundle.delivered[row, int(events[-1])])] = 1
            _require(torch.equal(native[row, command], expected), 'native/latest command delivery mismatch')
        for loc in range(4):
            events = torch.where(bundle.delivered[row] == loc)[0]
            last = int(events[-1]) if len(events) else -1
            _require(int(bundle.last_step[row, loc]) == last, 'past/latest delivery mismatch')
            _require(bool(original[row, 0, loc, 65]) == bool(len(events)), 'natural observed/history mismatch')
    expected_shapes = {'delivery': (n, 1, 4, 4), 'current': (n, 1, 4),
                       'refresh': (n, 1, 4), 'after': (n, 1, 4, 4), 'risk_hidden': (n, 1, 4, 32)}
    for variant in VARIANTS:
        memory, p = bundle.sensory_by_variant[variant], bundle.forecasts_by_variant[variant]
        _require(memory.shape == original.shape and bool(torch.isfinite(memory).all()), 'invalid memory view')
        _require(bool(((memory[..., 65] == 0) | (memory[..., 65] == 1)).all()), 'observed flag must be binary')
        unseen = memory[..., 65] == 0
        _require(not bool(memory[..., :65][unseen].any()), 'unseen memory must be zero')
        for key, shape in expected_shapes.items():
            _require(p[key].shape == shape and bool(torch.isfinite(p[key]).all()), f'invalid {key}')
            if key != 'risk_hidden':
                _require(bool(((p[key] >= 0) & (p[key] <= 1)).all()), 'forecast outside probability range')
        _require(torch.allclose(p['delivery'].sum(-1), torch.ones(n, 1, 4, dtype=p['delivery'].dtype),
                                atol=1e-12, rtol=0), 'delivery rows must sum to one')
        composed = compose_after(p['delivery'], p['current'], p['refresh'])
        _require(torch.equal(composed, p['after']), 'post-command forecast does not match exact composition')
        readout = bundle.physical_by_variant[variant]
        _require(readout.shape == (n, 4, 2) and bool(torch.isfinite(readout).all()), 'invalid external readout')
    _require(torch.equal(bundle.sensory_by_variant['memory_readdressed'], original.roll(1, -2)), 'wrong native memory permutation')
    for variant in ('sham', 'delivery_patched', 'memory_restored', 'delivery_restored'):
        _require(torch.equal(bundle.sensory_by_variant[variant], original), 'unchanged/restored memory differs')
    base = bundle.forecasts_by_variant['natural']
    for variant in ('sham', 'memory_restored', 'delivery_restored'):
        _require(all(torch.equal(base[k], bundle.forecasts_by_variant[variant][k]) for k in FORECAST_KEYS),
                 'exact forecast restoration/sham mismatch')
        _require(torch.equal(bundle.physical_by_variant[variant], bundle.physical_by_variant['natural']),
                 'exact readout restoration/sham mismatch')
    patched = bundle.forecasts_by_variant['delivery_patched']
    _require(torch.equal(patched['delivery'], base['delivery'].roll(1, -2)), 'wrong forecast-row patch')
    for key in ('risk_hidden', 'current', 'refresh'):
        _require(torch.equal(patched[key], base[key]), 'forecast patch changed upstream state')
    _require(torch.equal(bundle.physical_by_variant['delivery_patched'], bundle.physical_by_variant['natural']),
             'forecast patch changed physical readout')
    moved = bundle.forecasts_by_variant['memory_readdressed']
    _require(torch.equal(moved['delivery'], base['delivery']), 'memory readdressing changed route')
    for key in ('risk_hidden', 'current', 'refresh'):
        axis = -2 if key == 'risk_hidden' else -1
        _require(torch.allclose(moved[key], base[key].roll(1, axis), atol=SOURCE_ATOL, rtol=SOURCE_RTOL),
                 'memory readdressing changed item-wise risk outside frozen numerical tolerance')
    # Archive-only decoder coordinates and registered propagation envelope.
    params = bundle.risk_decoder_parameters
    _require(isinstance(params, dict) and set(params) == {'feature_scale', 'weights', 'target_scale'},
             'unexpected source-only decoder parameters')
    for key, shape in (('feature_scale', (32,)), ('weights', (32, 3)), ('target_scale', (2,))):
        value = params[key]
        _require(isinstance(value, torch.Tensor) and value.shape == shape
                 and bool(torch.isfinite(value).all()), 'invalid source-only decoder parameter')
    _require(bool((params['feature_scale'] > 0).all()) and bool((params['target_scale'] > 0).all()),
             'decoder scales must be positive')
    envelope = bundle.decode_permutation_error_bounds
    expected_keys = {'absolute_feature_difference', 'absolute_output_difference', 'propagated_bound'}
    _require(isinstance(envelope, dict) and set(envelope) == expected_keys, 'unexpected decoder bound fields')
    actual_feature_difference = (moved['risk_hidden'][:, 0].double()-base['risk_hidden'][:, 0].roll(1, 1).double()).abs()
    actual_output_difference = (bundle.physical_by_variant['memory_readdressed']
                                -bundle.physical_by_variant['natural'].roll(1, 1)).abs()
    recomputed_bound = ((actual_feature_difference/params['feature_scale'])
                        @params['weights'][:, :2].abs()*params['target_scale'])
    for key, expected in (('absolute_feature_difference', actual_feature_difference),
                          ('absolute_output_difference', actual_output_difference),
                          ('propagated_bound', recomputed_bound)):
        value = envelope[key]
        _require(isinstance(value, torch.Tensor) and value.shape == expected.shape
                 and bool(torch.isfinite(value).all()) and bool((value >= 0).all()), 'invalid decoder bound tensor')
        _require(torch.equal(value, expected), 'archived decoder discrepancy/bound differs from actual source/parameters')
    _require(bool((actual_output_difference <= recomputed_bound+DECODE_ROUNDOFF).all()),
             'readdressed external readout exceeds registered feature-roundoff envelope')


def compose_after(delivery, current, refresh):
    return delivery*refresh.unsqueeze(-2)+(1-delivery)*current.unsqueeze(-2)


def selection_indices(bundle):
    """Fixed opportunity rule; all four locations retained for each world."""
    _require(len(bundle.original_sensory) >= 16, 'sixteen preselected worlds required')
    return SELECTED_WORLD_INDICES


def _float(value):
    _require(bool(torch.isfinite(value)), 'nonfinite exported value')
    return float(value)


def record(bundle, row, variant='natural'):
    """Return Reporter input separately from source-aware audit provenance.

    This is a factual draft record, not a Reporter instruction or serialized output.
    Intervention IDs, physical-truth data and archive paths are audit-only.
    """
    _require(variant in VARIANTS, 'unknown source view')
    selection_indices(bundle)
    _require(isinstance(row, int) and row in SELECTED_WORLD_INDICES and row < len(bundle.original_sensory),
             'world index outside fixed first-sixteen opportunity rule')
    validate_bundle(bundle)
    memory = bundle.sensory_by_variant[variant][row, 0]
    p = {k: v[row, 0] for k, v in bundle.forecasts_by_variant[variant].items()}
    physical = bundle.physical_by_variant[variant][row]
    donor = [(loc-1) % 4 for loc in range(4)] if variant == 'memory_readdressed' else list(range(4))
    locations = []
    for loc in range(4):
        observed = bool(memory[loc, 65])
        origin = donor[loc]
        raw = _float(memory[loc, 64]) if observed else None
        # Missing observation is distinct from the fixed unseen query guess.
        content = {'status': 'sample_stored' if observed else 'no_sample_stored',
                   'raw_orientation_logit': raw,
                   'logit_sign': ('positive' if raw > 0 else 'negative' if raw < 0 else 'zero') if observed else None,
                   'stored_decision': ('vertical' if raw > 0 else 'horizontal') if observed else None,
                   'evidence_item': f'e{origin}' if observed else None,
                   'decision_if_no_sample': None if observed else {'value': 'horizontal', 'basis': 'fixed default guess'}}
        locations.append({'location': LOCATION_NAMES[loc], 'observed': observed, 'stored_evidence': content,
                          'estimated_current_decision_success': _float(p['current'][loc]),
                          'estimated_fresh_glimpse_decision_success': _float(p['refresh'][loc]),
                          'risk_feature_source': {'model': f'structured-{bundle.visual_seed}',
                                                  'field': 'risk_hidden', 'dimensions': 32,
                                                  'meaning': 'local learned decision-outcome representation; coordinates not interpreted'},
                          'external_image_statistic_readouts': {
                              'status': 'estimated_for_stored_sample' if observed else 'undefined_without_stored_sample',
                              'source': 'external shared linear readout of frozen risk_hidden',
                              'units': 'normalized image-value units',
                              'image_value_std_estimate': _float(physical[loc, 0]) if observed else None,
                              'nearest_neighbor_rms_estimate': _float(physical[loc, 1]) if observed else None,
                              'interpretation': 'linear estimates of objective stored-image statistics'}})
    commands = [{'command': COMMAND_NAMES[c],
                 'predicted_delivery_distribution': {LOCATION_NAMES[l]: _float(p['delivery'][c, l]) for l in range(4)},
                 'estimated_decision_success_after_command': {LOCATION_NAMES[l]: _float(p['after'][c, l]) for l in range(4)}}
                for c in range(4)]
    reporter = {'schema_version': 'grounded_report_interface_v1_draft', 'snapshot_after_commands': 8,
                'frozen_sources': {'visual_model': f'visual-{bundle.visual_seed}',
                                   'outcome_model': f'structured-{bundle.visual_seed}',
                                   'external_readout': f'risk-hidden-ridge-{bundle.visual_seed}'},
                'source_semantics': copy.deepcopy(SOURCE_SEMANTICS),
                'native_route_history': [{'command': COMMAND_NAMES[c],
                                          'status': 'observed_delivery' if bool(bundle.native_history[row, 0, c].any()) else 'no_delivery_observed',
                                          'last_observed_delivery_one_hot': {LOCATION_NAMES[l]: int(bundle.native_history[row, 0, c, l]) for l in range(4)}}
                                         for c in range(4)],
                'locations': locations, 'command_forecasts': commands,
                'external_readout_evaluation_summary': _error_summary(bundle.physical_error_summary)}
    audit = {'world_index': row, 'source_variant': variant,
             'synthetic_patched_source_not_natural_state': variant in ('memory_readdressed', 'delivery_patched'),
             'current_memory_precedence': True, 'history_retains_actual_past': True,
             'forecast_origin': 'export_only_delivery_patch' if variant in ('delivery_patched', 'delivery_restored')
                                else 'frozen_network_inference',
             'evidence_original_location_by_current_slot': donor,
             'evidence_acquisition_event_by_current_slot': [int(bundle.last_step[row, origin])+1
                                                           if int(bundle.last_step[row, origin]) >= 0 else None for origin in donor],
             'external_observed_history_instrumentation': {
                 'native_model_input': False, 'reporter_context': False,
                 'chronological_events': [{'event': step+1, 'command': COMMAND_NAMES[int(bundle.commands[row, step])],
                                           'delivered_location': LOCATION_NAMES[int(bundle.delivered[row, step])]} for step in range(8)]},
             'risk_hidden_tensor_pointer': {'variant': variant, 'row': row, 'snapshot_axis': 0},
             'frozen_provenance': copy.deepcopy(bundle.provenance),
             'unseen_physical_decoder_outputs_suppressed': True,
             'decoder_parameters_and_bounds_reporter_context': False,
             'source_numeric_rules': {'atol': SOURCE_ATOL, 'rtol': SOURCE_RTOL, 'decode_roundoff_floor': DECODE_ROUNDOFF},
             'interpretation_limits': 'Task-success forecasts and external physical-statistic estimates do not qualify appearance, felt confidence or qualia.',
             'reporter_run': False, 'appearance_qualified': False, 'goal_achieved': False}
    return {'reporter_input': reporter, 'audit_provenance': audit}


def _error_summary(source):
    # Only aggregate numeric MAE/RMSE coordinates, no evaluator strata or prose.
    keys = ('value', 'conditional_world_bootstrap_95', 'valid_bootstrap_counts')
    result = {metric: {key: copy.deepcopy(source[metric][key]) for key in keys}
              for metric in ('mae', 'rmse')}
    for metric in ('mae', 'rmse'):
        data = result[metric]
        _require(len(data['value']) == 2 and len(data['conditional_world_bootstrap_95']) == 2
                 and len(data['valid_bootstrap_counts']) == 2, 'two physical target errors required')
        numbers = list(data['value'])
        for bounds in data['conditional_world_bootstrap_95']:
            if bounds is not None:
                _require(len(bounds) == 2, 'invalid error interval')
                numbers.extend(bounds)
        _require(all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))
                     for v in numbers), 'error summary must contain finite numeric coordinates or missing values')
        _require(all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 256
                     for v in data['valid_bootstrap_counts']), 'invalid error bootstrap counts')
    result['scope'] = 'All seen evaluation rows; conditional fixed-model world-bootstrap intervals, not per-item error bounds.'
    return result


METADATA_KEYS = {'schema_version', 'snapshot_after_commands', 'frozen_sources', 'source_semantics',
                 'native_route_history', 'external_readout_evaluation_summary'}
LOCATION_KEYS = {'location', 'observed', 'stored_evidence', 'estimated_current_decision_success',
                 'estimated_fresh_glimpse_decision_success', 'risk_feature_source', 'external_image_statistic_readouts'}
COMMAND_KEYS = {'command', 'predicted_delivery_distribution', 'estimated_decision_success_after_command'}
TABLE_SCHEMAS = {
    'stored_evidence': ('location', {'location', 'observed', 'stored_evidence'}),
    'risk_forecasts': ('location', {'location', 'estimated_current_decision_success', 'estimated_fresh_glimpse_decision_success'}),
    'external_readouts': ('location', {'location', 'risk_feature_source', 'external_image_statistic_readouts'}),
    'delivery_forecasts': ('command', {'command', 'predicted_delivery_distribution'}),
    'post_command_forecasts': ('command', {'command', 'estimated_decision_success_after_command'})}


def _keys(value, expected, name):
    _require(isinstance(value, dict) and set(value) == set(expected), 'unexpected fields in '+name)


def _address_rows(rows, address, fields):
    expected = set(LOCATION_NAMES if address == 'location' else COMMAND_NAMES)
    _require(isinstance(rows, list) and len(rows) == 4, 'exactly four address rows required')
    for row in rows:
        _keys(row, fields, 'address row')
        _require(isinstance(row[address], str) and row[address] in expected, 'noncanonical address')
    _require(len({row[address] for row in rows}) == 4, 'duplicate/missing address')
    return {row[address]: row for row in rows}


def _number(value, missing=False, probability=False):
    if missing and value is None:
        return
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
             'finite numeric value required')
    if probability:
        _require(0 <= value <= 1, 'probability outside range')


def _validate_metadata(value, collection_keys):
    _keys(value, METADATA_KEYS | set(collection_keys), 'top metadata')
    _require(value['schema_version'] == 'grounded_report_interface_v1_draft'
             and value['snapshot_after_commands'] == 8, 'unknown source schema/snapshot')
    models = value['frozen_sources']
    _keys(models, {'visual_model', 'outcome_model', 'external_readout'}, 'frozen sources')
    _require(any(models == {'visual_model': f'visual-{seed}', 'outcome_model': f'structured-{seed}',
                            'external_readout': f'risk-hidden-ridge-{seed}'} for seed in SEEDS), 'unknown frozen source IDs')
    _keys(value['source_semantics'], {'stored_content', 'history', 'forecasts', 'logits', 'success',
                                     'after_composition', 'external_readout'}, 'source semantics')
    _require(value['source_semantics'] == SOURCE_SEMANTICS, 'source legend differs from fixed task semantics')
    history = _address_rows(value['native_route_history'], 'command',
                            {'command', 'status', 'last_observed_delivery_one_hot'})
    for row in history.values():
        mapping = row['last_observed_delivery_one_hot']
        _keys(mapping, LOCATION_NAMES, 'native delivery row')
        _require(all(type(v) is int and v in (0, 1) for v in mapping.values()) and sum(mapping.values()) <= 1,
                 'native row must be unknown-zero or one-hot')
        _require(row['status'] == ('observed_delivery' if sum(mapping.values()) else 'no_delivery_observed'),
                 'native history status mismatch')
    summary = value['external_readout_evaluation_summary']
    _keys(summary, {'mae', 'rmse', 'scope'}, 'external error summary')
    for metric in ('mae', 'rmse'):
        _keys(summary[metric], {'value', 'conditional_world_bootstrap_95', 'valid_bootstrap_counts'}, 'error metric')
    _require(summary == _error_summary(summary), 'noncanonical error scope/values')


def validate_reporter_input(value):
    """Strict schema boundary for helpers; reject rather than drop extra fields."""
    _validate_metadata(value, {'locations', 'command_forecasts'})
    nodes = _address_rows(value['locations'], 'location', LOCATION_KEYS)
    commands = _address_rows(value['command_forecasts'], 'command', COMMAND_KEYS)
    for node in nodes.values():
        _require(type(node['observed']) is bool, 'observed must be boolean')
        evidence = node['stored_evidence']
        _keys(evidence, {'status', 'raw_orientation_logit', 'logit_sign', 'stored_decision',
                         'evidence_item', 'decision_if_no_sample'}, 'stored evidence')
        raw = evidence['raw_orientation_logit']
        if node['observed']:
            _number(raw)
            _require(evidence['status'] == 'sample_stored' and evidence['evidence_item'] in {f'e{i}' for i in range(4)}
                     and evidence['decision_if_no_sample'] is None, 'stored evidence status mismatch')
            _require(evidence['logit_sign'] == ('positive' if raw > 0 else 'negative' if raw < 0 else 'zero')
                     and evidence['stored_decision'] == ('vertical' if raw > 0 else 'horizontal'), 'stored sign/decision mismatch')
        else:
            _require(evidence['status'] == 'no_sample_stored'
                     and all(evidence[k] is None for k in ('raw_orientation_logit', 'logit_sign', 'stored_decision', 'evidence_item')),
                     'unseen content must be undefined')
            _keys(evidence['decision_if_no_sample'], {'value', 'basis'}, 'default guess')
            _require(evidence['decision_if_no_sample'] == {'value': 'horizontal', 'basis': 'fixed default guess'}, 'unexpected default guess')
        for key in ('estimated_current_decision_success', 'estimated_fresh_glimpse_decision_success'):
            _number(node[key], probability=True)
        risk = node['risk_feature_source']
        _keys(risk, {'model', 'field', 'dimensions', 'meaning'}, 'risk feature provenance')
        _require(risk == {'model': value['frozen_sources']['outcome_model'], 'field': 'risk_hidden', 'dimensions': 32,
                          'meaning': 'local learned decision-outcome representation; coordinates not interpreted'}, 'unexpected risk provenance')
        external = node['external_image_statistic_readouts']
        _keys(external, {'status', 'source', 'units', 'image_value_std_estimate', 'nearest_neighbor_rms_estimate',
                         'interpretation'}, 'external readout')
        _require(external['source'] == 'external shared linear readout of frozen risk_hidden'
                 and external['units'] == 'normalized image-value units'
                 and external['interpretation'] == 'linear estimates of objective stored-image statistics', 'unexpected external semantics')
        _require(external['status'] == ('estimated_for_stored_sample' if node['observed'] else 'undefined_without_stored_sample'),
                 'external readout status mismatch')
        for key in ('image_value_std_estimate', 'nearest_neighbor_rms_estimate'):
            if node['observed']:
                _number(external[key])  # Unclipped linear estimates, negative allowed.
            else:
                _require(external[key] is None, 'unseen external readout must be undefined')
    for row in commands.values():
        for key in ('predicted_delivery_distribution', 'estimated_decision_success_after_command'):
            _keys(row[key], LOCATION_NAMES, 'forecast address map')
            for number in row[key].values():
                _number(number, probability=True)
        _require(abs(sum(row['predicted_delivery_distribution'].values())-1) <= 1e-12, 'delivery probabilities do not sum to one')
        for loc, node in nodes.items():
            d = row['predicted_delivery_distribution'][loc]
            expected = d*node['estimated_fresh_glimpse_decision_success']+(1-d)*node['estimated_current_decision_success']
            _require(row['estimated_decision_success_after_command'][loc] == expected, 'joined post-command composition mismatch')


def address_aware_tables(reporter_input):
    """Same information in unjoined address-aware tables, no dropped relation."""
    validate_reporter_input(reporter_input)
    value = copy.deepcopy(reporter_input)
    locations = value.pop('locations')
    commands = value.pop('command_forecasts')
    value['tables'] = {
        'stored_evidence': [{k: node[k] for k in ('location', 'observed', 'stored_evidence')} for node in locations],
        'risk_forecasts': [{k: node[k] for k in ('location', 'estimated_current_decision_success',
                                               'estimated_fresh_glimpse_decision_success')} for node in locations],
        'external_readouts': [{k: node[k] for k in ('location', 'risk_feature_source',
                                                  'external_image_statistic_readouts')} for node in locations],
        'delivery_forecasts': [{k: row[k] for k in ('command', 'predicted_delivery_distribution')} for row in commands],
        'post_command_forecasts': [{k: row[k] for k in ('command', 'estimated_decision_success_after_command')} for row in commands]}
    return value


def join_address_aware_tables(table_input):
    _validate_metadata(table_input, {'tables'})
    value = copy.deepcopy(table_input)
    tables = value.pop('tables')
    expected = {'stored_evidence', 'risk_forecasts', 'external_readouts', 'delivery_forecasts', 'post_command_forecasts'}
    _keys(tables, expected, 'table namespaces')
    for name, (address, fields) in TABLE_SCHEMAS.items():
        _address_rows(tables[name], address, fields)
    nodes = copy.deepcopy(tables['stored_evidence'])
    rows = copy.deepcopy(tables['delivery_forecasts'])
    for name in ('risk_forecasts', 'external_readouts'):
        by_location = {node['location']: node for node in tables[name]}
        _require(len(by_location) == 4 and set(by_location) == {node['location'] for node in nodes}, 'unmatched location addresses')
        for node in nodes:
            node.update(by_location[node['location']])
    by_command = {row['command']: row for row in tables['post_command_forecasts']}
    _require(len(by_command) == 4 and set(by_command) == {row['command'] for row in rows}, 'unmatched command addresses')
    for row in rows:
        row.update(by_command[row['command']])
    value['locations'], value['command_forecasts'] = nodes, rows
    validate_reporter_input(value)
    return value


