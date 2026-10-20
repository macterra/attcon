"""Explicit frozen same-address source extension; importing is stdlib-only.

No fitting, rendering, collection, Reporter, API or automatic retry. The original
heldout validation is applied to its ORIGINAL binding; a separate registration
releases this extension. All numerical work is confined to build_source_export.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = 'audits/attention_relational_report_v1/same_address_pilot_v1'
SCHEMA = 'same_address_model_report_source_v1_registration'
SCOPE = 'frozen_same_address_source_inference_only'
PLAN = 'docs/SAME_ADDRESS_MODEL_REPORT_PILOT_V1_PLAN.md'
INVENTORY = BASE + '/design_inventory.json'
ORIGINAL_INPUTS = 'audits/attention_relational_report_v1/registration/inputs.json'
OUTPUT = BASE + '/source_run'
ARTIFACT_PATHS = {
    'heldout': 'audits/attention_content_model_v1/run/heldout/states-1311.pt.gz',
    'fit': 'audits/attention_content_model_v1/run/fit/states-1311.pt.gz',
    'native': 'audits/spatial_native_task_v1/run/states-1311.pt.gz',
}
RECEIPT_PATHS = {
    'fit': 'audits/attention_content_model_v1/run/fit/receipt.json',
    'heldout': 'audits/attention_content_model_v1/run/heldout/receipt.json',
}
REQUIRED_SOURCES = {
    'scripts/same_address_model_report_source_v1.py',
    'tests/test_same_address_model_report_source_v1.py',
    'scripts/attention_relational_report_v1.py',
    'scripts/attention_content_model_v1_reference.py',
    'scripts/attention_content_model_v1.py',
    'scripts/attention_content_form_primitives_v1.py',
    'scripts/spatial_native_task_v1.py',
}
SOURCES = ('A', 'ordinary', 'analytic')
VARIANTS = ('baseline', 'source_donor', 'source_restored', 'W_changed', 'W_restored')
COMMANDS = ('k0', 'k1', 'k2')
ROLES = ('red', 'blue')
SCORE_RULE = ('maximum supplied localization probability for the queried role; '
              'choose first command index on exact ties')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def relative_path(value, root=ROOT):
    require(type(value) is str, 'path must be a string')
    path = Path(value)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == value,
            'canonical repository-relative path required')
    require((root / path).resolve().is_relative_to(root.resolve()), 'path escapes repository')
    return root / path


def descriptor(value, root=ROOT, expected=None):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'invalid descriptor')
    if expected is not None:
        require(value['path'] == expected, 'unexpected bound path')
    path = relative_path(value['path'], root)
    require(digest(path) == value['sha256'], 'bound bytes differ: ' + value['path'])
    return path


def expected_inventory():
    rows, pairs = [], []
    for world in range(4):
        for source in (*SOURCES, 'W'):
            variants = VARIANTS if source != 'W' else ('baseline', 'W_changed', 'W_restored')
            ids = {}
            for variant in variants:
                view = 'joined' if source != 'W' else 'W_only'
                key = f's1311_w{world:02d}_t12_{source}_{view}_{variant}_same_address_v1'
                ids[variant] = key
                rows.append(dict(id=key, native_split='heldout', encoder_seed=1311, world_index=world,
                                 decision_time=12, previous_node='n3', candidate_node='n3',
                                 queried_role=ROLES[world % 2], source=source, view=view, variant=variant))
            for kind, first, second in (
                (('source_donor', 'baseline', 'source_donor'),
                 ('source_donor_to_restored', 'source_donor', 'source_restored'),
                 ('source_baseline_to_restored', 'baseline', 'source_restored')) if source != 'W' else ()
            ):
                pairs.append(dict(kind=kind, first_id=ids[first], second_id=ids[second]))
            for kind, first, second in (
                ('W_changed', 'baseline', 'W_changed'),
                ('W_changed_to_restored', 'W_changed', 'W_restored'),
                ('W_baseline_to_restored', 'baseline', 'W_restored')):
                pairs.append(dict(kind=kind, first_id=ids[first], second_id=ids[second]))
    return rows, pairs


def validate_registration(registration, root=ROOT):
    """Only JSON/hash reads; no torch import/archive deserialization or inference."""
    require(set(registration) == {'schema', 'approved', 'scope', 'source_sha256', 'plan', 'inventory',
                                 'original_inputs', 'artifacts', 'receipts', 'output_dir', 'cases', 'execution_batch_context'},
            'unexpected registration fields')
    require(registration['schema'] == SCHEMA and registration['approved'] is True
            and registration['scope'] == SCOPE, 'new frozen source release required')
    require(registration['cases'] == dict(native_split='heldout', encoder_seed=1311,
                                         world_indices=[0, 1, 2, 3], decision_time=12, address=3),
            'fixed pilot cases differ')
    require(registration['execution_batch_context'] == 'original_archived_96_rows', 'original arithmetic batch context required')
    require(registration['output_dir'] == OUTPUT, 'fixed exclusive output directory required')
    hashes = registration['source_sha256']
    require(REQUIRED_SOURCES <= set(hashes), 'incomplete source closure')
    for name, value in hashes.items():
        descriptor(dict(path=name, sha256=value), root)
    descriptor(registration['plan'], root, PLAN)
    inventory_path = descriptor(registration['inventory'], root, INVENTORY)
    inventory = json.loads(inventory_path.read_text())
    rows, pairs = expected_inventory()
    require(inventory['opportunities'] == rows and inventory['pairs'] == pairs
            and inventory['opportunity_count'] == 72 and inventory['pair_count'] == 84,
            'design inventory differs from fixed source inventory')
    require(inventory['plan'] == registration['plan'], 'inventory plan binding differs')
    descriptor(registration['original_inputs'], root, ORIGINAL_INPUTS)
    require(set(registration['artifacts']) == set(ARTIFACT_PATHS), 'artifact inventory differs')
    require(set(registration['receipts']) == set(RECEIPT_PATHS), 'receipt inventory differs')
    for name, path in ARTIFACT_PATHS.items():
        descriptor(registration['artifacts'][name], root, path)
    for name, path in RECEIPT_PATHS.items():
        descriptor(registration['receipts'][name], root, path)
    return inventory


def tensor_hash(value):
    import torch
    require(type(value) is torch.Tensor and value.device.type == 'cpu'
            and not value.requires_grad, 'detached CPU tensor required')
    data = value.contiguous()
    # View avoids conversion/rounding and supports float32, long and boolean states.
    payload = data.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(str(data.dtype).encode() + str(tuple(data.shape)).encode() + payload).hexdigest()


def state_payload(state):
    if hasattr(state, 'hidden'):
        fields = {'hidden': state.hidden.clone()}
    else:
        fields = {'survivors': state.survivors.clone(), 'revealed_symbols': state.revealed_symbols.clone()}
    return dict(**fields, source_key=state.source_key, revealed_through=state.revealed_through,
                update_command_indices=state.update_command_indices)


def state_hashes(state):
    return {key: tensor_hash(value) if hasattr(value, 'shape') else value
            for key, value in state_payload(state).items()}


def observation_hashes(observation, candidate):
    return dict(V=tensor_hash(observation.v), W11=tensor_hash(observation.last_revealed_w),
                previous_address=tensor_hash(observation.previous_command.address),
                previous_symbol=tensor_hash(observation.previous_command.symbol),
                observed=tensor_hash(observation.observed), candidate=tensor_hash(candidate),
                decision_time=observation.decision_time, last_revealed_at=observation.last_revealed_at)


def predict_checked(source, observation, state, candidate, partial=None):
    """The only predictor boundary: V,W11,command11,state through11,candidate3."""
    import torch
    require(observation.decision_time == 12 and observation.last_revealed_at == 11
            and state.revealed_through == 11 and state.update_command_indices == tuple(range(11)),
            'decision12 pre-reveal permissions differ')
    require(bool((candidate[:4] == 3).all()) and bool((observation.previous_command.address == 3).all()),
            'selected candidates and all previous addresses must be item3')
    before_inputs, before_state = observation_hashes(observation, candidate), state_hashes(state)
    prediction = source.predict(observation, state, candidate)
    if partial is not None:
        partial['prediction'] = dict(current=prediction.current, next=prediction.next)
    prediction.validate()
    require(prediction.source_key == source.source_key == state.source_key, 'prediction source ownership differs')
    require(before_inputs == observation_hashes(observation, candidate), 'predictor mutated inputs')
    require(before_state == state_hashes(state), 'predictor mutated state')
    mask = torch.ones((candidate.shape[0], 4), dtype=torch.bool)
    mask[:, 3] = False
    require(torch.equal(prediction.current[mask], observation.last_revealed_w[mask]),
            'nonaddressed current must copy W11 exactly')
    inheritance = (prediction.next == prediction.current[:, None]).flatten(3).all(-1)
    inheritance[torch.arange(candidate.shape[0])[:, None], torch.arange(3)[None, :], candidate[:, None]] = True
    require(bool(inheritance.all()), 'noncandidate next must inherit current exactly')
    # Address ownership is established through the decoder, not inequality to W.
    rows = torch.arange(candidate.shape[0])
    if hasattr(state, 'hidden'):
        decoded = source.decode(observation.v[rows, observation.previous_command.address], state.hidden[rows, observation.previous_command.address],
                                observation.previous_command.symbol)
    else:
        decoded = source.decode(observation.v[rows, observation.previous_command.address], state.survivors[rows, observation.previous_command.address],
                                observation.previous_command.symbol)
    require(torch.equal(prediction.current[:, 3], decoded), 'current is not the addressed source decoder output')
    require(before_inputs == observation_hashes(observation, candidate) and before_state == state_hashes(state),
            'ownership decoder mutated inputs or state')
    return prediction, dict(inputs=before_inputs, state=before_state, source_key=source.source_key,
                            current=tensor_hash(prediction.current), next=tensor_hash(prediction.next),
                            current_source_decoder_exact=True, current_completion_visible=False,
                            fork_outcomes_visible=False)


def source_triplet(source, observation, recipient, donor, candidate, progress=None):
    from attention_content_model_v1 import snapshot_state, patch_addressed_state, restore_state
    import torch
    progress = {} if progress is None else progress
    saved = snapshot_state(recipient)
    progress['saved_state'] = state_payload(saved)
    patched = patch_addressed_state(recipient, donor, observation.previous_command.address)
    progress['donor_state'] = state_payload(donor)
    progress['patched_state'] = state_payload(patched)
    states = {'baseline': recipient, 'source_donor': patched, 'source_restored': restore_state(saved)}
    result = {}
    for variant, state in states.items():
        progress[variant] = dict(state=state_payload(state))
        prediction, receipt = predict_checked(source, observation, state, candidate, progress[variant])
        result[variant] = prediction
        progress[variant]['receipt'] = receipt
    require(state_hashes(saved) == state_hashes(states['source_restored']), 'restored source state differs')
    for name in ('current', 'next'):
        require(torch.equal(getattr(result['baseline'], name), getattr(result['source_restored'], name)),
                'restored source maps differ')
        axis = 1 if name == 'current' else 2
        before, after = getattr(result['baseline'], name), getattr(result['source_donor'], name)
        require(torch.equal(before.narrow(axis, 0, 3), after.narrow(axis, 0, 3)),
                'donor changed nonselected native maps')
    saved_fields, patched_fields, donor_fields = state_payload(saved), state_payload(patched), state_payload(donor)
    for name in ('hidden',) if hasattr(saved, 'hidden') else ('survivors', 'revealed_symbols'):
        require(torch.equal(saved_fields[name][:, :3], patched_fields[name][:, :3]), 'donor changed nonselected state')
        require(torch.equal(patched_fields[name][:, 3], donor_fields[name][:, 3]), 'addressed donor transfer differs')
    return result


def native_controls(current, alternatives):
    """Pure map assembly: fixed k0 counterfactual, original futures, saved restore."""
    import torch
    require(current.shape[1:] == (4, 24, 4, 4) and alternatives.shape == (current.shape[0], 3, 24, 4, 4),
            'invalid native control map shapes')
    future = current[:, None].expand(-1, 3, -1, -1, -1, -1).clone()
    future[:, :, 3] = alternatives
    changed = current.clone()
    changed[:, 3] = alternatives[:, 0]
    restored = current.clone()
    require(torch.equal(restored, current), 'native restoration differs')
    return {'baseline': dict(current=current.clone(), next=future.clone()),
            'W_changed': dict(current=changed, next=future.clone()),
            'W_restored': dict(current=restored, next=future.clone())}


def scores(fields, native_choice, native_maxima):
    import torch
    require(tuple(fields.shape) == (3, 4, 4), 'three queried-role fields required')
    values = fields.flatten(1).max(-1).values
    require(torch.equal(values, native_maxima), 'native processing score reconstruction differs')
    raw = values.tolist()
    winners = [COMMANDS[i] for i, value in enumerate(raw) if value == max(raw)]
    require(COMMANDS[int(native_choice)] == winners[0], 'native first-index choice differs')
    ordered = sorted(raw, reverse=True)
    return dict(rule=SCORE_RULE, scores=dict(zip(COMMANDS, raw)), tied_winners=winners,
                first_index_choice=winners[0], top_two_margin=ordered[0] - ordered[1])


def origin(source, view, variant):
    if view == 'candidate':
        return dict(kind='native_conditional_record' if source == 'W' else 'command_estimate',
                    represented_time=13, executed_branch_claim=False,
                    **({} if source == 'W' else {'feedback_through': 11}))
    if source != 'W':
        return dict(kind='history_estimate', represented_time=12, feedback_through=11)
    if variant == 'W_changed':
        return dict(kind='counterfactual_native_snapshot', represented_time=12, archived_origin_time=13,
                    executed_branch_claim=False, fixed_command='k0')
    return dict(kind='completed_record', represented_time=12, record_time=12)


def public_panel(name, source, variant, role, current_fields, future_fields, choice, maxima):
    from attention_relational_report_v1 import _field_bytes, localization_summary
    _field_bytes(current_fields.tolist(), (4, 4))
    for field in future_fields:
        _field_bytes(field.tolist(), (4, 4))
    def node(field, view):
        return dict(node='n3', role=role, origin=origin(source, view, variant), distribution=field.tolist(), derived=localization_summary(field.tolist()))
    return dict(panel=name, panel_role='native_reference' if source == 'W' else 'predictor_estimate', current=node(current_fields, 'current'),
                candidates={command: node(future_fields[index], 'candidate') for index, command in enumerate(COMMANDS)},
                processing_scores=scores(future_fields, choice, maxima))


def readout_prediction(readout, prediction, candidate, roles):
    import torch
    from attention_content_model_v1 import PredictionBatch
    selected = PredictionBatch(prediction.current[:4], prediction.next[:4], prediction.decision_time, prediction.source_key)
    current = readout.consume(selected.current[:, 3])
    future = readout.consume(selected.next[:, :, 3])
    choice = readout.choose(selected, candidate[:4], roles[:4])
    require(torch.equal(choice.candidate_fields, future.role_fields), 'native choose fields differ')
    return dict(current_logits=current.local_logits, current_fields=current.role_fields,
                next_logits=future.local_logits, next_fields=future.role_fields,
                choice=choice.symbol, maxima=choice.maxima)


def build_source_export(bundle, fit, native, inventory, progress=None):
    """Explicit bounded inference, only after separate registration gates pass.

    Arguments are private retained inputs; no file loading or writes occur here.
    No W12, evaluator mapping, pixels, world ID or forks cross predict_checked.
    """
    import torch
    from attention_content_model_v1 import (FitNormalizer, LearnedContentModel, AnalyticMappingSource,
        SourceState, AnalyticState, ObservationBatch, CommandBatch, PredictionBatch,
        NativePredictionReadout, snapshot_state)
    from spatial_native_task_v1 import LocalOccupancyHead, ChannelStandardizer
    from attention_relational_report_v1 import _actual_maps, _donor_evidence
    progress = {} if progress is None else progress
    rows, pairs = expected_inventory()
    require(inventory['opportunities'] == rows and inventory['pairs'] == pairs, 'inventory mismatch')
    episode, evaluation = bundle['episode'], bundle['evaluation']
    require(episode['split'] == 'heldout' and episode['encoder_seed'] == 1311, 'wrong native cases')
    shared = evaluation['tensors']['shared']
    indices = torch.arange(4)
    require(torch.equal(episode['command_addresses'][:4, 11], torch.full((4,), 3, dtype=torch.long)),
            'previous address must be3')
    require(torch.equal(episode['command_symbols'][:4, 11], torch.full((4,), 2, dtype=torch.long)),
            'registered previous command must be c2')
    roles = episode['query_roles'][:4, 12].clone()
    require(torch.equal(roles, torch.tensor([0, 1, 0, 1])), 'queried role differs')
    require(episode['V'].shape[0] == 96, 'original96 arithmetic context required')
    v = episode['V'].clone()
    w11 = torch.stack([_actual_maps(episode, shared, torch, world, 11) for world in range(96)])
    # W12 is constructed only in evaluator scope, never supplied to predictor.
    w12 = torch.stack([_actual_maps(episode, shared, torch, world, 12) for world in range(4)])
    candidate = episode['command_addresses'][:, 12].clone()
    candidate[:4] = 3
    previous_address = episode['command_addresses'][:, 11].clone()
    observation = ObservationBatch(v, w11, 12, 11, CommandBatch(previous_address,
                                   episode['command_symbols'][:, 11].clone()),
                                   torch.ones((96, 4), dtype=torch.bool))
    observation.validate()
    progress.update(observation=dict(V=v, W11=w11, previous_address=previous_address.clone(),
                    previous_symbol=observation.previous_command.symbol.clone()),
                    native_current_W12=w12, source_states={}, native_readouts={}, source_predictions={})
    head = LocalOccupancyHead().float().cpu()
    head.load_state_dict(native['native_head_fit']['weights'])
    coordinates = native['native_head_fit']['fit_only_standardizer']
    readout = NativePredictionReadout(head, ChannelStandardizer(tuple(coordinates['mean']), tuple(coordinates['scale'])))
    head_before = {name: tensor_hash(value) for name, value in readout.head.state_dict().items()}
    normalizer = FitNormalizer(tuple(fit['normalizer']['mean']), tuple(fit['normalizer']['scale']))
    source_readouts, provenance = {}, {}
    for key in SOURCES:
        retained = evaluation['tensors']['sources'][key]
        if key == 'analytic':
            source = AnalyticMappingSource()
            recipient = AnalyticState(retained['pre_reveal_survivors'][:, 12].clone(),
                                      retained['pre_reveal_revealed_symbols'][:, 12].clone(),
                                      11, tuple(range(11)))
        else:
            source = LearnedContentModel(normalizer, 0, key).eval()
            source.load_state_dict(fit['checkpoints'][key])
            recipient = SourceState(retained['pre_reveal_hidden'][:, 12].clone(), 11, tuple(range(11)), key)
        recipient.validate(96)
        weights_before = {name: tensor_hash(value) for name, value in source.state_dict().items()} if key != 'analytic' else {}
        donor = snapshot_state(recipient)
        ledger = retained['donors']['current']
        if key == 'analytic':
            donor.survivors[:, 3] = ledger['donor_survivors_at_patched_address'][:, 0]
            donor.revealed_symbols[:, 3] = ledger['donor_revealed_symbols_at_patched_address'][:, 0]
        else:
            donor.hidden[:, 3] = ledger['donor_hidden_at_patched_address'][:, 0]
        evidence = [_donor_evidence(evaluation, key, 'current', world, 12, torch) for world in range(4)]
        require(all(item['patched_address'] == 3 for item in evidence), 'donor receipt address differs')
        progress['source_states'][key] = {}
        predictions = source_triplet(source, observation, recipient, donor, candidate, progress['source_states'][key])
        progress['source_predictions'][key] = {variant: dict(current=value.current, next=value.next)
                                                       for variant, value in predictions.items()}
        source_readouts[key] = {}
        for variant, prediction in predictions.items():
            values = readout_prediction(readout, prediction, candidate, roles)
            source_readouts[key][variant] = values
            progress['native_readouts'][key + '/' + variant] = values
            archived_current = retained['selected_current_maps'][:, 11] if variant != 'source_donor' else ledger['selected_current_maps'][:, 0]
            archived_fields = retained['selected_current_fields'][:4, 11] if variant != 'source_donor' else ledger['selected_current_fields'][:4, 0]
            require(torch.equal(prediction.next[:4, 2, 3], prediction.current[:4, 3]), 'same-address c2 conditional/current identity differs')
            require(torch.equal(prediction.current[:, 3], archived_current), 'archived current native map mismatch: ' + key + '/' + variant)
            require(torch.equal(values['current_fields'], archived_fields), 'archived current H mismatch: ' + key + '/' + variant)
            archived_next = retained['selected_next_maps'][3, 12] if variant != 'source_donor' else ledger['selected_next_maps'][3, 0]
            archived_next_fields = retained['selected_next_fields'][3, 12] if variant != 'source_donor' else ledger['policy_candidate_fields'][3, 0]
            require(torch.equal(prediction.next[3, :, 3], archived_next), 'world3 original next map mismatch')
            require(torch.equal(values['next_fields'][3], archived_next_fields), 'world3 original next H mismatch')
            context_address = candidate[4:]
            context_next = prediction.next[torch.arange(4, 96)[:, None], torch.arange(3)[None, :], context_address[:, None]]
            archived_context_next = retained['selected_next_maps'][4:, 12] if variant != 'source_donor' else ledger['selected_next_maps'][4:, 0]
            require(torch.equal(context_next, archived_context_next), 'original92 context native next mismatch')
        for name in ('current_fields', 'next_fields', 'current_logits', 'next_logits', 'choice', 'maxima'):
            require(torch.equal(source_readouts[key]['baseline'][name], source_readouts[key]['source_restored'][name]),
                    'restored native readout differs')
        if key != 'analytic':
            require(weights_before == {name: tensor_hash(value) for name, value in source.state_dict().items()},
                    'fitted weights changed')
        provenance[key] = dict(weights=weights_before, donor_evidence=evidence,
                               baseline_state=state_hashes(recipient), donor_state=state_hashes(donor))
    bands = episode['evaluator_mappings'][:4, 3]
    family_indices = torch.empty_like(bands)
    for band, index in ((1, 0), (2, 1), (4, 2)):
        family_indices[bands == band] = index
    require(bool(((bands == 1) | (bands == 2) | (bands == 4)).all()), 'invalid native operator family index')
    alternatives = shared['actual_operator_family'][indices[:, None], 3, family_indices]
    controls = native_controls(w12, alternatives)
    progress['W_controls'] = controls
    w_readouts = {}
    for variant, maps in controls.items():
        prediction = PredictionBatch(maps['current'], maps['next'], 12, 'W')
        w_readouts[variant] = readout_prediction(readout, prediction, candidate, roles)
        expected_current = shared['actual_completed_fields'][:4, 12, 3] if variant != 'W_changed' else shared['actual_family_fields'][indices, 3, family_indices[:, 0]]
        require(torch.equal(w_readouts[variant]['current_fields'], expected_current), 'native W current readout mismatch')
        require(torch.equal(w_readouts[variant]['next_fields'], shared['actual_family_fields'][indices[:, None], 3, family_indices]),
                'native W conditional readout mismatch')
    progress['W_readouts'] = w_readouts
    require(head_before == {name: tensor_hash(value) for name, value in readout.head.state_dict().items()}, 'native H weights changed')
    packets, audit_rows = [], []
    for case in rows:
        world, source, variant = case['world_index'], case['source'], case['variant']
        panels, private_panels = [], []
        for name, panel_source, panel_variant in (
            [('p0', source, variant), ('p1', 'W', variant if variant.startswith('W_') else 'baseline')]
            if source != 'W' else [('p0', 'W', variant)]):
            values = (w_readouts[panel_variant] if panel_source == 'W' else
                      source_readouts[panel_source][panel_variant if panel_variant.startswith('source_') else 'baseline'])
            role = int(roles[world])
            panels.append(public_panel(name, panel_source, panel_variant, ROLES[role],
                                       values['current_fields'][world, role], values['next_fields'][world, :, role],
                                       values['choice'][world], values['maxima'][world]))
            private_panels.append(dict(panel=name, source_key=panel_source,
                current_fields_sha256=tensor_hash(values['current_fields'][world]),
                next_fields_sha256=tensor_hash(values['next_fields'][world])))
        content = dict(schema='same_address_model_report_source_v1_packet', decision_time=12, selected_node='n3',
                       queried_role=ROLES[int(roles[world])], panels=panels,
                       coordinate_system=dict(rows=4, columns=4, row_direction='down', column_direction='right',
                                              separate_local_grid=True),
                       field_semantics='model/readout localization values; not calibrated confidence or physical truth',
                       conditional_semantics='possible processing operations; supplied estimates/records, not executed outcomes',
                       processing_semantics='native operation-score selection; not Controller targeting or forecast execution')
        packets.append(dict(id=case['id'], content=content, content_sha256=hashlib.sha256(encoded(content)).hexdigest()))
        audit_rows.append(dict(id=case['id'], case=case, panels=private_panels))
    index = {row['id']: row for row in packets}
    checks = []
    for pair in pairs:
        first, second = index[pair['first_id']], index[pair['second_id']]
        identical = first['content_sha256'] == second['content_sha256']
        if pair['kind'] in ('source_baseline_to_restored', 'W_baseline_to_restored'):
            require(identical, 'restored public input differs')
        a, b = first['content']['panels'], second['content']['panels']
        if pair['kind'].startswith('source_'):
            require(a[1] == b[1], 'source intervention changed W panel')
        elif len(a) == 2:
            require(a[0] == b[0], 'W control changed source panel')
        checks.append(dict(**pair, public_input_identical=identical, null_pair_retained=True))
    require(len(packets) == 72 and len(checks) == 84, 'bounded counts differ')
    return dict(schema='same_address_model_report_source_v1_export', native_head_weights=head_before,
                execution_batch_context='original_archived_96_rows', output_world_indices=[0, 1, 2, 3],
                torch_num_threads=4, packets=packets, provenance=audit_rows,
                pairs=checks, source_receipts=provenance, case_count=4, opportunity_count=72, pair_count=84,
                reused_exposed_heldout_native_cases=True, fresh_independent_sample=False, reporter_called=False)


def _exclusive_json(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))


def selected_private(value):
    """Retain only selected row values; full96 receipts retain context hashes."""
    if hasattr(value, 'shape'):
        return value[:4].clone() if value.ndim and value.shape[0] == 96 else value.clone()
    if isinstance(value, dict):
        return {key: selected_private(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(selected_private(item) for item in value)
    return value


def run_registration(registration_path, registration_sha256):
    """One exclusive first source run. Root publishes/calls this separately."""
    registration_file = relative_path(registration_path)
    require(digest(registration_file) == registration_sha256, 'registration hash differs')
    registration = json.loads(registration_file.read_text())
    output = relative_path(OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    _exclusive_json(output / 'first_claim.json', dict(schema='same_address_model_report_source_v1_claim',
        registration_path=registration_path, registration_sha256=registration_sha256,
        started_at_utc=datetime.now(timezone.utc).isoformat(), retry_performed=False))
    progress, receipt = {}, dict(schema='same_address_model_report_source_v1_receipt',
        registration_sha256=registration_sha256, retry_performed=False, reporter_called=False)
    try:
        inventory = validate_registration(registration)
        from attention_relational_report_v1 import validate_input_bindings
        original = json.loads(descriptor(registration['original_inputs']).read_text())
        # ORIGINAL closure and ORIGINAL archive paths only. This validates prior
        # provenance; it does not release inference. The new registration does.
        old_paths = {seed: f'audits/attention_content_model_v1/run/heldout/states-{seed}.pt.gz'
                     for seed in (1301, 1311, 1321)}
        context = validate_input_bindings(old_paths, original)
        for phase in ('fit', 'heldout'):
            require(registration['receipts'][phase]['sha256'] == original[phase + '_receipt']['sha256']
                    and registration['receipts'][phase]['path'] == original[phase + '_receipt']['path'],
                    'new receipt differs from original accepted receipt')
            require(registration['artifacts'][phase]['sha256'] == context[phase + '_receipt']['artifact_sha256'][ARTIFACT_PATHS[phase]],
                    'new artifact differs from original accepted phase')
        require(registration['artifacts']['native']['sha256'] == context['registration']['input_hashes'][ARTIFACT_PATHS['native']],
                'native archive differs from original registered bytes')
        receipt['original_binding_context_sha256'] = hashlib.sha256(encoded(context)).hexdigest()
        import torch
        torch.set_num_threads(4)
        def load(name):
            with gzip.open(descriptor(registration['artifacts'][name]), 'rb') as stream:
                return torch.load(stream, weights_only=True, map_location='cpu')
        bundle, fitted, native = load('heldout'), load('fit'), load('native')
        require(bundle['fitted_artifact_sha256'] == context['fit_receipt']['artifact_sha256'], 'heldout fitted identity differs')
        with torch.no_grad():
            result = build_source_export(bundle, fitted['fit'], native, inventory, progress)
        result['bindings'] = deepcopy(registration)
        _exclusive_json(output / 'first_exports.json', result)
        _exclusive_json(output / 'neutral_packets.json', result['packets'])
        _exclusive_json(output / 'pair_index.json', result['pairs'])
        receipt.update(status='completed', opportunity_count=72, pair_count=84)
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        # Private tensors/state retained on success and failure; no partial
        # public source inventory is released before all checks complete.
        private_error = None
        if progress:
            try:
                import torch
                with gzip.open(output / 'first_private.pt.gz', 'xb') as stream:
                    torch.save(selected_private(progress), stream)
            except Exception as exc:
                private_error = exc
                receipt.update(status='failed', private_retention_error_type=type(exc).__name__,
                               private_retention_error=str(exc))
        receipt['artifact_sha256'] = {path.name: digest(path) for path in output.iterdir()
                                      if path.is_file() and path.name != 'first_receipt.json'}
        receipt['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        _exclusive_json(output / 'first_receipt.json', receipt)
        if private_error is not None:
            raise private_error
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run_registration(args.registration, args.registration_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
