"""Prospective experiment engine; definitions only, no automatic execution.

The external phase runner owns frozen-artifact loading, releases and immutable
receipts. Evaluator episode dictionaries NEVER enter a source's prediction API.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib

import torch

from attention_content_model_v1 import (
    AnalyticMappingSource, AnalyticState, CommandBatch, ExactAlias,
    FitNormalizer, MAPPINGS, NativePredictionReadout, ObservationBatch,
    RevealedFeedback, SourceState, build_learned_sources, choose_from_maxima,
    patch_addressed_state, restore_state, snapshot_state,
)
from attention_content_model_v1_data import (
    ENCODER_SEEDS, WORLDS, donor_prefix, execute_command, operator_family,
)
from attention_content_form_primitives_v1 import (
    AddressedItem, AddressedState, PendingOperator, execute_endogenous,
    execute_exogenous, restore_saved,
)
from spatial_native_task_v1 import ADDRESSES, NativeController

SOURCES = ('A', 'ordinary', 'analytic')


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f'{value.dtype}:{tuple(value.shape)}:'.encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _emit(log, record):
    if log is not None:
        log(record)


def _validate_episode(episode, split, encoder_seed=None):
    if split not in ('fit', 'dev', 'heldout') or episode.get('split') != split:
        raise ValueError('episode belongs to a different registered split')
    seed = episode.get('encoder_seed')
    if seed not in ENCODER_SEEDS or (encoder_seed is not None and seed != encoder_seed):
        raise ValueError('wrong frozen encoder seed')
    v = episode['V']
    if (v.shape != (WORLDS, 4, 24, 4, 4) or v.dtype != torch.float32
            or v.device.type != 'cpu' or not torch.isfinite(v).all()):
        raise ValueError('registered evaluation requires 96 finite float32 worlds')
    if (episode['actual_completed_w'].shape != (WORLDS, 16, 4, 24, 4, 4)
            or episode['actual_forks'].shape != (WORLDS, 15, 3, 4, 24, 4, 4)):
        raise ValueError('wrong fifteen-transition actual ledger')
    for key in ('actual_completed_w', 'actual_forks'):
        value = episode[key]
        if value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
            raise ValueError(f'invalid {key}')
    for key, upper in (('command_addresses', 4), ('command_symbols', 3)):
        value = episode[key]
        if (value.shape != (WORLDS, 15) or value.dtype != torch.long
                or value.device.type != 'cpu' or ((value < 0) | (value >= upper)).any()):
            raise ValueError(f'invalid {key}')
    mappings = episode['evaluator_mappings']
    if (mappings.shape != (WORLDS, 4, 3) or mappings.dtype != torch.long
            or mappings.device.type != 'cpu' or not torch.equal(
                mappings.sort(-1).values, torch.tensor((1, 2, 4)).expand(WORLDS, 4, 3))):
        raise ValueError('invalid evaluator mapping permutations')
    indices = episode['evaluator_mapping_indices']
    if (indices.shape != (WORLDS, 4) or indices.dtype != torch.long
            or indices.device.type != 'cpu' or ((indices < 0) | (indices >= 6)).any()
            or not torch.equal(torch.tensor(MAPPINGS)[indices], mappings)):
        raise ValueError('mapping indices disagree with evaluator assignments')
    roles = episode['query_roles']
    if (roles.shape != (WORLDS, 15) or roles.dtype != torch.long
            or roles.device.type != 'cpu' or not (roles[:, :12] == -1).all()
            or ((roles[:, 12:] < 0) | (roles[:, 12:] > 1)).any()):
        raise ValueError('three final single-role queries required')
    cells = episode['evaluator_cells']
    if (cells.shape != (WORLDS, 4, 2, 2) or cells.dtype != torch.long
            or cells.device.type != 'cpu' or ((cells < 0) | (cells > 3)).any()):
        raise ValueError('invalid evaluator constituent cells')
    return v


def _command(episode, index):
    return CommandBatch(episode['command_addresses'][:, index],
                        episode['command_symbols'][:, index])


def _observation(episode, time):
    # Construct a restricted boundary, never forward the evaluator dictionary.
    return ObservationBatch(episode['V'], episode['actual_completed_w'][:, max(0, time-1)],
                            time, max(0, time-1), _command(episode, time-1) if time else None,
                            torch.ones(WORLDS, 4, dtype=torch.bool))


def _feedback(episode, completion):
    return RevealedFeedback(episode['actual_completed_w'][:, completion],
                            _command(episode, completion-1), completion, completion-1, completion)


def _visibility(observation, state, kind='primary'):
    return {'kind': kind, 'decision_time': observation.decision_time,
            'last_revealed_at': observation.last_revealed_at,
            'source_revealed_through': state.revealed_through,
            'update_command_indices': list(state.update_command_indices),
            'V_hash': tensor_hash(observation.v),
            'last_revealed_W_hash': tensor_hash(observation.last_revealed_w),
            'observed_mask_hash': tensor_hash(observation.observed),
            'previous_address_hash': tensor_hash(observation.previous_command.address) if observation.previous_command else None,
            'previous_symbol_hash': tensor_hash(observation.previous_command.symbol) if observation.previous_command else None,
            'current_completion_visible': observation.decision_time == 0,
            'fork_outcomes_visible': False}


def fit_sources(encoder_seed, evaluator_episode, log=None, partial=None):
    """One exact fit per source; external runner must release fit before calling."""
    if partial is not None:
        partial['engine_fit'] = {'stage':'before_input_validation'}
    v = _validate_episode(evaluator_episode, 'fit', encoder_seed)
    torch.set_num_threads(4)
    normalizer = FitNormalizer.fit(v, split='fit')
    attention, ordinary, _alias = build_learned_sources(normalizer, encoder_seed)
    models = {'A': attention, 'ordinary': ordinary}
    initial = {name:deepcopy(model.state_dict()) for name,model in models.items()}
    terminal, training_log, visibility = {}, [], {}
    progress = {'stage':'initialized','normalizer':normalizer,'initial_checkpoints':initial,
                'training_log':training_log,'fit_visibility':visibility,'completed_updates':{}}
    if partial is not None:
        partial['engine_fit'] = progress
    rows = torch.arange(WORLDS)
    for name, model in models.items():
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999),
                                     eps=1e-8, weight_decay=0, amsgrad=False, foreach=False)
        source_visibility = []
        visibility[name] = source_visibility
        for update in range(300):
            progress.update({'stage':'before_update','source':name,'update_index':update,
                             'live_checkpoint':model.state_dict(),'live_optimizer':optimizer.state_dict()})
            optimizer.zero_grad(set_to_none=True)
            state = model.initial_state(WORLDS)
            current_losses, next_losses = [], []
            for time in range(15):
                obs = _observation(evaluator_episode, time)
                address = evaluator_episode['command_addresses'][:, time]
                progress.update({'stage':'before_prediction','decision_time':time,
                                 'current_observation':_visibility(obs,state)})
                prediction = model.predict(obs, state, address)
                if update == 0:
                    source_visibility.append(_visibility(obs, state))
                if time:
                    previous = obs.previous_command.address
                    predicted = prediction.current[rows, previous]
                    target = evaluator_episode['actual_completed_w'][rows, time, previous]
                    current_losses.append((normalizer.normalize(predicted) - normalizer.normalize(target)).square().mean())
                predicted = prediction.next[rows[:, None], torch.arange(3)[None, :], address[:, None]]
                target = evaluator_episode['actual_forks'][rows[:, None], time, torch.arange(3)[None, :], address[:, None]]
                next_losses.append((normalizer.normalize(predicted) - normalizer.normalize(target)).square().mean())
                # Targets have entered only the loss; feedback enters after scoring.
                if time:
                    state = model.update(state, v, _feedback(evaluator_episode, time))
            current_loss = torch.stack(current_losses).mean()
            next_loss = torch.stack(next_losses).mean()
            loss = .5 * current_loss + .5 * next_loss
            progress.update({'stage':'before_backward','current_loss':current_loss.detach(),
                             'next_loss':next_loss.detach(),'loss':loss.detach()})
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite {name} loss at fixed update {update}')
            try:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                progress['stage'] = 'before_optimizer_step'
                optimizer.step()
            except BaseException:
                progress.update({'stage':'failed_optimizer_or_backward',
                    'live_checkpoint':model.state_dict(),'live_optimizer':optimizer.state_dict(),
                    'live_gradients':{key:parameter.grad for key,parameter in model.named_parameters()}})
                raise
            progress['completed_updates'][name] = update+1
            record = {'source': name, 'update_index': update, 'completed_updates': update+1,
                      'loss': float(loss.detach()), 'current_loss': float(current_loss.detach()),
                      'next_loss': float(next_loss.detach()), 'preclip_gradient_norm': float(gradient_norm)}
            training_log.append(record); _emit(log, record)
        model.eval()
        terminal[name] = deepcopy(optimizer.state_dict())
    return {'models': models, 'normalizer': normalizer,
            'initial_checkpoints':initial,
            'checkpoints': {name: deepcopy(model.state_dict()) for name, model in models.items()},
            'optimizer_terminal_states': terminal, 'training_log': training_log,
            'fit_visibility': visibility, 'completed_updates_per_source': 300,
            'visibility_reconstruction':{'updates_per_source':300,'reset_state_each_update':True,
                'decision_indices':list(range(15)),'revealed_at':list(range(1,15)),
                'source_input_manifest':'identical permission/observation manifests to first update; fitted h differs',
                'fit_trajectory_independently_proven':False,
                'trajectory_reconstruction_requires_deterministic_optimizer_replay':True}}


def _per_world_equal(left, right):
    return (left == right).reshape(left.shape[0], -1).all(-1)


def _per_world_close(left, right):
    return torch.isclose(left, right, atol=1e-6, rtol=1e-5).reshape(left.shape[0], -1).all(-1)


def _coarse(fields):
    columns = (fields.double() * torch.arange(4, dtype=torch.float64)).sum((-2, -1))
    delta = columns[..., 0] - columns[..., 1]
    order = torch.where(delta.abs() <= 1e-6, 0, torch.where(delta < 0, 1, -1))
    return order, columns


def _field_metrics(estimated, actual, truth_cells):
    estimated_cells = estimated.flatten(-2).argmax(-1)
    actual_cells = actual.flatten(-2).argmax(-1)
    truth = truth_cells[..., 0] * 4 + truth_cells[..., 1]
    estimated_order, columns = _coarse(estimated)
    actual_order, _ = _coarse(actual)
    truth_order = torch.where(truth_cells[..., 0, 1] < truth_cells[..., 1, 1], 1, -1)
    flat = estimated.flatten(-2)
    top = flat.topk(2, dim=-1).values
    return {'field_l1': (estimated - actual).abs().mean((-3, -2, -1)),
            'cell_disagreement': estimated_cells != actual_cells,
            'estimated_cell_correct': estimated_cells == truth,
            'actual_cell_correct': actual_cells == truth,
            'coarse_disagreement': estimated_order != actual_order,
            'estimated_coarse_correct': estimated_order == truth_order,
            'actual_coarse_correct': actual_order == truth_order,
            'expected_columns': columns, 'entropy': -(flat.double() * flat.double().clamp_min(1e-300).log()).sum(-1),
            'maxima': top[..., 0], 'top_two_margin': top[..., 0] - top[..., 1],
            'max_tie_count': (flat == top[..., :1]).sum(-1)}


def _append(records, key, value):
    records.setdefault(key, []).append(value.detach().clone())


def _stack(records):
    return {key: torch.stack(values, 1) for key, values in records.items()}


def _actual_state(v_item, w_item, time):
    return AddressedState(tuple(AddressedItem.from_tensors(address, v_item[index], w_item[index],
                           True, 'delivered-spatial-item', 0., 0, time)
                               for index, address in enumerate(ADDRESSES)), time)


def _native_action(controller, v, w, addresses, roles, time):
    cells, fields, ties, orders = [], [], [], []
    for world in range(v.shape[0]):
        state = _actual_state(v[world], w[world], time)
        snapshot = controller.consume(state)
        address, role = ADDRESSES[int(addresses[world])], ('red', 'blue')[int(roles[world])]
        action = controller.target(snapshot, address, role)
        cells.append(action.row * 4 + action.column)
        fields.append(snapshot.item(address).role_fields.tensor())
        ties.append(len(action.tied_cells))
        orders.append(controller.coarse_order(snapshot, address).order)
    return {'cell': torch.tensor(cells), 'role_fields': torch.stack(fields),
            'tie_count': torch.tensor(ties), 'coarse_order': orders}


def _controls(controller, episode):
    checks, provenance, outputs = {}, [], {}
    v = episode['V']
    for time in (12, 13, 14):
        rows, emitted = [], {}
        for world in range(WORLDS):
            address = ADDRESSES[int(episode['command_addresses'][world, time])]
            role = ('red', 'blue')[int(episode['query_roles'][world, time])]
            saved = _actual_state(v[world], episode['actual_completed_w'][world, time], time)
            sham = execute_endogenous(saved, PendingOperator(address, 1, time, time+1))
            endogenous = execute_endogenous(saved, PendingOperator(address, 2, time, time+1))
            exogenous = execute_exogenous(saved, PendingOperator(address, 2, time, time+1))
            restored = restore_saved(saved)
            # Saved actual-W restoration after the T2 branch, never T inversion.
            changed_record = endogenous
            before_snapshot, restored_snapshot = controller.consume(saved), controller.consume(restored)
            end_snapshot, ex_snapshot = controller.consume(endogenous), controller.consume(exogenous)
            moved = AddressedState(tuple(replace(saved.items[(index-1)%4],address=ADDRESSES[index])
                                         for index in range(4)),time)
            moved_address = ADDRESSES[(ADDRESSES.index(address)+1)%4]
            moved_snapshot = controller.consume(moved)
            old_action = controller.target(before_snapshot,address,role)
            moved_action = controller.target(moved_snapshot,moved_address,role)
            rows.append({'sham_selected_exact_v': torch.equal(sham.item(address).actual_w.tensor(), saved.item(address).source_v.tensor()),
                         'sham_nonselected_unchanged': all(sham.item(a) == saved.item(a) for a in ADDRESSES if a != address),
                         'same_w_driver_exact': all(endogenous.item(a).actual_w == exogenous.item(a).actual_w for a in ADDRESSES),
                         'same_w_driver_fields_exact': end_snapshot == ex_snapshot,
                         'same_w_driver_action_exact': controller.target(end_snapshot, address, role) == controller.target(ex_snapshot, address, role),
                         'actual_restoration_exact': restored == saved,
                         'actual_restoration_fields_exact': before_snapshot == restored_snapshot,
                         'actual_restoration_action_exact': controller.target(before_snapshot, address, role) == controller.target(restored_snapshot, address, role),
                         'actual_address_fields_exact': all(moved_snapshot.item(ADDRESSES[(i+1)%4]).role_fields == before_snapshot.item(ADDRESSES[i]).role_fields for i in range(4)),
                         'actual_address_action_exact': replace(moved_action,address=address) == old_action})
            for label,state,snapshot,a in (('sham',sham,controller.consume(sham),address),
                                           ('endogenous',endogenous,end_snapshot,address),
                                           ('exogenous',exogenous,ex_snapshot,address),
                                           ('restored',restored,restored_snapshot,address),
                                           ('address_remapped',moved,moved_snapshot,moved_address)):
                emitted.setdefault(label+'_role_fields',[]).append(torch.stack([snapshot.item(a0).role_fields.tensor() for a0 in ADDRESSES]))
                emitted.setdefault(label+'_local_logits',[]).append(torch.stack([snapshot.item(a0).local_logits.tensor() for a0 in ADDRESSES]))
                action = controller.target(snapshot,a,role)
                emitted.setdefault(label+'_action_cell',[]).append(torch.tensor(action.row*4+action.column))
            provenance.append({'time': time, 'world_index': world, 'address': address,
                               'saved_w_hashes': [tensor_hash(saved.item(a).actual_w.tensor()) for a in ADDRESSES],
                               'changed_w_hashes': [tensor_hash(changed_record.item(a).actual_w.tensor()) for a in ADDRESSES],
                               'drivers': ['endogenous', 'exogenous'], 'restoration': 'exact_saved_snapshot'})
        for key in rows[0]:
            _append(checks, key, torch.tensor([row[key] for row in rows]))
        for key,value in emitted.items():
            _append(outputs,key,torch.stack(value))
    return _stack(checks), provenance, _stack(outputs)


def _donors(episode):
    result, provenance = {}, []
    for time in (12, 13, 14):
        for schedule, address in (('current', episode['command_addresses'][:, time-1]),
                                  ('prospective', episode['command_addresses'][:, time])):
            prefix = donor_prefix(episode['V'], episode, time, address)
            result[(time, schedule)] = prefix
            updates = []
            for index in range(time-1):
                updates.append({'command_index': index, 'completed_at': index+1,
                                'address_hash': tensor_hash(episode['command_addresses'][:, index]),
                                'symbol_hash': tensor_hash(episode['command_symbols'][:, index]),
                                'completed_map_hash': tensor_hash(prefix['actual_completed_w'][:, index+1])})
            provenance.append({'time': time, 'schedule': schedule, 'patched_addresses': address.tolist(),
                               'executor_command_indices': list(range(time)),
                               'permitted_feedback_command_indices': list(range(time-1)),
                               'concealed_completion_hash': tensor_hash(prefix['actual_completed_w'][:, time]),
                               'pre_reveal_mask': [index < time for index in range(time+1)],
                               'post_reveal_mask_not_used_for_donor_state': [True]*(time+1),
                               'updates': updates, 'mapping_indices': prefix['evaluator_mapping_indices'].tolist()})
    return result, provenance


def _replay_donor(source, episode, prefix):
    state = source.initial_state(WORLDS)
    for index in prefix['permitted_feedback_command_indices']:
        completion = index+1
        fb = RevealedFeedback(prefix['actual_completed_w'][:, completion], _command(episode, index),
                              completion, index, completion)
        state = source.update(state, episode['V'], fb)
    return state


def _remap_state(state):
    if isinstance(state, SourceState):
        return replace(state, hidden=state.hidden.roll(1, 1))
    return replace(state, survivors=state.survivors.roll(1, 1),
                   revealed_symbols=state.revealed_symbols.roll(1, 1))


def _normal_distances(normalizer, maps, family):
    # maps[B,...,24,4,4], family[B,...,3,24,4,4].
    return (normalizer.normalize(maps).unsqueeze(-4) - normalizer.normalize(family)).square().mean((-3, -2, -1))


def _paired_bootstrap(source_metrics, seed):
    generator = torch.Generator().manual_seed(350000000 + seed)
    indices = torch.randint(WORLDS, (256, WORLDS), generator=generator)
    result = {'scope': 'conditional fixed source/layout/mapping design; no population significance', 'draws': 256}
    for metric in source_metrics['A']:
        for name in SOURCES:
            value = source_metrics[name][metric].double()
            draws = value[indices].mean(-1)
            result[f'{name}/{metric}'] = {'mean': float(value.mean()),
                'interval': torch.quantile(draws, torch.tensor((.025, .975), dtype=torch.float64)).tolist()}
        for other in ('ordinary', 'analytic'):
            difference = source_metrics['A'][metric].double() - source_metrics[other][metric].double()
            result[f'A_minus_{other}/{metric}'] = {'mean': float(difference.mean()),
                'interval': torch.quantile(difference[indices].mean(-1), torch.tensor((.025, .975), dtype=torch.float64)).tolist()}
    return result, indices


@torch.no_grad()
def evaluate_sources(models, episode, native_head, native_standardizer, extractor, split, log=None, partial=None):
    """Fixed diagnostics only; no fitting, loading, sealing or semantic promotion."""
    progress = {'stage':'before_input_validation'}
    if partial is not None:
        partial['engine_evaluation'] = progress
    v = _validate_episode(episode, split)
    if set(models) != {'A', 'ordinary'} or any(model.source_key != key for key, model in models.items()):
        raise ValueError('the immutable A and independently fitted ordinary sources are required')
    normalizer = models['A'].normalizer
    if extractor is None:
        raise ValueError('registered evaluation requires frozen W-derived control extractor')
    if normalizer != models['ordinary'].normalizer:
        raise ValueError('equal-information sources must share fit-only coordinates')
    torch.set_num_threads(4)
    for model in models.values():
        model.eval()
    sources = {**models, 'analytic': AnalyticMappingSource()}
    weights_before = {name:{key:tensor_hash(value) for key,value in model.state_dict().items()}
                      for name,model in models.items()}
    readout = NativePredictionReadout(native_head, native_standardizer)
    controller = NativeController(readout.head, readout.standardizer)
    rows = torch.arange(WORLDS)
    progress['stage'] = 'before_shared_readouts'
    family = operator_family(v)
    progress['actual_operator_family'] = family
    family_evidence = readout.consume(family)
    actual_evidence = readout.consume(episode['actual_completed_w'])
    source_tensors, source_checks, visibility, per_world_metrics = {}, {}, {}, {}
    progress.update({'source_tensors':source_tensors,'source_checks':source_checks,'visibility':visibility,
                     'actual_evidence':actual_evidence})
    shared_checks, actual_provenance, actual_control_outputs = _controls(controller, episode)
    donors, donor_provenance = _donors(episode)
    actual_actions = {}
    for time in (12, 13, 14):
        addresses, roles = episode['command_addresses'][:, time], episode['query_roles'][:, time]
        for symbol in range(3):
            actual_actions[(time, symbol)] = _native_action(controller, v, episode['actual_forks'][:, time, symbol],
                                                           addresses, roles, time+1)
    analytic_secondary = {}
    for name, source in sources.items():
        state = source.initial_state(WORLDS)
        records, checks, calls = {}, {}, []
        donor_records = {}
        progress.update({'source':name,'in_progress_records':records,'in_progress_checks':checks,
                         'in_progress_calls':calls,'in_progress_donor_records':donor_records})
        for time in range(15):
            obs = _observation(episode, time)
            address = episode['command_addresses'][:, time]
            progress.update({'stage':'before_primary_prediction','decision_time':time,
                             'current_state':snapshot_state(state),'current_observation':_visibility(obs,state)})
            prediction = source.predict(obs, state, address)
            progress.update({'stage':'before_predicted_readout','latest_prediction':prediction})
            calls.append(_visibility(obs, state))
            selected_next = prediction.next[rows[:, None], torch.arange(3)[None, :], address[:, None]]
            next_evidence = readout.consume(selected_next)
            target_next = episode['actual_forks'][rows[:, None], time, torch.arange(3)[None, :], address[:, None]]
            bandwidths = episode['evaluator_mappings'][rows[:, None], address[:, None], torch.arange(3)[None, :]]
            family_indices = torch.where(bandwidths == 1, 0, torch.where(bandwidths == 2, 1, 2))
            actual_next_fields = family_evidence.role_fields[rows[:, None], address[:, None], family_indices]
            cells = episode['evaluator_cells'][rows, address][:, None].expand(-1, 3, -1, -1)
            _append(records, 'selected_next_maps', selected_next)
            _append(records, 'selected_next_logits', next_evidence.local_logits)
            _append(records, 'selected_next_fields', next_evidence.role_fields)
            _append(records, 'next_selected_normalized_mse', (normalizer.normalize(selected_next)-normalizer.normalize(target_next)).square().mean((-3,-2,-1)))
            _append(records, 'next_full_normalized_mse_by_address', (normalizer.normalize(prediction.next)-normalizer.normalize(episode['actual_forks'][:, time])).square().mean((-3,-2,-1)))
            next_family = family[rows, address][:, None].expand(-1, 3, -1, -1, -1, -1)
            _append(records, 'next_distance_to_each_actual_operator', _normal_distances(normalizer, selected_next, next_family))
            for key, value in _field_metrics(next_evidence.role_fields, actual_next_fields, cells).items():
                _append(records, 'next_'+key, value)
            current_fields = actual_evidence.role_fields[:, max(0,time-1)].clone()
            if time:
                previous_address = obs.previous_command.address
                selected_current = prediction.current[rows, previous_address]
                evidence = readout.consume(selected_current)
                current_fields[rows, previous_address] = evidence.role_fields
                target = episode['actual_completed_w'][rows, time, previous_address]
                _append(records, 'selected_current_maps', selected_current)
                _append(records, 'selected_current_logits', evidence.local_logits)
                _append(records, 'selected_current_fields', evidence.role_fields)
                _append(records, 'current_selected_normalized_mse', (normalizer.normalize(selected_current)-normalizer.normalize(target)).square().mean((-3,-2,-1)))
                _append(records, 'current_distance_to_each_actual_operator', _normal_distances(normalizer, selected_current, family[rows, previous_address]))
                for key, value in _field_metrics(evidence.role_fields, actual_evidence.role_fields[rows,time,previous_address], episode['evaluator_cells'][rows,previous_address]).items():
                    _append(records, 'current_'+key, value)
            _append(records, 'current_full_normalized_mse_by_address', (normalizer.normalize(prediction.current)-normalizer.normalize(episode['actual_completed_w'][:,time])).square().mean((-3,-2,-1)))
            assembled_next_fields = current_fields[:, None].expand(-1,3,-1,-1,-1,-1).clone()
            assembled_next_fields[rows[:,None],torch.arange(3)[None,:],address[:,None]] = next_evidence.role_fields
            actual_fork_fields = actual_evidence.role_fields[:,time,None].expand(-1,3,-1,-1,-1,-1).clone()
            actual_fork_fields[rows[:,None],torch.arange(3)[None,:],address[:,None]] = actual_next_fields
            _append(records, 'next_full_field_l1_by_address', (assembled_next_fields-actual_fork_fields).abs().mean((-3,-2,-1)))
            _append(records, 'current_full_field_l1_by_address', (current_fields-actual_evidence.role_fields[:,time]).abs().mean((-3,-2,-1)))
            for label,value in (('next',selected_next),('current',prediction.current)):
                _append(records,label+'_channel_mean',value.mean((-2,-1)))
                _append(records,label+'_channel_std',value.std((-2,-1),correction=0))
                _append(records,label+'_channel_norm',value.square().sum((-2,-1)).sqrt())
            if isinstance(state, SourceState):
                _append(records, 'pre_reveal_hidden', state.hidden)
            else:
                _append(records, 'pre_reveal_survivors', state.survivors)
                _append(records, 'pre_reveal_revealed_symbols', state.revealed_symbols)
                for key, value in state.summary().items():
                    _append(records, key, value)
                weighted_fields = []
                weights = state.survivors[rows,address].double()
                weights /= weights.sum(-1,keepdim=True)
                for symbol in range(3):
                    indices = torch.tensor([(1,2,4).index(mapping[symbol]) for mapping in MAPPINGS])
                    hypotheses = family_evidence.role_fields[rows[:,None], address[:,None], indices[None,:]]
                    weighted_fields.append((hypotheses.double()*weights[:,:,None,None,None]).sum(1).float())
                _append(analytic_secondary, 'expectation_of_H_fields', torch.stack(weighted_fields,1))
            if name == 'A':
                duplicate = ExactAlias(source).predict(obs,state,address)
                calls.append(_visibility(obs,state,'same_weight_alias_duplicate'))
                _append(checks,'alias_current_exact',_per_world_equal(duplicate.current,prediction.current))
                _append(checks,'alias_next_exact',_per_world_equal(duplicate.next,prediction.next))
            if time >= 12:
                roles = episode['query_roles'][:,time]
                choice = readout.choose(prediction,address,roles)
                executed = execute_command(v,episode['actual_completed_w'][:,time],address,choice.symbol,episode['evaluator_mappings'])
                action = _native_action(controller,v,executed,address,roles,time+1)
                expected = episode['actual_forks'][rows,time,choice.symbol]
                _append(checks,'selected_policy_fork_exact',_per_world_equal(executed,expected))
                shared_action_cells = torch.stack([actual_actions[(time,s)]['cell'] for s in range(3)],1)[rows,choice.symbol]
                _append(checks,'selected_policy_native_action_exact',action['cell'] == shared_action_cells)
                _append(checks,'confidence_choice_exact',choice.symbol == choose_from_maxima(choice.maxima))
                truth = episode['evaluator_cells'][rows,address,roles]
                _append(records,'policy_symbol',choice.symbol)
                _append(records,'policy_role_maxima',choice.maxima)
                _append(records,'policy_action_cell',action['cell'])
                _append(records,'policy_action_correct',action['cell'] == truth[:,0]*4+truth[:,1])
                _append(records,'policy_action_tie_count',action['tie_count'])
                _append(records,'policy_actual_role_fields',action['role_fields'])
                if name == 'analytic':
                    expected_fields = analytic_secondary['expectation_of_H_fields'][-1]
                    role_fields = expected_fields[rows[:,None],torch.arange(3)[None,:],roles[:,None]]
                    _append(analytic_secondary,'expectation_of_H_policy_symbol',choose_from_maxima(role_fields.flatten(-2).max(-1).values))
                moved_command = CommandBatch((obs.previous_command.address+1)%4,obs.previous_command.symbol)
                moved_obs = replace(obs,v=v.roll(1,1),last_revealed_w=obs.last_revealed_w.roll(1,1),previous_command=moved_command,observed=obs.observed.roll(1,1))
                moved_prediction = source.predict(moved_obs,_remap_state(state),(address+1)%4)
                calls.append(_visibility(moved_obs,_remap_state(state),'address_remapping'))
                _append(checks,'address_current_equivariance',_per_world_close(moved_prediction.current,prediction.current.roll(1,1)))
                _append(checks,'address_next_equivariance',_per_world_close(moved_prediction.next,prediction.next.roll(1,2)))
                for schedule in ('current','prospective'):
                    prefix = donors[(time,schedule)]
                    donor_state = _replay_donor(source,episode,prefix)
                    saved = snapshot_state(state)
                    progress.update({'stage':'before_donor_patch','donor_schedule':schedule,
                                     'donor_state':snapshot_state(donor_state),'recipient_state':saved})
                    patched = patch_addressed_state(state,donor_state,prefix['patched_addresses'])
                    patched_prediction = source.predict(obs,patched,address)
                    calls.append(_visibility(obs,patched,'donor_'+schedule))
                    patched_choice = readout.choose(patched_prediction,address,roles)
                    restored = restore_state(saved)
                    restored_prediction = source.predict(obs,restored,address)
                    restored_choice = readout.choose(restored_prediction,address,roles)
                    calls.append(_visibility(obs,restored,'restored_'+schedule))
                    for key,value in (('current',_per_world_equal(restored_prediction.current,prediction.current)),
                                      ('next',_per_world_equal(restored_prediction.next,prediction.next)),
                                      ('policy',restored_choice.symbol == choice.symbol),
                                      ('policy_fields',_per_world_equal(restored_choice.candidate_fields,choice.candidate_fields))):
                        _append(checks,'restore_'+schedule+'_'+key+'_exact',value)
                    if isinstance(state,SourceState):
                        state_equal = _per_world_equal(restored.hidden,state.hidden)
                        donor_h = patched.hidden == state.hidden
                        donor_h[rows,prefix['patched_addresses']] = True
                        isolation = donor_h.flatten(1).all(-1)
                    else:
                        state_equal = _per_world_equal(restored.survivors,state.survivors) & _per_world_equal(restored.revealed_symbols,state.revealed_symbols)
                        masks = patched.survivors == state.survivors
                        masks[rows,prefix['patched_addresses']] = True
                        isolation = masks.flatten(1).all(-1)
                    _append(checks,'restore_'+schedule+'_state_exact',state_equal)
                    _append(checks,'patch_'+schedule+'_nonselected_state_exact',isolation)
                    patch_address = prefix['patched_addresses']
                    unchanged_current = (patched_prediction.current == prediction.current).flatten(2).all(-1)
                    unchanged_next = (patched_prediction.next == prediction.next).flatten(3).all(-1).all(1)
                    unchanged_current[rows,patch_address] = True
                    unchanged_next[rows,patch_address] = True
                    _append(checks,'patch_'+schedule+'_nonselected_current_exact',unchanged_current.all(-1))
                    _append(checks,'patch_'+schedule+'_nonselected_next_exact',unchanged_next.all(-1))
                    previous = obs.previous_command.address
                    donor_ledger = donor_records.setdefault(schedule,{})
                    _append(donor_ledger,'selected_current_maps',patched_prediction.current[rows,previous])
                    _append(donor_ledger,'selected_next_maps',patched_prediction.next[rows[:,None],torch.arange(3)[None,:],address[:,None]])
                    _append(donor_ledger,'policy_symbol',patched_choice.symbol)
                    _append(donor_ledger,'policy_role_maxima',patched_choice.maxima)
                    _append(donor_ledger,'policy_candidate_fields',patched_choice.candidate_fields)
                    _append(donor_ledger,'restored_policy_symbol',restored_choice.symbol)
                    _append(donor_ledger,'restored_policy_candidate_fields',restored_choice.candidate_fields)
                    _append(donor_ledger,'restored_selected_current_maps',restored_prediction.current[rows,previous])
                    _append(donor_ledger,'restored_selected_next_maps',restored_prediction.next[rows[:,None],torch.arange(3)[None,:],address[:,None]])
                    selected_action = torch.stack([actual_actions[(time,s)]['cell'] for s in range(3)],1)[rows,patched_choice.symbol]
                    _append(donor_ledger,'policy_action_cell_reference',selected_action)
                    _append(donor_ledger,'current_changed',~_per_world_equal(patched_prediction.current,prediction.current))
                    _append(donor_ledger,'next_changed',~_per_world_equal(patched_prediction.next,prediction.next))
                    _append(donor_ledger,'policy_changed',patched_choice.symbol != choice.symbol)
                    _append(donor_ledger,'action_changed',selected_action != action['cell'])
                    if isinstance(donor_state,SourceState):
                        _append(donor_ledger,'donor_hidden_at_patched_address',donor_state.hidden[rows,patch_address])
                    else:
                        _append(donor_ledger,'donor_survivors_at_patched_address',donor_state.survivors[rows,patch_address])
            if time:
                state = source.update(state,v,_feedback(episode,time))
        # W15 is archived by the evaluator. There is no scored decision15 and
        # no fictitious post-scoring feedback tag or source update at that time.
        tensors = _stack(records)
        tensors['donors'] = {schedule:_stack(values) for schedule,values in donor_records.items()}
        tensors['terminal_state'] = (state.hidden.detach().clone() if isinstance(state,SourceState)
                                     else state.survivors.detach().clone())
        source_tensors[name],source_checks[name],visibility[name] = tensors,_stack(checks),calls
        per_world_metrics[name] = {'current_mse':tensors['current_selected_normalized_mse'].mean(1),
                                  'next_mse':tensors['next_selected_normalized_mse'].mean((1,2)),
                                  'current_field_l1':tensors['current_field_l1'].mean(1),
                                  'next_field_l1':tensors['next_field_l1'].mean((1,2)),
                                  'policy_targeting_accuracy':tensors['policy_action_correct'].float().mean(1)}
        for role_index,role_name in enumerate(('red','blue')):
            role_mask = episode['query_roles'][:,12:] == role_index
            per_world_metrics[name]['policy_'+role_name+'_accuracy'] = (
                tensors['policy_action_correct'].double()*role_mask).sum(1)/role_mask.sum(1)
        _emit(log,{'split':split,'source':name,'worlds':WORLDS,'completed_decisions':15})
    # Input/actual arrays are compacted only after exact reconstruction checks.
    family_matches = []
    for time in range(15):
        addresses = episode['command_addresses'][:,time]
        for symbol in range(3):
            bandwidth = episode['evaluator_mappings'][rows,addresses,symbol]
            index = torch.where(bandwidth==1,0,torch.where(bandwidth==2,1,2))
            predicted = episode['actual_completed_w'][:,time].clone()
            predicted[rows,addresses] = family[rows,addresses,index]
            family_matches.append(_per_world_equal(predicted,episode['actual_forks'][:,time,symbol]))
        actual_next = episode['actual_forks'][rows,time,episode['command_symbols'][:,time]]
        family_matches.append(_per_world_equal(actual_next,episode['actual_completed_w'][:,time+1]))
    shared_checks['actual_family_reconstruction_exact'] = torch.stack(family_matches,1)
    for name,model in models.items():
        source_checks[name]['fitted_weights_unchanged'] = torch.tensor([
            all(tensor_hash(value) == weights_before[name][key] for key,value in model.state_dict().items())])
    original_controls = {}
    for key in ('category_logits','category_probabilities','orientation_logits'):
        if extractor is not None:
            values = [extractor.controls(item.unsqueeze(0))[key].squeeze(0)
                      for item in episode['actual_completed_w'].reshape(-1,24,4,4)]
            original_controls[key] = torch.stack(values).reshape(WORLDS,16,4,*values[0].shape)
    shared = {key:episode[key].detach().clone() for key in ('V','pixels','evaluator_cells','evaluator_mappings',
              'evaluator_mapping_indices','command_addresses','command_symbols','query_roles','world_indices')}
    shared.update({'actual_operator_family':family,'actual_completed_logits':actual_evidence.local_logits,
                   'actual_completed_fields':actual_evidence.role_fields,'actual_family_fields':family_evidence.role_fields,
                   'W_derived_controls':original_controls,
                   'actual_query_action_cells':torch.stack([torch.stack([actual_actions[(t,s)]['cell'] for s in range(3)],1)
                                                           for t in (12,13,14)],1),
                   'actual_completed_channel_mean':episode['actual_completed_w'].mean((-2,-1)),
                   'actual_completed_channel_std':episode['actual_completed_w'].std((-2,-1),correction=0),
                   'actual_completed_channel_norm':episode['actual_completed_w'].square().sum((-2,-1)).sqrt()})
    bootstrap,bootstrap_indices = _paired_bootstrap(per_world_metrics,episode['encoder_seed'])
    shared['bootstrap_indices'] = bootstrap_indices
    shared['actual_controls'] = actual_control_outputs
    all_checks = [value for value in shared_checks.values()] + [value for checks in source_checks.values() for value in checks.values()]
    hard_pass = all(bool(value.all()) for value in all_checks)
    metadata = {'split':split,'encoder_seed':episode['encoder_seed'],'source_indices':list(SOURCES),
                'fit_diagnostics_are_independent_evidence':False if split == 'fit' else None,
                'worlds':WORLDS,'world_namespace':episode.get('world_namespace'),
                'render_nuisance':episode.get('render_nuisance'),
                'source_call_visibility':visibility,'donor_provenance':donor_provenance,
                'actual_control_provenance':actual_provenance,'software_invariants_pass':hard_pass,
                'reconstruction':{'actual_W':'start V; replace previous issued address using actual_operator_family and evaluator mapping',
                                  'current_source':'last revealed actual W; replace previous address with selected_current_maps[t-1]',
                                  'next_source':'source current; replace scheduled address with selected_next_maps[t,s]',
                                  'terminal_feedback':'W15 archived by evaluator; no scored decision15 or source update; source terminal state through revealW14'},
                'denominators':{'worlds_shared':WORLDS,'common_transitions_shared':WORLDS*15,
                    'actual_forks_shared':WORLDS*45,'current_selected_per_source':WORLDS*14,
                    'next_selected_per_source':WORLDS*45,'full_next_item_records_per_source':WORLDS*180,
                    'query_actual_actions_shared':WORLDS*9,'executed_policy_duplicates_per_source':WORLDS*3,
                    'donor_pairs_per_source_per_schedule':WORLDS*3,'donor_transitions_shared':WORLDS*78,
                    'donor_feedback_events_per_source':WORLDS*72,'donor_policy_fork_references_per_source':WORLDS*6},
                'bootstrap':bootstrap,
                'literal_content_disposition':'pending independent fixed-dev evidence assessment',
                'presentation_manner_qualified':False,'reports_generated':False}
    return {'tensors':{'shared':shared,'sources':source_tensors,'analytic_secondary':_stack(analytic_secondary)},
            'metadata':metadata,'software_checks':{'shared':shared_checks,'sources':source_checks}}


def compact_episode(episode):
    """Preserve inputs/actual target references after exact numeric reconstruction.

    Failed reconstruction retains original actual arrays instead of discarding
    their evidence. This helper neither releases a split nor loads any artifact.
    """
    v = _validate_episode(episode,episode['split'])
    family = operator_family(v)
    rows = torch.arange(WORLDS)
    completed = v.clone()
    checks = [_per_world_equal(completed,episode['actual_completed_w'][:,0])]
    for time in range(15):
        address = episode['command_addresses'][:,time]
        reconstructed = []
        for symbol in range(3):
            bandwidth = episode['evaluator_mappings'][rows,address,symbol]
            index = torch.where(bandwidth==1,0,torch.where(bandwidth==2,1,2))
            fork = completed.clone()
            fork[rows,address] = family[rows,address,index]
            checks.append(_per_world_equal(fork,episode['actual_forks'][:,time,symbol]))
            reconstructed.append(fork)
        completed = torch.stack(reconstructed,1)[rows,episode['command_symbols'][:,time]]
        checks.append(_per_world_equal(completed,episode['actual_completed_w'][:,time+1]))
    check_tensor = torch.stack(checks,1)
    result = {key:value for key,value in episode.items() if key not in ('actual_completed_w','actual_forks')}
    result.update({'actual_operator_family':family,'actual_reconstruction_exact':check_tensor,
                   'actual_reconstruction_schema':'attention_content_model_v1_savedV_address_replacement',
                   'actual_reconstruction_recipe':{'initial_completed':'V',
                      'fork':'copy completed[t]; replace command_addresses[t] by actual_operator_family at mapped symbol bandwidth',
                      'common_completed':'fork[t,command_symbols[t]]',
                      'terminal_completed':'common_completed after command14; archive only'}})
    if not bool(check_tensor.all()):
        result['actual_completed_w'] = episode['actual_completed_w']
        result['actual_forks'] = episode['actual_forks']
        result['actual_compaction_status'] = 'failed_reconstruction_raw_first_evidence_retained'
    else:
        result['actual_compaction_status'] = 'exact_reconstruction_bound'
    return result
