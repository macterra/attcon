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


def _state_hashes(state):
    if isinstance(state,SourceState):
        return {'hidden':tensor_hash(state.hidden)}
    return {'survivors':tensor_hash(state.survivors),
            'revealed_symbols':tensor_hash(state.revealed_symbols)}


def _visibility(observation, state, candidate_address, kind='primary'):
    return {'kind': kind, 'decision_time': observation.decision_time,
            'last_revealed_at': observation.last_revealed_at,
            'source_revealed_through': state.revealed_through,
            'update_command_indices': list(state.update_command_indices),
            'source_key':state.source_key,'state_hashes':_state_hashes(state),
            'candidate_address_hash':tensor_hash(candidate_address),
            'V_hash': tensor_hash(observation.v),
            'last_revealed_W_hash': tensor_hash(observation.last_revealed_w),
            'observed_mask_hash': tensor_hash(observation.observed),
            'previous_address_hash': tensor_hash(observation.previous_command.address) if observation.previous_command else None,
            'previous_symbol_hash': tensor_hash(observation.previous_command.symbol) if observation.previous_command else None,
            'current_completion_visible': observation.decision_time == 0,
            'fork_outcomes_visible': False}


def _predict_logged(source,observation,state,candidate_address,calls,kind='primary'):
    record = _visibility(observation,state,candidate_address,kind)
    record['call_index'] = len(calls)
    calls.append(record)
    prediction = source.predict(observation,state,candidate_address)
    record.update({'completed':True,'post_call_state_hashes':_state_hashes(state),
                   'current_output_hash':tensor_hash(prediction.current),
                   'next_output_hash':tensor_hash(prediction.next),
                   'post_call_input_hashes':{key:value for key,value in
                       _visibility(observation,state,candidate_address,kind).items()
                       if key in ('V_hash','last_revealed_W_hash','observed_mask_hash',
                                  'candidate_address_hash','previous_address_hash','previous_symbol_hash')}})
    return prediction


def _update_logged(source,state,v,feedback,calls,kind,donor_schedule=None,donor_decision=None):
    record = {'kind':kind,'source_key':state.source_key,'call_index':len(calls),
        'pre_revealed_through':state.revealed_through,
        'pre_update_command_indices':list(state.update_command_indices),
        'pre_state_hashes':_state_hashes(state),'V_hash':tensor_hash(v),
        'completed_W_hash':tensor_hash(feedback.completed_w),
        'completed_at':feedback.completed_at,'command_index':feedback.command_index,
        'scored_decision_at':feedback.scored_decision_at,
        'command_address_hash':tensor_hash(feedback.command.address),
        'command_symbol_hash':tensor_hash(feedback.command.symbol),
        'observed_mask_hash':tensor_hash(torch.ones(v.shape[0],4,dtype=torch.bool)),
        'pre_reveal_mask':[index<=state.revealed_through for index in range(feedback.completed_at+1)],
        'post_reveal_mask':[True]*(feedback.completed_at+1),
        'donor_schedule':donor_schedule,'donor_decision_time':donor_decision}
    calls.append(record)
    updated = source.update(state,v,feedback)
    record.update({'completed':True,'post_state_hashes':_state_hashes(updated),
        'post_revealed_through':updated.revealed_through,
        'post_update_command_indices':list(updated.update_command_indices)})
    return updated


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
                'training_log':training_log,'fit_visibility':visibility,'completed_updates':{},
                'source_states':{name:{'checkpoint':model.state_dict(),'optimizer':None,
                    'completed_updates':0,'status':'not_started'} for name,model in models.items()}}
    if partial is not None:
        partial['engine_fit'] = progress
    rows = torch.arange(WORLDS)
    for name, model in models.items():
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999),
                                     eps=1e-8, weight_decay=0, amsgrad=False, foreach=False)
        source_progress = progress['source_states'][name]
        source_progress.update({'status':'active','checkpoint':model.state_dict(),
                                'optimizer':optimizer.state_dict()})
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
                                 'current_observation':_visibility(obs,state,address)})
                prediction = model.predict(obs, state, address)
                if update == 0:
                    source_visibility.append(_visibility(obs, state,address))
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
                source_progress.update({'status':'failed','checkpoint':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'gradients':{key:p.grad for key,p in model.named_parameters()}})
                progress.update({'stage':'failed_optimizer_or_backward',
                    'live_checkpoint':model.state_dict(),'live_optimizer':optimizer.state_dict(),
                    'live_gradients':{key:parameter.grad for key,parameter in model.named_parameters()}})
                raise
            progress['completed_updates'][name] = update+1
            source_progress.update({'checkpoint':model.state_dict(),'optimizer':optimizer.state_dict(),
                                    'completed_updates':update+1})
            record = {'source': name, 'update_index': update, 'completed_updates': update+1,
                      'loss': float(loss.detach()), 'current_loss': float(current_loss.detach()),
                      'next_loss': float(next_loss.detach()), 'preclip_gradient_norm': float(gradient_norm)}
            training_log.append(record); _emit(log, record)
        model.eval()
        terminal[name] = deepcopy(optimizer.state_dict())
        source_progress.update({'status':'completed','checkpoint':deepcopy(model.state_dict()),
                                'optimizer':terminal[name],'completed_updates':300})
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


def _field_scalars(fields):
    flat = fields.flatten(-2)
    top = flat.topk(2,dim=-1).values
    _order,columns = _coarse(fields)
    return {'entropy':-(flat.double()*flat.double().clamp_min(1e-300).log()).sum(-1),
            'maxima':top[...,0],'top_two_margin':top[...,0]-top[...,1],
            'expected_columns':columns}


def _donor_field_diagnostics(ledger,patched_prediction,prediction,current_fields,assembled_next_fields,
                             previous,address,readout,actual_evidence,actual_next_fields,episode,time,normalizer):
    rows = torch.arange(WORLDS)
    patched_current_evidence = readout.consume(patched_prediction.current[rows,previous])
    patched_next_evidence = readout.consume(patched_prediction.next[rows[:,None],torch.arange(3)[None,:],address[:,None]])
    patched_current_fields = current_fields.clone()
    patched_current_fields[rows,previous] = patched_current_evidence.role_fields
    patched_next_fields = patched_current_fields[:,None].expand(-1,3,-1,-1,-1,-1).clone()
    patched_next_fields[rows[:,None],torch.arange(3)[None,:],address[:,None]] = patched_next_evidence.role_fields
    actual_current_fields = actual_evidence.role_fields[:,time]
    actual_full_next = actual_current_fields[:,None].expand(-1,3,-1,-1,-1,-1).clone()
    actual_full_next[rows[:,None],torch.arange(3)[None,:],address[:,None]] = actual_next_fields
    _append(ledger,'selected_current_fields',patched_current_evidence.role_fields)
    _append(ledger,'current_full_field_changed',~_per_world_equal(patched_current_fields,current_fields))
    _append(ledger,'next_full_field_changed',~_per_world_equal(patched_next_fields,assembled_next_fields))
    for label,patched,baseline,actual in (
            ('current',patched_current_fields,current_fields,actual_current_fields),
            ('next',patched_next_fields,assembled_next_fields,actual_full_next)):
        _append(ledger,label+'_full_field_l1_change_by_address',(patched-baseline).abs().mean((-3,-2,-1)))
        _append(ledger,label+'_full_field_l1_to_fixed_W_by_address',(patched-actual).abs().mean((-3,-2,-1)))
        before,after = _field_scalars(baseline),_field_scalars(patched)
        for key in before:
            _append(ledger,label+'_full_'+key+'_change',after[key]-before[key])
        _append(ledger,label+'_full_cell_disagreement_with_fixed_W',patched.flatten(-2).argmax(-1) != actual.flatten(-2).argmax(-1))
        _append(ledger,label+'_full_coarse_disagreement_with_fixed_W',_coarse(patched)[0] != _coarse(actual)[0])
    for label,estimated,actual,cells in (
            ('current',patched_current_evidence.role_fields,actual_current_fields[rows,previous],episode['evaluator_cells'][rows,previous]),
            ('next',patched_next_evidence.role_fields,actual_next_fields,episode['evaluator_cells'][rows,address][:,None].expand(-1,3,-1,-1))):
        for key,value in _field_metrics(estimated,actual,cells).items():
            _append(ledger,label+'_factual_'+key,value)
    _append(ledger,'current_full_normalized_mse_to_fixed_W_by_address',
            (normalizer.normalize(patched_prediction.current)-normalizer.normalize(episode['actual_completed_w'][:,time])).square().mean((-3,-2,-1)))
    _append(ledger,'next_full_normalized_mse_to_fixed_W_by_address',
            (normalizer.normalize(patched_prediction.next)-normalizer.normalize(episode['actual_forks'][:,time])).square().mean((-3,-2,-1)))


def _donor_summaries(ledgers):
    result = {}
    for schedule,ledger in ledgers.items():
        summary = {'all_predetermined_pairs':WORLDS*3,'changed_only_subset_used':False,
                   'coincident_address_pairs':int(ledger['coincident_schedule_addresses'].sum())}
        for key,value in ledger.items():
            if key.endswith('_changed') or key.endswith('_disagreement_with_fixed_W'):
                summary[key+'_fraction_all_pairs'] = float(value.double().mean())
            elif '_l1_' in key or key.endswith('_mse_to_fixed_W_by_address'):
                summary[key+'_mean_all_pairs'] = float(value.double().mean())
            elif '_full_' in key and key.endswith('_change'):
                summary[key+'_mean_abs_all_pairs'] = float(value.double().abs().mean())
            if key.endswith('_disagreement_with_fixed_W') and 'cell_' in key:
                summary[key+'_red_fraction'] = float(value[...,0].double().mean())
                summary[key+'_blue_fraction'] = float(value[...,1].double().mean())
        result[schedule] = summary
    return result


def _world_primary_metrics(tensors,episode):
    metrics = {'current_mse':tensors['current_selected_normalized_mse'].mean(1),
        'next_mse':tensors['next_selected_normalized_mse'].mean((1,2)),
        'current_full_mse':tensors['current_full_normalized_mse_by_address'][:,1:].mean((1,2)),
        'next_full_mse':tensors['next_full_normalized_mse_by_address'].mean((1,2,3)),
        'current_field_l1':tensors['current_field_l1'].mean(1),
        'next_field_l1':tensors['next_field_l1'].mean((1,2)),
        'current_full_field_l1':tensors['current_full_field_l1_by_address'][:,1:].mean((1,2)),
        'next_full_field_l1':tensors['next_full_field_l1_by_address'].mean((1,2,3)),
        'current_coarse_disagreement':tensors['current_coarse_disagreement'].double().mean(1),
        'next_coarse_disagreement':tensors['next_coarse_disagreement'].double().mean((1,2)),
        'current_full_coarse_disagreement':tensors['current_full_coarse_disagreement_by_address'][:,1:].double().mean((1,2)),
        'next_full_coarse_disagreement':tensors['next_full_coarse_disagreement_by_address'].double().mean((1,2,3)),
        'current_full_cell_disagreement':tensors['current_full_cell_disagreement_by_address'][:,1:].double().mean((1,2,3)),
        'next_full_cell_disagreement':tensors['next_full_cell_disagreement_by_address'].double().mean((1,2,3,4)),
        'policy_targeting_accuracy':tensors['policy_action_correct'].float().mean(1)}
    inherited = tensors['next_full_normalized_mse_by_address'].clone()
    row = torch.arange(WORLDS)[:,None,None]
    time = torch.arange(15)[None,:,None]
    symbol = torch.arange(3)[None,None,:]
    inherited[row,time,symbol,episode['command_addresses'][:,:,None]] = 0
    metrics['next_inherited_mse'] = inherited.sum((1,2,3))/(15*3*3)
    for role,role_name in enumerate(('red','blue')):
        for quantity in ('cell_disagreement','estimated_cell_correct','actual_cell_correct'):
            metrics['current_'+role_name+'_'+quantity] = tensors['current_'+quantity][...,role].double().mean(1)
            metrics['next_'+role_name+'_'+quantity] = tensors['next_'+quantity][...,role].double().mean((1,2))
        metrics['current_full_'+role_name+'_cell_disagreement'] = tensors['current_full_cell_disagreement_by_address'][:,1:,...,role].double().mean((1,2))
        metrics['next_full_'+role_name+'_cell_disagreement'] = tensors['next_full_cell_disagreement_by_address'][...,role].double().mean((1,2,3))
        mask = episode['query_roles'][:,12:] == role
        metrics['policy_'+role_name+'_accuracy'] = (tensors['policy_action_correct'].double()*mask).sum(1)/mask.sum(1)
    metrics['current_cell_disagreement'] = tensors['current_cell_disagreement'].double().mean((1,2))
    metrics['next_cell_disagreement'] = tensors['next_cell_disagreement'].double().mean((1,2,3))
    return metrics


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


def _controls(controller, episode, partial=None):
    checks, provenance, outputs = {}, [], {}
    if partial is not None:
        partial.update({'actual_control_checks_in_progress':checks,
                        'actual_control_provenance':provenance,'actual_control_outputs_in_progress':outputs})
    v = episode['V']
    for time in (12, 13, 14):
        rows, emitted = [], {}
        if partial is not None:
            partial.update({'actual_control_time_in_progress':time,
                'actual_control_world_checks':rows,'actual_control_world_outputs':emitted})
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


def _donors(episode, partial=None):
    result, provenance = {}, []
    if partial is not None:
        partial.update({'shared_donor_prefixes':result,'donor_provenance':provenance})
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


def _replay_donor(source, episode, prefix, calls, schedule):
    state = source.initial_state(WORLDS)
    for index in prefix['permitted_feedback_command_indices']:
        completion = index+1
        fb = RevealedFeedback(prefix['actual_completed_w'][:, completion], _command(episode, index),
                              completion, index, completion)
        state = _update_logged(source,state,episode['V'],fb,calls,'donor_replay',
                               schedule,prefix['decision_time'])
    return state


def _decoder_probes(source,v,state,records,time,schedule):
    batch = v.shape[0]
    selected = v.reshape(batch*4,24,4,4)
    latent = (state.hidden.reshape(batch*4,48) if isinstance(state,SourceState)
              else state.survivors.reshape(batch*4,6))
    outputs = []
    for symbol in range(3):
        record = {'kind':'address_local_decoder_probe','source_key':state.source_key,
            'decision_time':time,'donor_schedule':schedule,'symbol':symbol,
            'state_hashes':_state_hashes(state),'V_hash':tensor_hash(v),
            'addresses':'all four, flattened within each world','completed':False}
        records.append(record)
        output = source.decode(selected,latent,torch.full((batch*4,),symbol,dtype=torch.long))
        record.update({'completed':True,'output_hash':tensor_hash(output),
                       'output_shape':list(output.shape)})
        outputs.append(output.reshape(batch,4,24,4,4))
    return torch.stack(outputs,2)


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
    source_tensors, source_checks, visibility, update_visibility, decoder_visibility, per_world_metrics = {}, {}, {}, {}, {}, {}
    progress.update({'source_tensors':source_tensors,'source_checks':source_checks,'visibility':visibility,
                     'actual_evidence':actual_evidence})
    shared_checks, actual_provenance, actual_control_outputs = _controls(controller, episode,progress)
    progress.update({'shared_checks':shared_checks,'actual_control_outputs':actual_control_outputs})
    donors, donor_provenance = _donors(episode,progress)
    actual_actions = {}
    for time in (12, 13, 14):
        addresses, roles = episode['command_addresses'][:, time], episode['query_roles'][:, time]
        for symbol in range(3):
            actual_actions[(time, symbol)] = _native_action(controller, v, episode['actual_forks'][:, time, symbol],
                                                           addresses, roles, time+1)
    analytic_secondary = {}
    for name, source in sources.items():
        state = source.initial_state(WORLDS)
        records, checks, calls, updates, decoder_calls = {}, {}, [], [], []
        donor_records = {}
        progress.update({'source':name,'in_progress_records':records,'in_progress_checks':checks,
                         'in_progress_calls':calls,'in_progress_donor_records':donor_records})
        progress.update({'in_progress_update_calls':updates,'in_progress_decoder_calls':decoder_calls})
        for time in range(15):
            obs = _observation(episode, time)
            address = episode['command_addresses'][:, time]
            progress.update({'stage':'before_primary_prediction','decision_time':time,
                             'current_state':snapshot_state(state),'current_observation':_visibility(obs,state,address)})
            prediction = _predict_logged(source,obs,state,address,calls)
            progress.update({'stage':'before_predicted_readout','latest_prediction':prediction})
            call = calls[-1]
            _append(checks,'observation_history_exact',torch.full((WORLDS,),
                obs.decision_time == time and obs.last_revealed_at == max(0,time-1)
                and state.revealed_through == max(0,time-1)
                and state.update_command_indices == tuple(range(max(0,time-1)))
                and bool(obs.observed.all()) and torch.equal(obs.v,v)
                and (obs.previous_command is None if time == 0 else
                     torch.equal(obs.previous_command.address,_command(episode,time-1).address)
                     and torch.equal(obs.previous_command.symbol,_command(episode,time-1).symbol)),dtype=torch.bool))
            _append(checks,'last_revealed_snapshot_exact',_per_world_equal(obs.last_revealed_w,episode['actual_completed_w'][:,max(0,time-1)]))
            _append(checks,'prediction_inputs_unchanged_exact',torch.full((WORLDS,),
                all(call[key] == value for key,value in call['post_call_input_hashes'].items()),dtype=torch.bool))
            _append(checks,'prediction_state_unchanged_exact',torch.full((WORLDS,),call['state_hashes'] == call['post_call_state_hashes'],dtype=torch.bool))
            if time == 0:
                _append(checks,'initial_current_exact_v',_per_world_equal(prediction.current,v))
            else:
                copies = (prediction.current == obs.last_revealed_w).flatten(2).all(-1)
                copies[rows,obs.previous_command.address] = True
                _append(checks,'current_nonselected_copy_exact',copies.all(-1))
            inheritance = (prediction.next == prediction.current[:,None]).flatten(3).all(-1)
            inheritance[rows[:,None],torch.arange(3)[None,:],address[:,None]] = True
            _append(checks,'next_noncandidate_inheritance_exact',inheritance.flatten(1).all(-1))
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
            _append(records,'current_full_cell_disagreement_by_address',current_fields.flatten(-2).argmax(-1) != actual_evidence.role_fields[:,time].flatten(-2).argmax(-1))
            _append(records,'next_full_cell_disagreement_by_address',assembled_next_fields.flatten(-2).argmax(-1) != actual_fork_fields.flatten(-2).argmax(-1))
            _append(records,'current_full_coarse_disagreement_by_address',_coarse(current_fields)[0] != _coarse(actual_evidence.role_fields[:,time])[0])
            _append(records,'next_full_coarse_disagreement_by_address',_coarse(assembled_next_fields)[0] != _coarse(actual_fork_fields)[0])
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
                duplicate = _predict_logged(ExactAlias(source),obs,state,address,calls,'same_weight_alias_duplicate')
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
                moved_prediction = _predict_logged(source,moved_obs,_remap_state(state),(address+1)%4,calls,'address_remapping')
                _append(checks,'address_current_equivariance',_per_world_close(moved_prediction.current,prediction.current.roll(1,1)))
                _append(checks,'address_next_equivariance',_per_world_close(moved_prediction.next,prediction.next.roll(1,2)))
                baseline_probes = _decoder_probes(source,v,state,decoder_calls,time,'baseline')
                for schedule in ('current','prospective'):
                    prefix = donors[(time,schedule)]
                    donor_state = _replay_donor(source,episode,prefix,updates,schedule)
                    saved = snapshot_state(state)
                    progress.update({'stage':'before_donor_patch','donor_schedule':schedule,
                                     'donor_state':snapshot_state(donor_state),'recipient_state':saved})
                    patched = patch_addressed_state(state,donor_state,prefix['patched_addresses'])
                    patched_prediction = _predict_logged(source,obs,patched,address,calls,'donor_'+schedule)
                    patched_probes = _decoder_probes(source,v,patched,decoder_calls,time,schedule)
                    probe_equal = (patched_probes == baseline_probes).flatten(3).all(-1).all(-1)
                    probe_equal[rows,prefix['patched_addresses']] = True
                    _append(checks,'patch_'+schedule+'_nonselected_decoder_exact',probe_equal.all(-1))
                    patched_inheritance = (patched_prediction.next == patched_prediction.current[:,None]).flatten(3).all(-1)
                    patched_inheritance[rows[:,None],torch.arange(3)[None,:],address[:,None]] = True
                    _append(checks,'patch_'+schedule+'_next_inheritance_exact',patched_inheritance.flatten(1).all(-1))
                    patched_choice = readout.choose(patched_prediction,address,roles)
                    restored = restore_state(saved)
                    restored_prediction = _predict_logged(source,obs,restored,address,calls,'restored_'+schedule)
                    restored_choice = readout.choose(restored_prediction,address,roles)
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
                        selected_transfer = _per_world_equal(patched.hidden[rows,prefix['patched_addresses']],donor_state.hidden[rows,prefix['patched_addresses']])
                    else:
                        state_equal = _per_world_equal(restored.survivors,state.survivors) & _per_world_equal(restored.revealed_symbols,state.revealed_symbols)
                        masks = patched.survivors == state.survivors
                        masks[rows,prefix['patched_addresses']] = True
                        seen_masks = patched.revealed_symbols == state.revealed_symbols
                        seen_masks[rows,prefix['patched_addresses']] = True
                        isolation = masks.flatten(1).all(-1) & seen_masks.flatten(1).all(-1)
                        selected_transfer = (_per_world_equal(patched.survivors[rows,prefix['patched_addresses']],donor_state.survivors[rows,prefix['patched_addresses']])
                            & _per_world_equal(patched.revealed_symbols[rows,prefix['patched_addresses']],donor_state.revealed_symbols[rows,prefix['patched_addresses']]))
                    _append(checks,'restore_'+schedule+'_state_exact',state_equal)
                    _append(checks,'patch_'+schedule+'_nonselected_state_exact',isolation)
                    _append(checks,'patch_'+schedule+'_selected_state_transfer_exact',selected_transfer)
                    patch_address = prefix['patched_addresses']
                    unchanged_current = (patched_prediction.current == prediction.current).flatten(2).all(-1)
                    unchanged_next = (patched_prediction.next == prediction.next).flatten(3).all(-1).all(1)
                    unchanged_current[rows,patch_address] = True
                    unchanged_next[rows,patch_address] = True
                    _append(checks,'patch_'+schedule+'_nonselected_current_exact',unchanged_current.all(-1))
                    _append(checks,'patch_'+schedule+'_nonselected_next_exact',unchanged_next.all(-1))
                    previous = obs.previous_command.address
                    donor_ledger = donor_records.setdefault(schedule,{})
                    _append(donor_ledger,'coincident_schedule_addresses',previous == address)
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
                    _donor_field_diagnostics(donor_ledger,patched_prediction,prediction,current_fields,
                        assembled_next_fields,previous,address,readout,actual_evidence,actual_next_fields,
                        episode,time,normalizer)
                    if isinstance(donor_state,SourceState):
                        _append(donor_ledger,'donor_hidden_at_patched_address',donor_state.hidden[rows,patch_address])
                    else:
                        _append(donor_ledger,'donor_survivors_at_patched_address',donor_state.survivors[rows,patch_address])
                        _append(donor_ledger,'donor_revealed_symbols_at_patched_address',donor_state.revealed_symbols[rows,patch_address])
            if time:
                state = _update_logged(source,state,v,_feedback(episode,time),updates,'primary_feedback')
        # W15 is archived by the evaluator. There is no scored decision15 and
        # no fictitious post-scoring feedback tag or source update at that time.
        tensors = _stack(records)
        tensors['donors'] = {schedule:_stack(values) for schedule,values in donor_records.items()}
        tensors['terminal_state'] = (state.hidden.detach().clone() if isinstance(state,SourceState)
                                     else state.survivors.detach().clone())
        if isinstance(state,AnalyticState):
            tensors['terminal_revealed_symbols'] = state.revealed_symbols.detach().clone()
        tensors['terminal_revealed_through'] = state.revealed_through
        tensors['terminal_update_command_indices'] = state.update_command_indices
        source_tensors[name],source_checks[name],visibility[name] = tensors,_stack(checks),calls
        update_visibility[name],decoder_visibility[name] = updates,decoder_calls
        per_world_metrics[name] = _world_primary_metrics(tensors,episode)
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
    compact = compact_episode(episode)
    shared_checks['actual_compaction_reconstruction_exact'] = compact['actual_reconstruction_exact']
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
                'source_call_visibility':visibility,'source_update_visibility':update_visibility,
                'source_decoder_visibility':decoder_visibility,'donor_provenance':donor_provenance,
                'source_call_inventory':{name:{'prediction_calls':len(visibility[name]),
                    'primary_update_calls':sum(call['kind']=='primary_feedback' for call in update_visibility[name]),
                    'donor_update_calls':sum(call['kind']=='donor_replay' for call in update_visibility[name]),
                    'address_local_decoder_probe_calls':len(decoder_visibility[name])} for name in SOURCES},
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
                'bootstrap_primary_metric_names':list(per_world_metrics['A']),
                'donor_summaries':{name:_donor_summaries(source_tensors[name]['donors']) for name in SOURCES},
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
