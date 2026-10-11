"""Read-only retained-result summary; outside the frozen execution closure.

No models, renderer, source forward, optimizer, fitting or Reporter are invoked.
Run only after the coordinator preserves and releases the phase outputs.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import torch

SEEDS = (1301, 1311, 1321)
SOURCES = ('A', 'ordinary', 'analytic')
SUFFIXES = ('field_l1','cell_disagreement','estimated_cell_correct','actual_cell_correct',
    'coarse_disagreement','estimated_coarse_correct','actual_coarse_correct',
    'entropy','maxima','top_two_margin','expected_columns','max_tie_count')


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def plain(value):
    if isinstance(value,torch.Tensor):
        return plain(value.detach().cpu().tolist())
    if isinstance(value,dict):
        return {str(key):plain(child) for key,child in value.items()}
    if isinstance(value,(list,tuple)):
        return [plain(child) for child in value]
    if isinstance(value,float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Infinity' if value>0 else '-Infinity')
    return value


def describe(value,retain_values=False):
    """Descriptive reductions only; preserve nonfinite counts and world pairing."""
    numbers = value.detach().cpu().double()
    flat = numbers.flatten()
    finite = flat[torch.isfinite(flat)]
    result = {'shape':list(value.shape),'dtype':str(value.dtype),
        'opportunities':value.numel(),'nonfinite_count':int((~torch.isfinite(flat)).sum()),
        'mean_all_entries':float(flat.mean()),
        'finite_minimum':float(finite.min()) if finite.numel() else None,
        'finite_maximum':float(finite.max()) if finite.numel() else None}
    if numbers.ndim:
        result['per_world_mean'] = numbers.reshape(numbers.shape[0],-1).mean(-1)
    if numbers.ndim>=2:
        # Preserve remaining role/symbol/address axes in the world-mean curve.
        result['per_decision_mean_over_worlds'] = numbers.mean(0)
    if retain_values:
        result['values'] = value
    return plain(result)


def map_errors(retained,addresses):
    current=retained['current_selected_normalized_mse']
    next_value=retained['next_selected_normalized_mse']
    full=retained['next_full_normalized_mse_by_address']
    candidate = addresses[:,:,None,None].expand(-1,-1,3,1)
    mask = torch.ones_like(full,dtype=torch.bool)
    mask.scatter_(-1,candidate,False)
    inherited=full[mask].reshape(full.shape[0],15,3,3)
    return {
        'selected_current_t1_to14':describe(current,True),
        'selected_next_t0_to14':describe(next_value,True),
        'full_current_t1_to14':describe(retained['current_full_normalized_mse_by_address'][:,1:]),
        'full_next_t0_to14':describe(full),
        'inherited_next_t0_to14':describe(inherited),
        'fixed_temporal_windows':{
            'current_calibration_commands_t1_to11':describe(current[:,:11]),
            'current_final_queries_t12_to14':describe(current[:,11:14]),
            'next_calibration_commands_t0_to11':describe(next_value[:,:12]),
            'next_final_queries_t12_to14':describe(next_value[:,12:15])},
        'initial_current_t0_excluded_from_learned_current_means':True}


def family_distances(retained):
    result={'last_axis_order':['T1(V)','T2(V)','T4(V)'],'support_cutoff_used':False}
    for prefix in ('current','next'):
        value=retained[prefix+'_distance_to_each_actual_operator']
        result[prefix]={'all_three_distances':describe(value,True),
            'continuous_nearest_distance':describe(value.min(-1).values,True),
            'mean_distance_by_operator':value.double().reshape(-1,3).mean(0)}
    return plain(result)


def h_diagnostics(retained):
    result={'role_axis_order':['red','blue'],
        'attribution':'estimated means H(source map); actual means H(actual W); correctness uses evaluator truth',
        'forced_localization_is_not_calibrated_presence':True,'selected':{},'full':{}}
    for prefix in ('current','next'):
        selected={}
        for suffix in SUFFIXES:
            value=retained[prefix+'_'+suffix]
            selected[suffix]=describe(value,True)
            if suffix in ('cell_disagreement','estimated_cell_correct','actual_cell_correct',
                           'entropy','maxima','top_two_margin','expected_columns','max_tie_count'):
                selected[suffix]['red_mean']=float(value[...,0].double().mean())
                selected[suffix]['blue_mean']=float(value[...,1].double().mean())
        result['selected'][prefix]=selected
        for suffix in ('field_l1_by_address','cell_disagreement_by_address','coarse_disagreement_by_address'):
            value=retained[prefix+'_full_'+suffix]
            if prefix=='current':
                value=value[:,1:]
            result['full'][prefix+'_'+suffix]=describe(value)
    return plain(result)


def policy_summary(retained,roles):
    correct=retained['policy_action_correct']
    result={'all_query_accuracy':describe(correct,True),
        'symbol':plain(retained['policy_symbol']),
        'role_maxima':describe(retained['policy_role_maxima'],True),
        'actual_action_cell':plain(retained['policy_action_cell']),
        'actual_action_tie_count':describe(retained['policy_action_tie_count'],True),
        'by_role':{}}
    for role,name in enumerate(('red','blue')):
        mask=roles[:,12:]==role
        per_world=(correct.double()*mask).sum(1)/mask.sum(1)
        result['by_role'][name]={'query_count':int(mask.sum()),
            'correct_count':int(correct[mask].sum()),
            'pooled_query_accuracy':float(correct[mask].double().mean()),
            'equal_weight_per_world_role_accuracy':float(per_world.mean()),
            'per_world_role_accuracy':plain(per_world)}
    return plain(result)


def analytic_summary(retained,addresses):
    result={'counts_by_address':{},'scheduled_address':{},
        'ambiguity_definition':'survivor_count > 1; no cases omitted',
        'final_query_window_does_not_guarantee_identification':True}
    for key in ('survivor_count','posterior_entropy','revealed_symbol_count'):
        value=retained[key]
        result['counts_by_address'][key]=describe(value,True)
        selected=value.gather(-1,addresses[:,:,None]).squeeze(-1)
        result['scheduled_address'][key]=describe(selected,True)
    counts=retained['survivor_count']
    selected=counts.gather(-1,addresses[:,:,None]).squeeze(-1)
    result['scheduled_address']['ambiguous']=describe(selected>1,True)
    result['all_address_ambiguity']=describe(counts>1,True)
    result['survivor_histogram_all_addresses']={str(n):int((counts==n).sum()) for n in range(7)}
    result['revealed_symbol_histogram_all_addresses']={str(n):int((retained['revealed_symbol_count']==n).sum()) for n in range(4)}
    result['terminal_revealed_through']=retained['terminal_revealed_through']
    result['terminal_update_command_indices']=list(retained['terminal_update_command_indices'])
    result['terminal_revealed_symbols']=plain(retained['terminal_revealed_symbols'])
    return plain(result)


def hard_inventory(checks):
    if isinstance(checks,dict):
        return {name:hard_inventory(child) for name,child in checks.items()}
    if not isinstance(checks,torch.Tensor) or checks.dtype!=torch.bool:
        raise ValueError('unexpected hard-check archive format')
    return {'shape':list(checks.shape),'dtype':str(checks.dtype),'total':checks.numel(),
        'true_count':int(checks.sum()),'false_count':int((~checks).sum()),
        'values':plain(checks)}


def source_summary(retained,shared):
    donor={}
    for schedule,ledger in retained['donors'].items():
        donor[schedule]={key:describe(value,True) for key,value in ledger.items()
            if (key.endswith('_changed') or key=='coincident_schedule_addresses'
                or '_factual_' in key or '_full_' in key or key in
                ('policy_symbol','policy_role_maxima','policy_action_cell_reference','restored_policy_symbol'))}
    return {'map_errors':map_errors(retained,shared['command_addresses']),
        'family_distances':family_distances(retained),'H_diagnostics':h_diagnostics(retained),
        'policy':policy_summary(retained,shared['query_roles']),
        'donor_all_predetermined_pairs':donor,
        'channel_access_controls':{prefix+'_'+quantity:describe(retained[prefix+'_'+quantity])
            for prefix in ('current','next') for quantity in ('channel_mean','channel_std','channel_norm')},
        'time_indices':{'selected_current_columns':list(range(1,15)),
                        'selected_next_columns':list(range(15)),
                        'policy_and_donor_columns':[12,13,14]},
        'terminal_revealed_through':retained['terminal_revealed_through'],
        'terminal_update_command_indices':list(retained['terminal_update_command_indices'])}


def fit_summary(fitted):
    result={'completed_updates_per_source':fitted['completed_updates_per_source'],
        'recorded_trajectory_is_not_independently_proven':True,
        'losses_are_preupdate_postfit_evaluation_is_after_update300':True,
        'normalizer':fitted['normalizer'],'sources':{}}
    for source in ('A','ordinary'):
        rows=[row for row in fitted['training_log'] if row['source']==source]
        if len(rows)!=300 or [r['update_index'] for r in rows]!=list(range(300)):
            raise ValueError('unexpected retained fixed fit-log inventory')
        result['sources'][source]={'first_preupdate_loss_record':rows[0],
            'last_preupdate_loss_record':rows[-1],'all_300_preupdate_records':rows}
    return plain(result)


def summarize_bundle(bundle,seed,phase):
    evaluation=bundle['evaluation']; metadata=evaluation['metadata']
    if metadata['encoder_seed']!=seed or metadata['split']!=phase:
        raise ValueError('archive source/phase binding mismatch')
    shared=evaluation['tensors']['shared']; sources=evaluation['tensors']['sources']
    if set(sources)!=set(SOURCES):
        raise ValueError('missing or extra retained primary source')
    result={'metadata':{key:metadata[key] for key in (
        'split','encoder_seed','worlds','source_indices','denominators','source_call_inventory',
        'software_invariants_pass','bootstrap','bootstrap_primary_metric_names','donor_summaries',
        'literal_content_disposition','presentation_manner_qualified','reports_generated',
        'fit_diagnostics_are_independent_evidence','reconstruction')},
        'hard_check_inventory':hard_inventory(evaluation['software_checks']),
        'saved_bootstrap_indices':plain(shared['bootstrap_indices']),
        'sources':{name:source_summary(sources[name],shared) for name in SOURCES},
        'analytic_mapping_diagnostics':analytic_summary(sources['analytic'],shared['command_addresses']),
        'analytic_expectation_of_H_secondary':{},
        'W_derived_controls':{key:describe(value) for key,value in shared['W_derived_controls'].items()},
        'actual_W_channel_access_controls':{quantity:describe(shared['actual_completed_'+quantity])
            for quantity in ('channel_mean','channel_std','channel_norm')}}
    secondary=evaluation['tensors']['analytic_secondary']
    choices=secondary['expectation_of_H_policy_symbol']
    actual_cells=shared['actual_query_action_cells'].gather(-1,choices[:,:,None]).squeeze(-1)
    query_addresses=shared['command_addresses'][:,12:]
    query_roles=shared['query_roles'][:,12:]
    rows=torch.arange(choices.shape[0])[:,None]
    truth=shared['evaluator_cells'][rows,query_addresses,query_roles]
    secondary_correct=actual_cells==truth[...,0]*4+truth[...,1]
    result['analytic_expectation_of_H_secondary']={
        'symbol':plain(choices),'actual_action_cell_from_saved_fork':plain(actual_cells),
        'actual_targeting_accuracy':describe(secondary_correct,True),
        'field_difference_from_primary_H_of_mean_map':describe(
            (secondary['expectation_of_H_fields']-sources['analytic']['selected_next_fields']).abs().mean((-3,-2,-1))),
        'new_actual_execution_or_independent_worlds':False}
    if phase=='fit':
        result['fit']=fit_summary(bundle['fit'])
    else:
        result['fitted_artifact_sha256']=bundle['fitted_artifact_sha256']
    return plain(result)


def summarize_phase(phase_dir):
    phase_dir=Path(phase_dir).resolve()
    receipt_path=phase_dir/'receipt.json'
    receipt_bytes=receipt_path.read_bytes()
    receipt=json.loads(receipt_bytes)
    phase=receipt['phase']
    if phase not in ('fit','dev','heldout') or receipt['status']!='completed':
        raise ValueError('summary requires a retained completed phase receipt')
    result={'schema':'attention_content_model_v1_read_only_phase_summary_v1',
        'phase':phase,'receipt_path':str(receipt_path),'receipt_sha256':sha256_bytes(receipt_bytes),
        'receipt':receipt,'summary_source_sha256':sha256_bytes(Path(__file__).read_bytes()),
        'nonfinite_number_encoding':['NaN','+Infinity','-Infinity'],
        'interpretation':{'new_thresholds_or_case_selection':False,
            'bootstrap_scope':'per-encoder fixed-source paired worlds, not population significance',
            'fit_evaluation_is_fitting_diagnostics':phase=='fit',
            'semantic_or_manner_qualification_asserted_by_summary':False},'seeds':{}}
    for seed in SEEDS:
        path=phase_dir/f'states-{seed}.pt.gz'
        matches=[key for key in receipt['artifact_sha256'] if Path(key).name==path.name]
        if len(matches)!=1:
            raise ValueError('ambiguous or missing archive receipt binding')
        # Load exactly the compressed bytes whose digest is reported and checked.
        compressed=path.read_bytes(); digest=sha256_bytes(compressed)
        if digest!=receipt['artifact_sha256'][matches[0]]:
            raise ValueError('retained archive SHA does not match its phase receipt')
        with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode='rb') as stream:
            bundle=torch.load(stream,map_location='cpu',weights_only=True)
        result['seeds'][str(seed)]={'archive_path':str(path),'receipt_artifact_key':matches[0],
            'archive_sha256':digest,'results':summarize_bundle(bundle,seed,phase)}
        del bundle,compressed
    if receipt_path.read_bytes()!=receipt_bytes:
        raise ValueError('phase receipt changed during read-only summarization')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase_dir',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('exclusive summary output already exists')
    result=summarize_phase(args.phase_dir)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(plain(result),stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__=='__main__':
    main()
