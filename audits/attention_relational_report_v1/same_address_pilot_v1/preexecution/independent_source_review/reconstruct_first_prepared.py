"""Method-aware independent source reconstruction; never imports new exporter.

Preparation/import/tests read no target tensors. Actual execution requires root's
published output hashes and --execute, and creates one exclusive first claim.
Original frozen predictors, H, and old native-map assembly APIs are reused.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
DEST = HERE / 'numerical_review'
CORRECTED_REGISTRATION = 'audits/attention_relational_report_v1/same_address_pilot_v1/preexecution/source_registration_operational_correction_v1.json'
CORRECTED_REGISTRATION_SHA = '60a8ffd223ba7d9531c2d9c87076b161bbc747bbfd2459ecdcfedf48681bbdb0'
COMMANDS = ('k0', 'k1', 'k2')
SCORE_RULE = 'maximum supplied localization probability for the queried role; choose first command index on exact ties'
EPS_BOUND = 256 * sys.float_info.epsilon
COUNTS = Counter()
ERRORS = []
MAX_ERROR = {'absolute': 0.0, 'scaled': 0.0, 'path': None}


def canonical(value, newline=False):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + ('\n' if newline else '')).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def jhash(value, newline=False): return hashlib.sha256(canonical(value, newline)).hexdigest()


def thash(value):
    import torch
    v = value.detach().cpu().contiguous()
    return hashlib.sha256(str(v.dtype).encode() + str(tuple(v.shape)).encode() + v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def check(condition, path, detail):
    COUNTS['checks'] += 1
    if not condition: ERRORS.append({'path': path, 'detail': detail})


def compare(actual, expected, path, approximate=False):
    COUNTS['tree_nodes'] += 1
    if hasattr(expected, 'shape'):
        COUNTS['exact_tensors'] += 1
        check(hasattr(actual, 'shape') and type(actual) is type(expected)
              and actual.shape == expected.shape and actual.dtype == expected.dtype
              and thash(actual) == thash(expected), path, 'tensor dtype/shape/raw bytes mismatch')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict): check(False, path, 'object expected'); return
        check(actual.keys() == expected.keys(), path, 'object keys differ')
        for key in expected.keys() & actual.keys():
            compare(actual[key], expected[key], path + '/' + str(key),
                    approximate or key in ('entropy_nats','expected_row','expected_column'))
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected): check(False, path, 'list/tuple type differs'); return
        check(len(actual) == len(expected), path, 'sequence length differs')
        for i, (a,e) in enumerate(zip(actual, expected)): compare(a,e,path+'/'+str(i),approximate)
    elif approximate and type(expected) is float and type(actual) in (int,float):
        COUNTS['float64_leaves'] += 1
        error = abs(actual-expected); scale = max(1.0,abs(actual),abs(expected))
        if error > MAX_ERROR['absolute']: MAX_ERROR.update(absolute=error,path=path)
        MAX_ERROR['scaled'] = max(MAX_ERROR['scaled'],error/scale)
        check(math.isfinite(actual) and error <= EPS_BOUND*scale,path,{'actual':actual,'expected':expected,'error':error,'bound':EPS_BOUND*scale})
    else:
        COUNTS['exact_leaves'] += 1
        check(type(actual) is type(expected) and actual == expected,path,{'actual':actual,'expected':expected})


def location(grid):
    cells=[v for row in grid for v in row]; peak=max(cells)
    return {'peaks':[[r,c] for r in range(4) for c in range(4) if grid[r][c] == peak],
            'peak_probability':peak,'entropy_nats':sum(-v*math.log(v) for v in cells if v > 0),
            'expected_row':sum(grid[r][c]*r for r in range(4) for c in range(4)),
            'expected_column':sum(grid[r][c]*c for r in range(4) for c in range(4))}


def score_values(fields):
    raw=[max(max(row) for row in field) for field in fields]
    winners=[COMMANDS[i] for i in range(3) if raw[i] == max(raw)]
    ordered=sorted(raw)
    return {'rule':SCORE_RULE,'scores':dict(zip(COMMANDS,raw)),'tied_winners':winners,
            'first_index_choice':winners[0],'top_two_margin':ordered[-1]-ordered[-2]}


def origin(source, future=False, changed=False):
    if future:
        result={'kind':'native_conditional_record' if source == 'W' else 'command_estimate','represented_time':13,'executed_branch_claim':False}
        if source != 'W': result['feedback_through']=11
        return result
    if source != 'W': return {'kind':'history_estimate','represented_time':12,'feedback_through':11}
    if changed: return {'kind':'counterfactual_native_snapshot','represented_time':12,'archived_origin_time':13,'executed_branch_claim':False,'fixed_command':'k0'}
    return {'kind':'completed_record','represented_time':12,'record_time':12}


def public_panel(name, source, variant, role, current, future):
    def entry(field,is_future):
        return {'node':'n3','role':role,'origin':origin(source,is_future,variant == 'W_changed'),
                'distribution':field,'derived':location(field)}
    return {'panel':name,'panel_role':'native_reference' if source == 'W' else 'predictor_estimate',
            'current':entry(current,False),'candidates':{c:entry(future[i],True) for i,c in enumerate(COMMANDS)},
            'processing_scores':score_values(future)}


def design():
    rows=[];pairs=[]
    for world in range(4):
        for source in ('A','ordinary','analytic','W'):
            variants=['baseline','source_donor','source_restored','W_changed','W_restored'] if source != 'W' else ['baseline','W_changed','W_restored']
            ids={}
            for variant in variants:
                view='W_only' if source == 'W' else 'joined'
                key=f's1311_w{world:02d}_t12_{source}_{view}_{variant}_same_address_v1';ids[variant]=key
                rows.append({'id':key,'native_split':'heldout','encoder_seed':1311,'world_index':world,'decision_time':12,
                             'previous_node':'n3','candidate_node':'n3','queried_role':('red','blue')[world%2],
                             'source':source,'view':view,'variant':variant})
            edges=[] if source == 'W' else [('source_donor','baseline','source_donor'),('source_donor_to_restored','source_donor','source_restored'),('source_baseline_to_restored','baseline','source_restored')]
            edges += [('W_changed','baseline','W_changed'),('W_changed_to_restored','W_changed','W_restored'),('W_baseline_to_restored','baseline','W_restored')]
            pairs.extend({'kind':kind,'first_id':ids[a],'second_id':ids[b]} for kind,a,b in edges)
    return rows,pairs


def state_payload(state):
    names=('hidden',) if hasattr(state,'hidden') else ('survivors','revealed_symbols')
    return {**{name:getattr(state,name).clone() for name in names},'source_key':state.source_key,
            'revealed_through':state.revealed_through,'update_command_indices':state.update_command_indices}


def state_hashes(state):
    return {k:thash(v) if hasattr(v,'shape') else v for k,v in state_payload(state).items()}


def selected(value):
    if hasattr(value,'shape'): return value[:4].clone() if value.ndim and value.shape[0] == 96 else value.clone()
    if isinstance(value,dict): return {k:selected(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return type(value)(selected(v) for v in value)
    return value


def donor_evidence(evaluation,source,world):
    metadata=evaluation['metadata']; ledger=evaluation['tensors']['sources'][source]['donors']['current']
    rows=[r for r in metadata['donor_provenance'] if r['time'] == 12 and r['schedule'] == 'current']
    check(len(rows) == 1,'donor provenance','unique schedule row'); row=rows[0]
    compare(row['permitted_feedback_command_indices'],list(range(11)),'donor permitted feedback')
    names=['restore_current_current_exact','restore_current_next_exact','restore_current_policy_fields_exact','restore_current_state_exact',
           'patch_current_nonselected_state_exact','patch_current_selected_state_transfer_exact','patch_current_nonselected_current_exact',
           'patch_current_nonselected_next_exact','patch_current_nonselected_decoder_exact','patch_current_next_inheritance_exact']
    for name in names: check(bool(evaluation['software_checks']['sources'][source][name][world,0]),'donor/'+name,'archived donor gate')
    fields=('donor_hidden_at_patched_address',) if source != 'analytic' else ('donor_survivors_at_patched_address','donor_revealed_symbols_at_patched_address')
    state_fields={name:ledger[name][world,0].tolist() for name in fields}
    calls=[r for r in metadata['source_call_visibility'][source] if r['decision_time'] == 12 and r['kind'] in ('primary','donor_current','restored_current')]
    updates=[r for r in metadata['source_update_visibility'][source] if r['kind'] == 'donor_replay' and r['donor_schedule'] == 'current' and r['donor_decision_time'] == 12]
    check(len(calls) == 3 and len(updates) == 11,'donor calls','prediction/update inventory')
    return {'schedule':'current','patched_address':row['patched_addresses'][world],'prefix_receipt_sha256':jhash(row),
            'updates':row['updates'],'executor_command_indices':row['executor_command_indices'],
            'permitted_feedback_command_indices':row['permitted_feedback_command_indices'],'pre_reveal_mask':row['pre_reveal_mask'],
            'mapping_indices':row['mapping_indices'][world],'concealed_completion_sha256':row['concealed_completion_hash'],
            'source_prediction_receipts':calls,'source_update_receipts':updates,'donor_selected_state_sha256':jhash(state_fields),
            'archived_exact_gates':dict.fromkeys(names,True)}


def reconstruct(bundle,fit,native,inventory):
    import torch
    from attention_content_model_v1 import (FitNormalizer,LearnedContentModel,AnalyticMappingSource,SourceState,AnalyticState,
        ObservationBatch,CommandBatch,PredictionBatch,NativePredictionReadout)
    from spatial_native_task_v1 import LocalOccupancyHead,ChannelStandardizer
    from attention_relational_report_v1 import _actual_maps
    episode=bundle['episode'];evaluation=bundle['evaluation'];shared=evaluation['tensors']['shared']
    compare((episode['split'],episode['encoder_seed']),('heldout',1311),'case identity')
    compare(episode['V'].shape[0],96,'execution batch size')
    rows,pairs=design();compare(inventory['opportunities'],rows,'design opportunities');compare(inventory['pairs'],pairs,'design pairs')
    previous=episode['command_addresses'][:,11].clone();symbols=episode['command_symbols'][:,11].clone()
    check(bool((previous == 3).all()),'previous address','all96 n3')
    compare(symbols[:4].tolist(),[0,1,1,0],'registered actual previous symbols')
    roles=episode['query_roles'][:4,12].clone();compare(roles.tolist(),[0,1,0,1],'actual queried roles')
    v=episode['V'].clone();w11=torch.stack([_actual_maps(episode,shared,torch,i,11) for i in range(96)])
    w12=torch.stack([_actual_maps(episode,shared,torch,i,12) for i in range(4)])
    candidate=episode['command_addresses'][:,12].clone();candidate[:4]=3
    obs=ObservationBatch(v,w11,12,11,CommandBatch(previous,symbols),torch.ones(96,4,dtype=torch.bool));obs.validate()
    input_hashes={'V':thash(v),'W11':thash(w11),'previous_address':thash(previous),'previous_symbol':thash(symbols),
                  'observed':thash(obs.observed),'candidate':thash(candidate),'decision_time':12,'last_revealed_at':11}
    private={'observation':{'V':v,'W11':w11,'previous_address':previous.clone(),'previous_symbol':symbols.clone()},
             'native_current_W12':w12,'source_states':{},'native_readouts':{},'source_predictions':{}}
    head=LocalOccupancyHead().float().cpu();head.load_state_dict(native['native_head_fit']['weights'])
    coords=native['native_head_fit']['fit_only_standardizer']
    readout=NativePredictionReadout(head,ChannelStandardizer(tuple(coords['mean']),tuple(coords['scale'])))
    head_hashes={k:thash(t) for k,t in readout.head.state_dict().items()}
    normalizer=FitNormalizer(tuple(fit['normalizer']['mean']),tuple(fit['normalizer']['scale']))
    readouts={};receipts={}
    def h(prediction):
        subset=PredictionBatch(prediction.current[:4],prediction.next[:4],12,prediction.source_key)
        current=readout.consume(subset.current[:,3]);future=readout.consume(subset.next[:,:,3])
        choice=readout.choose(subset,candidate[:4],roles)
        compare(choice.candidate_fields,future.role_fields,'native choose fields')
        result={'current_logits':current.local_logits,'current_fields':current.role_fields,
                'next_logits':future.local_logits,'next_fields':future.role_fields,'choice':choice.symbol,'maxima':choice.maxima}
        for world in range(4):
            independent=score_values(future.role_fields[world,:,int(roles[world])].tolist())
            compare(independent['scores'],dict(zip(COMMANDS,choice.maxima[world].tolist())),'independent native scores')
            compare(independent['first_index_choice'],COMMANDS[int(choice.symbol[world])],'independent tie choice')
        return result
    for key in ('A','ordinary','analytic'):
        archive=evaluation['tensors']['sources'][key];ledger=archive['donors']['current']
        if key == 'analytic':
            source=AnalyticMappingSource()
            recipient=AnalyticState(archive['pre_reveal_survivors'][:,12].clone(),archive['pre_reveal_revealed_symbols'][:,12].clone(),11,tuple(range(11)))
            donor=AnalyticState(recipient.survivors.clone(),recipient.revealed_symbols.clone(),11,tuple(range(11)))
            donor.survivors[:,3]=ledger['donor_survivors_at_patched_address'][:,0]
            donor.revealed_symbols[:,3]=ledger['donor_revealed_symbols_at_patched_address'][:,0]
            restored=AnalyticState(recipient.survivors.clone(),recipient.revealed_symbols.clone(),11,tuple(range(11)))
        else:
            source=LearnedContentModel(normalizer,0,key).eval();source.load_state_dict(fit['checkpoints'][key])
            recipient=SourceState(archive['pre_reveal_hidden'][:,12].clone(),11,tuple(range(11)),key)
            donor=SourceState(recipient.hidden.clone(),11,tuple(range(11)),key);donor.hidden[:,3]=ledger['donor_hidden_at_patched_address'][:,0]
            restored=SourceState(recipient.hidden.clone(),11,tuple(range(11)),key)
        weights={k:thash(t) for k,t in source.state_dict().items()} if key != 'analytic' else {}
        source_progress={'saved_state':state_payload(restored),'donor_state':state_payload(donor),'patched_state':state_payload(donor)}
        private['source_states'][key]=source_progress;private['source_predictions'][key]={};readouts[key]={}
        predictions={}
        for variant,state in [('baseline',recipient),('source_donor',donor),('source_restored',restored)]:
            before=state_hashes(state);prediction=source.predict(obs,state,candidate);predictions[variant]=prediction
            compare(state_hashes(state),before,'predict state immutability')
            compare(thash(obs.v),input_hashes['V'],'predict V fixed');compare(thash(obs.last_revealed_w),input_hashes['W11'],'predict W11 fixed')
            latent=state.hidden[:,3] if key != 'analytic' else state.survivors[:,3]
            compare(prediction.current[:,3],source.decode(v[:,3],latent,symbols),'actual current source ownership')
            compare(prediction.current[:,:3],w11[:,:3],'current nonselected W11')
            saved_current=archive['selected_current_maps'][:,11] if variant != 'source_donor' else ledger['selected_current_maps'][:,0]
            compare(prediction.current[:,3],saved_current,'original96 current map replay')
            compare(prediction.next[torch.arange(4),symbols[:4],3],prediction.current[:4,3],'same-address actual previous symbol/current')
            expected_next=archive['selected_next_maps'][:,12] if variant != 'source_donor' else ledger['selected_next_maps'][:,0]
            compare(prediction.next[3,:,3],expected_next[3],'world3 old next maps')
            compare(prediction.next[torch.arange(4,96)[:,None],torch.arange(3)[None,:],candidate[4:,None]],expected_next[4:],'original92 future replay')
            result=h(prediction);readouts[key][variant]=result;private['native_readouts'][key+'/'+variant]=result
            expected_fields=archive['selected_current_fields'][:4,11] if variant != 'source_donor' else ledger['selected_current_fields'][:4,0]
            compare(result['current_fields'],expected_fields,'old current H replay')
            expected_fields=archive['selected_next_fields'][3,12] if variant != 'source_donor' else ledger['policy_candidate_fields'][3,0]
            compare(result['next_fields'][3],expected_fields,'world3 old next H replay')
            receipt={'inputs':input_hashes,'state':before,'source_key':key,'current':thash(prediction.current),'next':thash(prediction.next),
                     'current_source_decoder_exact':True,'current_completion_visible':False,'fork_outcomes_visible':False}
            maps={'current':prediction.current,'next':prediction.next}
            source_progress[variant]={'state':state_payload(state),'prediction':maps,'receipt':receipt}
            private['source_predictions'][key][variant]=maps
            COUNTS['source_predictions'] += 1
        for attribute in ('current','next'):
            compare(getattr(predictions['baseline'],attribute),getattr(predictions['source_restored'],attribute),'exact saved restoration')
            axis=1 if attribute == 'current' else 2
            compare(getattr(predictions['baseline'],attribute).narrow(axis,0,3),getattr(predictions['source_donor'],attribute).narrow(axis,0,3),'donor native isolation')
        for attribute in ('hidden',) if key != 'analytic' else ('survivors','revealed_symbols'):
            compare(getattr(recipient,attribute)[:,:3],getattr(donor,attribute)[:,:3],'donor state isolation')
        receipts[key]={'weights':weights,'donor_evidence':[donor_evidence(evaluation,key,i) for i in range(4)],
                       'baseline_state':state_hashes(recipient),'donor_state':state_hashes(donor)}
        if key != 'analytic': compare({k:thash(t) for k,t in source.state_dict().items()},weights,'frozen source weights')
    bands=episode['evaluator_mappings'][:4,3]
    family_indices=torch.tensor([[{1:0,2:1,4:2}[int(band)] for band in row] for row in bands])
    alternative=shared['actual_operator_family'][torch.arange(4)[:,None],3,family_indices]
    future=w12[:,None].repeat(1,3,1,1,1,1);future[:,:,3]=alternative
    changed=w12.clone();changed[:,3]=alternative[:,0]
    controls={variant:{'current':current.clone(),'next':future.clone()} for variant,current in [('baseline',w12),('W_changed',changed),('W_restored',w12)]}
    private['W_controls']=controls;private['W_readouts']={};wreadouts={}
    for variant,maps in controls.items():
        result=h(PredictionBatch(maps['current'],maps['next'],12,'W'));wreadouts[variant]=result;private['W_readouts'][variant]=result
        expected=shared['actual_completed_fields'][:4,12,3] if variant != 'W_changed' else shared['actual_family_fields'][torch.arange(4),3,family_indices[:,0]]
        compare(result['current_fields'],expected,'native W current H')
        compare(result['next_fields'],shared['actual_family_fields'][torch.arange(4)[:,None],3,family_indices],'native W future H')
        COUNTS['W_control_predictions'] += 1
    compare({k:thash(t) for k,t in readout.head.state_dict().items()},head_hashes,'frozen H weights')
    packets=[];provenance=[]
    for case in rows:
        world=case['world_index'];source=case['source'];variant=case['variant'];panels=[];private_panels=[]
        panel_specs=[('p0','W',variant)] if source == 'W' else [('p0',source,variant),('p1','W',variant if variant.startswith('W_') else 'baseline')]
        for name,panel_source,panel_variant in panel_specs:
            values=wreadouts[panel_variant] if panel_source == 'W' else readouts[panel_source][panel_variant if panel_variant.startswith('source_') else 'baseline']
            role=int(roles[world]);panels.append(public_panel(name,panel_source,panel_variant,('red','blue')[role],values['current_fields'][world,role].tolist(),values['next_fields'][world,:,role].tolist()))
            private_panels.append({'panel':name,'source_key':panel_source,'current_fields_sha256':thash(values['current_fields'][world]),'next_fields_sha256':thash(values['next_fields'][world])})
            COUNTS['public_panels'] += 1;COUNTS['public_distributions'] += 4
        content={'schema':'same_address_model_report_source_v1_packet','decision_time':12,'selected_node':'n3','queried_role':('red','blue')[world%2],'panels':panels,
                 'coordinate_system':{'rows':4,'columns':4,'row_direction':'down','column_direction':'right','separate_local_grid':True},
                 'field_semantics':'model/readout localization values; not calibrated confidence or physical truth',
                 'conditional_semantics':'possible processing operations; supplied estimates/records, not executed outcomes',
                 'processing_semantics':'native operation-score selection; not Controller targeting or forecast execution'}
        packets.append({'id':case['id'],'content':content});provenance.append({'id':case['id'],'case':case,'panels':private_panels})
    by_id={p['id']:p['content'] for p in packets};pair_checks=[]
    for pair in pairs:
        a=by_id[pair['first_id']];b=by_id[pair['second_id']];equal=a == b
        if pair['kind'].endswith('baseline_to_restored'): check(equal,'pair restoration','public input restored')
        if pair['kind'].startswith('source_'): compare(a['panels'][1],b['panels'][1],'W fixed under source intervention')
        elif len(a['panels']) == 2: compare(a['panels'][0],b['panels'][0],'source fixed under W change')
        pair_checks.append({**pair,'public_input_identical':equal,'null_pair_retained':True})
    expected={'schema':'same_address_model_report_source_v1_export','native_head_weights':head_hashes,
              'execution_batch_context':'original_archived_96_rows','output_world_indices':[0,1,2,3],'torch_num_threads':4,
              'packets':packets,'provenance':provenance,'pairs':pair_checks,'source_receipts':receipts,
              'case_count':4,'opportunity_count':72,'pair_count':84,'reused_exposed_heldout_native_cases':True,
              'fresh_independent_sample':False,'reporter_called':False}
    return expected,selected(private)


def execute(args):
    require_hash=lambda path,expected: check(file_hash(path) == expected,str(path),'SHA-256 mismatch')
    compare(args.registration,CORRECTED_REGISTRATION,'fixed operational-correction registration path')
    compare(args.registration_sha256,CORRECTED_REGISTRATION_SHA,'fixed operational-correction registration hash')
    registration_path=ROOT/args.registration
    require_hash(registration_path,args.registration_sha256)
    reg=json.loads(registration_path.read_text())
    for path,expected in reg['source_sha256'].items(): require_hash(ROOT/path,expected)
    for descriptor in [reg['plan'],reg['inventory'],reg['original_inputs'],*reg['receipts'].values(),*reg['artifacts'].values()]:
        require_hash(ROOT/descriptor['path'],descriptor['sha256'])
    output=ROOT/reg['output_dir']
    descriptors=[('first_exports.json',args.export_sha256),('first_private.pt.gz',args.private_sha256),('first_receipt.json',args.receipt_sha256)]
    for filename,expected in descriptors: require_hash(output/filename,expected)
    receipt=json.loads((output/'first_receipt.json').read_text())
    compare(receipt['schema'],'same_address_model_report_source_v1_receipt','source receipt schema')
    compare({k:receipt[k] for k in ('status','registration_sha256','retry_performed','reporter_called','opportunity_count','pair_count')},
            {'status':'completed','registration_sha256':args.registration_sha256,'retry_performed':False,'reporter_called':False,'opportunity_count':72,'pair_count':84},'source first receipt')
    compare(set(receipt['artifact_sha256']),{'first_claim.json','first_exports.json','neutral_packets.json','pair_index.json','first_private.pt.gz'},'source receipt artifact inventory')
    source_claim=json.loads((output/'first_claim.json').read_text())
    compare({k:v for k,v in source_claim.items() if k != 'started_at_utc'},
            {'schema':'same_address_model_report_source_v1_claim','registration_path':args.registration,
             'registration_sha256':args.registration_sha256,'retry_performed':False},'source first claim')
    for filename,expected in receipt['artifact_sha256'].items(): require_hash(output/filename,expected)
    if ERRORS: raise ValueError('Preload binding errors; refusing numerical deserialization')
    sys.path.insert(0,str(ROOT/'scripts'))
    import torch
    torch.set_num_threads(4)
    def load(path):
        with gzip.open(path,'rb') as stream:return torch.load(stream,weights_only=True,map_location='cpu')
    bundle=load(ROOT/reg['artifacts']['heldout']['path']);fit=load(ROOT/reg['artifacts']['fit']['path'])['fit'];native=load(ROOT/reg['artifacts']['native']['path'])
    check(bundle['fitted_artifact_sha256'] == json.loads((ROOT/reg['receipts']['fit']['path']).read_text())['artifact_sha256'],'heldout fitted identity','saved fit hash inventory')
    inventory=json.loads((ROOT/reg['inventory']['path']).read_text())
    with torch.no_grad():expected,expected_private=reconstruct(bundle,fit,native,inventory)
    del bundle,fit,native
    observed=json.loads((output/'first_exports.json').read_text())
    expected['bindings']=reg
    # Derived scalar roundoff is permitted; each actual JSON self-hash must still
    # bind its exact stored serialized content, independent of that tolerance.
    for packet,wanted in zip(observed['packets'],expected['packets']):
        wanted['content_sha256']=jhash(packet['content'],True)
    compare(observed,expected,'source export')
    actual_private=load(output/'first_private.pt.gz');compare(actual_private,expected_private,'retained private')
    compare(json.loads((output/'neutral_packets.json').read_text()),observed['packets'],'neutral packet artifact')
    compare(json.loads((output/'pair_index.json').read_text()),expected['pairs'],'pair index artifact')
    compare(len(observed['packets']),72,'all72 opportunities');compare(len(observed['pairs']),84,'all84 pairs')
    COUNTS['opportunities']=len(observed['packets']);COUNTS['pairs']=len(observed['pairs'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--registration',required=True);parser.add_argument('--registration-sha256',required=True)
    parser.add_argument('--export-sha256',required=True);parser.add_argument('--private-sha256',required=True);parser.add_argument('--receipt-sha256',required=True)
    args=parser.parse_args();DEST.mkdir(parents=True,exist_ok=True)
    claim={'schema':'same_address_independent_source_reconstruction_claim','started_utc':datetime.now(timezone.utc).isoformat(),
           'calculator_sha256':file_hash(Path(__file__)),'registration_sha256':args.registration_sha256,'published_output_hashes':vars(args),
           'actual_comparison_runs':1,'retry_performed':False,'tolerance':'256eps scaled for derived float64 localization summaries only; everything else exact, including all float32 tensor bytes and source scores'}
    with (DEST/'first_claim.json').open('x') as stream:json.dump(claim,stream,indent=2,sort_keys=True);stream.write('\n')
    t0=time.monotonic();fatal=None
    try: execute(args)
    except BaseException:fatal=traceback.format_exc();print(fatal,flush=True)
    result={'schema':'same_address_independent_source_reconstruction_first_execution','claim':claim,'elapsed_seconds':time.monotonic()-t0,
            'finished_utc':datetime.now(timezone.utc).isoformat(),'passed':not ERRORS and fatal is None,'counts':dict(COUNTS),
            'error_count':len(ERRORS),'errors':ERRORS,'fatal_error':fatal,'maximum_float64_error':MAX_ERROR,
            'independence':'Previously reviewed method/source; no new exporter import. Reused original frozen predictors/H and old native map assembly; independent state/control/public/provenance/inventory construction.',
            'semantic_or_theory_verdict':None}
    with (DEST/'first_execution.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,default=repr);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','counts','error_count','fatal_error','maximum_float64_error')}),flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':main()
