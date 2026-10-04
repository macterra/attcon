"""Prepare/export/replay three-way visual records without generating prose."""
import argparse
import copy
import gzip
import io
import itertools
import json
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder, scenes
from attcon.predictive_attention import PredictiveAttention, Forecast
from attcon.functional_controls import CONDITIONS, WINDOWS, last_forecast, advance_model, counterfactual_recovery
from attcon.functional_interface import record, remap, without_relation
from attcon.functional_model import buffer_contents
from run_functional_controls import ROOT as PARENT, SEEDS, digest, save_trace, check_sources
from verify_functional_controls import compare

ROOT=Path('audits/functional_interface_v1')
VISUAL_SEEDS=(1301,1311,1321)
ORDERS=tuple(itertools.permutations(range(3)))
ROWS=tuple(range(6))
DIAGNOSTIC_ROWS=(17,54,96,156,183,232,310,379)
SOURCES=('scripts/integrate_functional_controls.py','src/attcon/functional_interface.py',
    'src/attcon/functional_model.py','src/attcon/functional_controls.py',
    'src/attcon/predictive_attention.py','src/attcon/bound_content.py',
    'scripts/run_functional_controls.py','scripts/verify_functional_controls.py',
    'docs/FUNCTIONAL_INTERFACE_PROTOCOL.md')


def read_trace(path):
    with gzip.open(path,'rb') as stream:return torch.load(io.BytesIO(stream.read()),weights_only=True)


def visual_checkpoint(seed):
    return Path(f'audits/bound_content/confirmation_v2/seed{seed}.pt')


def prepare():
    if (ROOT/'manifest.json').exists():return json.loads((ROOT/'manifest.json').read_text())
    parent_manifest=json.loads((PARENT/'manifest.json').read_text());check_sources(parent_manifest)
    dependencies=[PARENT/'manifest.json']
    for seed, visual_seed in zip(SEEDS,VISUAL_SEEDS):
        dependencies.extend((PARENT/f'seed{seed}.json',PARENT/f'seed{seed}.pt',
                             PARENT/f'seed{seed}_trace.pt.gz',visual_checkpoint(visual_seed)))
    sources={p:Path(p).read_text() for p in SOURCES}
    manifest={'kind':'engineering_development_interface_export',
        'attention_seeds':SEEDS,'visual_seeds':VISUAL_SEEDS,'count':512,
        'scene_seed_rule':'750000000 + visual_seed','sample_rows':ROWS,
        'diagnostic_rows_seed2021_route1_to_none':DIAGNOSTIC_ROWS,
        'diagnostic_selection':'post-hoc retained failures from parent study',
        'static_variants':['neutral','remapped','conflict','missing_relation','model_swap','restored'],
        'revision_window':16,'revision_routes':['no_feedback','feedback'],
        'expected_records':664,'new_language_reports':0,'no_retraining':True,
        'source_sha256':{p:digest(p) for p in SOURCES},
        'dependency_sha256':{str(p):digest(p) for p in dependencies},
        'torch_version':torch.__version__}
    ROOT.mkdir(parents=True)
    (ROOT/'source_code.json').write_text(json.dumps(sources,indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def check_manifest(manifest):
    for p,h in {**manifest['source_sha256'],**manifest['dependency_sha256']}.items():
        assert digest(p)==h,p


def pack(current, hidden, predicted, observed, anticipated=None, step=12):
    return {'allocation':current.allocation.clone(),'access':current.access.clone(),
        'effects':current.effects.clone(),'hidden':hidden.clone(),'prospective_recovery':predicted.clone(),
        'observed':observed.clone(),
        'anticipated':torch.empty(len(observed),0,20) if anticipated is None else anticipated.clone(),
        'modeled_step':step}


def render(visual,state,row,order):
    current=Forecast(state['allocation'],state['access'],state['effects'])
    return record(visual,current,state['prospective_recovery'],state['observed'],row,order,
                  anticipated=state['anticipated'],modeled_step=state['modeled_step'])


@torch.no_grad()
def build(seed,visual_seed):
    parent=read_trace(PARENT/f'seed{seed}_trace.pt.gz')
    model=PredictiveAttention();encoder=VisualEncoder()
    model.load_state_dict(torch.load(PARENT/f'seed{seed}.pt',weights_only=True)['state_dict']);model.eval()
    encoder.load_state_dict(torch.load(visual_checkpoint(visual_seed),weights_only=True)['state_dict']);encoder.eval()
    patches,colors,shapes=scenes(750000000+visual_seed,512)
    patches=patches[:,:1].expand(-1,2,-1,-1,-1,-1).clone()
    visual=encoder(patches)
    joint=((visual[..., :4].argmax(-1)==colors[:,:1]) &
           (visual[...,4:].argmax(-1)==shapes[:,:1])).double().mean().item()
    states={'visual':visual.clone(),'colors':colors[:,0].clone(),'shapes':shapes[:,0].clone(),
            'static':{},'revision':{}}
    samples=[];checks={'visual_joint_accuracy':joint,'visual_minimum_pass':joint>=.99,
        'restoration_exact':True,'unchanged_current_under_model_swap':True,
        'parent_forecasts_exact':True,'no_feedback_new_world_invariance':True,
        'all_six_presentation_orders':True,'readout_matches_buffer0':True,
        'new_language_reports':0}
    for condition in CONDITIONS:
        source=parent['static'][condition]
        seq,hidden=model(source['observations']);current=last_forecast(seq)
        for key,value in (('current_allocation',current.allocation),('current_access',current.access),
                          ('current_effects',current.effects),('hidden',hidden)):
            assert torch.equal(value,source[key]),(seed,condition,key)
        baseline=counterfactual_recovery(model,current,hidden)
        altered=current.intervene('effects',current.effects.flip(-2))
        changed=counterfactual_recovery(model,altered,hidden)
        restored=counterfactual_recovery(model,altered.intervene('effects',current.effects),hidden)
        for value,key in ((baseline,'predicted_recovery'),(changed,'changed_recovery'),(restored,'restored_recovery')):
            assert torch.equal(value,source[key]),(seed,condition,key)
        assert torch.equal(restored,baseline)
        base_state=pack(current,hidden,baseline,source['observations'])
        changed_state=pack(altered,hidden,changed,source['observations'])
        states['static'][condition]={'neutral':base_state,'model_swap':changed_state}
        for row,order in zip(ROWS,ORDERS):
            base=render(visual,base_state,row,order)
            swap=render(visual,changed_state,row,order)
            assert base['predicted_current']==swap['predicted_current']
            conflict=copy.deepcopy(base)
            conflict['informal_operator_note']='Every command selects p0 at every buffer, and commands never change output contents.'
            variants={'neutral':base,'remapped':remap(base),'conflict':conflict,
                'missing_relation':without_relation(base),'model_swap':swap,
                'restored':record(visual,current,restored,source['observations'],row,order,
                    anticipated=torch.empty(512,0,20),modeled_step=12)}
            assert variants['restored']==base
            for variant,payload in variants.items():
                samples.append({'id':f'{seed}_c{condition}_r{row}_{variant}',
                    'kind':'static_development','attention_seed':seed,'visual_seed':visual_seed,
                    'condition':condition,'row':row,'order':order,'variant':variant,'payload':payload})
    for old in CONDITIONS:
        no_feedback_reference=None
        for new in CONDITIONS:
            key=f'{old}->{new}';source=parent['revision'][key]
            history=parent['static'][old]['observations']
            sequence,hidden=model(history);initial=last_forecast(sequence)
            for field in ('allocation','access','effects'):
                assert torch.equal(getattr(initial,field),source['initial_'+field])
            assert torch.equal(hidden,source['initial_hidden'])
            baseline,feedback=initial.clone(),initial.clone()
            bh,fh=hidden.clone(),hidden.clone()
            anticipated=[]
            for t in range(1,17):
                baseline,bh,x=advance_model(model,baseline,bh,source['commands'][:,t-1])
                anticipated.append(x)
                sequence,fh=model(source['observed'][:,t-1:t],fh)
                feedback=last_forecast(sequence)
                if t in WINDOWS:
                    target=source['windows'][t]
                    for context,current,h in (('no_feedback',baseline,bh),('feedback',feedback,fh)):
                        assert torch.equal(current.effects,target[context+'_effects'])
                        assert torch.equal(counterfactual_recovery(model,current,h),target[context+'_recovery'])
                    assert torch.equal(feedback.access[:,0],target['feedback_current_recovery'])
            anticipated=torch.stack(anticipated,1)
            assert torch.equal(anticipated,source['anticipated'])
            baseline_state=pack(baseline,bh,source['windows'][16]['no_feedback_recovery'],history,anticipated,28)
            feedback_state=pack(feedback,fh,source['windows'][16]['feedback_recovery'],
                                torch.cat((history,source['observed']),1),step=28)
            if no_feedback_reference is None:no_feedback_reference=baseline_state
            else:compare(no_feedback_reference,baseline_state,'unobserved_new_world_invariance')
            states['revision'][key]={'no_feedback':baseline_state,'feedback':feedback_state}
            rows=list(ROWS)
            if seed==2021 and old==1 and new==-1:rows.extend(DIAGNOSTIC_ROWS)
            for row in rows:
                order=ORDERS[row%6]
                for context,state in states['revision'][key].items():
                    samples.append({'id':f'{seed}_{key}_r{row}_{context}',
                        'kind':'targeted_parent_failure_diagnostic' if row in DIAGNOSTIC_ROWS and row not in ROWS else 'revision_development',
                        'attention_seed':seed,'visual_seed':visual_seed,'old_condition':old,
                        'new_condition':new,'row':row,'order':order,'context':context,
                        'payload':render(visual,state,row,order)})
    for sample in samples:
        p=sample['payload']
        forbidden={'condition','owner','physical_owner','old_condition','new_condition','attention_seed','visual_seed','variant','gates'}
        def inspect(value):
            if isinstance(value,dict):
                assert not forbidden.intersection(value)
                for v in value.values():inspect(v)
            elif isinstance(value,list):
                for v in value:inspect(v)
        inspect(p)
        if sample.get('variant')=='remapped':
            order=sample['order'];buf0={'n0':'q7','n1':'q2','n2':'q9'}[f'n{order.index(0)}']
        else:buf0=f'n{sample['order'].index(0)}'
        nodes={n['node']:n for n in p['predicted_current']}
        readout=nodes[p['output_node']]
        assert 'selection_distribution' not in readout
        for a,b in zip(readout['objects'],nodes[buf0]['objects']):
            for k in ('position','color_distribution','shape_distribution','identified_color','identified_shape'):assert a[k]==b[k]
    checks['records']=len(samples)
    return states,samples,checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','export','verify'),required=True)
    args=parser.parse_args();torch.set_num_threads(2);manifest=prepare();check_manifest(manifest)
    if args.stage=='prepare':print('Prepared three-way visual/interface export, no language requests');return
    if args.stage=='export':
        if (ROOT/'attempt.json').exists():raise SystemExit('refusing to overwrite export attempt')
        (ROOT/'attempt.json').write_text(json.dumps({'status':'reserved','new_language_reports':0})+'\n')
    outcomes=[];all_samples=[];hashes={}
    for seed,visual_seed in zip(SEEDS,VISUAL_SEEDS):
        state,samples,checks=build(seed,visual_seed)
        path=ROOT/f'seed{seed}_states.pt.gz'
        if args.stage=='export':save_trace(state,path)
        else:compare(state,read_trace(path),f'integration_seed{seed}')
        hashes[str(path)]=digest(path);outcomes.append({'attention_seed':seed,'visual_seed':visual_seed,**checks})
        all_samples.extend(samples)
        print(seed,checks,flush=True)
    assert len(all_samples)==manifest['expected_records']
    summary={'kind':'engineering_development_interface_export','results':outcomes,
        'records':len(all_samples),'new_language_reports':0,'parent_engineering_verdict':'fail_retained',
        'archive_sha256':hashes,'source_sha256':manifest['source_sha256']}
    if args.stage=='export':
        (ROOT/'records.json').write_text(json.dumps(all_samples,separators=(',',':'))+'\n')
        summary['records_sha256']=digest(ROOT/'records.json')
        (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (ROOT/'attempt.json').write_text(json.dumps({'status':'completed','new_language_reports':0})+'\n')
    else:
        assert json.loads((ROOT/'records.json').read_text())==json.loads(json.dumps(all_samples))
        summary['records_sha256']=digest(ROOT/'records.json')
        assert summary==json.loads((ROOT/'summary.json').read_text())
        assert json.loads((ROOT/'attempt.json').read_text())['status']=='completed'
        print('All 664 neutral records and visual/model/revision tensors exactly replayed')


if __name__=='__main__':main()
