"""Fixed eight-observation diagnostic of saved disconnection states; no retraining."""
import argparse
import gzip
import io
import json
from pathlib import Path
import torch
from attcon.predictive_attention import PredictiveAttention,Forecast
from attcon.functional_controls import control_mask,true_control_mask
from attcon.control_probes import POLICIES,WINDOWS,probe_commands,probe
from run_functional_controls import ROOT as PARENT,SEEDS,digest,save_trace
from verify_functional_controls import compare

ROOT=Path('audits/control_disconnection_diagnostic_v1')
INTERFACE=Path('audits/functional_interface_v1')
SOURCES=('src/attcon/control_probes.py','src/attcon/functional_controls.py',
    'src/attcon/predictive_attention.py','scripts/run_control_probes.py',
    'scripts/run_functional_controls.py','scripts/verify_functional_controls.py',
    'docs/CONTROL_DISCONNECTION_DIAGNOSTIC_PROTOCOL.md')


def read_trace(path):
    with gzip.open(path,'rb') as stream:return torch.load(io.BytesIO(stream.read()),weights_only=True)


def prepare():
    if (ROOT/'manifest.json').exists():return json.loads((ROOT/'manifest.json').read_text())
    dependencies=[]
    for seed in SEEDS:
        dependencies.extend((PARENT/f'seed{seed}.json',PARENT/f'seed{seed}.pt',
            PARENT/f'seed{seed}_trace.pt.gz',INTERFACE/f'seed{seed}_states.pt.gz'))
    sources={p:Path(p).read_text() for p in SOURCES}
    manifest={'kind':'post_hoc_engineering_diagnostic','seeds':SEEDS,'episodes_per_context':512,
        'prior_controlled_channels':[0,1],'starting_point':'parent 16-observation feedback state after disconnection',
        'continued_worlds':['none','restored_prior_control'],'policies':POLICIES,
        'additional_observation_windows':WINDOWS,'quality':.8,
        'random_command_seed_rule':'770000000 + model_seed',
        'nonzero_offset_seed_rule':'780000000 + model_seed',
        'contradictory_command_rule':'automatic destination + uniform offset 1..3 (mod 4)',
        'examiner_policy_uses_physical_phase':True,'native_controller_benchmark':False,
        'new_language_reports':0,'no_retraining':True,'parent_verdict_unchanged':True,
        'new_success_gate':None,'source_sha256':{p:digest(p) for p in SOURCES},
        'dependency_sha256':{str(p):digest(p) for p in dependencies},'torch_version':torch.__version__}
    ROOT.mkdir(parents=True);(ROOT/'source_code.json').write_text(json.dumps(sources,indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def check(manifest):
    for p,h in {**manifest['source_sha256'],**manifest['dependency_sha256']}.items():assert digest(p)==h,p


@torch.no_grad()
def assess(seed):
    parent=read_trace(PARENT/f'seed{seed}_trace.pt.gz')
    integration=read_trace(INTERFACE/f'seed{seed}_states.pt.gz')
    model=PredictiveAttention();model.load_state_dict(torch.load(PARENT/f'seed{seed}.pt',weights_only=True)['state_dict']);model.eval()
    random_commands=torch.randint(4,(512,8),generator=torch.Generator().manual_seed(770000000+seed))
    offsets=torch.randint(1,4,(512,8),generator=torch.Generator().manual_seed(780000000+seed))
    results={};traces={};initial_diagnostics={}
    for old in (0,1):
        route=f'{old}->-1';initial=integration['revision'][route]['feedback'];source=parent['revision'][route]
        current=Forecast(initial['allocation'],initial['access'],initial['effects']);hidden=initial['hidden']
        for field in ('effects',):assert torch.equal(getattr(current,field),source['windows'][16]['feedback_'+field])
        replay=source['windows'][16]['physical_replay'];q=source['windows'][16]['physical_current_recovery']
        direction=parent['static'][old]['direction']
        correct=(control_mask(current)==true_control_mask(torch.full((512,),-1))).all(-1)
        matches=(source['observed'][:,:,old*4:(old+1)*4].argmax(-1)==source['commands']).sum(1)
        initial_diagnostics[old]={'incorrect_episode_ids':(~correct).nonzero().flatten().tolist(),
            'initial_control_accuracy':float(correct.double().mean()),
            'mean_command_allocation_matches_all_episodes':float(matches.double().mean()),
            'mean_command_allocation_matches_incorrect_episodes':float(matches[~correct].double().mean()) if bool((~correct).any()) else None}
        for policy in POLICIES:
            commands=probe_commands(replay,direction,old,random_commands,offsets,policy)
            for new in (-1,old):
                key=f'{old}_{policy}_{new}'
                records,trace=probe(model,current,hidden,replay,direction,q,commands,new)
                for record in records:
                    window=trace['windows'][record['additional_observations']]
                    before_wrong=~correct
                    record['original_disconnection_errors']=int(before_wrong.sum())
                    record['original_error_subset_correct_in_current_world']=int(window['control_correct'][before_wrong].sum())
                results[key]={'prior_controlled_channel':old,'policy':policy,
                    'continued_controlled_channel':new,'windows':records}
                traces[key]=trace
            a,b=traces[f'{old}_{policy}_-1'],traces[f'{old}_{policy}_{old}']
            initial_keys=('initial_allocation','initial_access','initial_effects','initial_hidden',
                          'initial_physical_recovery','initial_replay','direction','commands')
            assert all(torch.equal(a[k],b[k]) for k in initial_keys)
            if policy=='agree':
                assert torch.equal(a['observations'],b['observations'])
                for t in WINDOWS:
                    for key in ('modeled_allocation','modeled_access','modeled_effects','hidden','predicted_recovery',
                                'physical_current_recovery','physical_replay'):
                        assert torch.equal(a['windows'][t][key],b['windows'][t][key])
                    assert not torch.equal(a['windows'][t]['target_effects'],b['windows'][t]['target_effects'])
    return {'seed':seed,'kind':'post_hoc_engineering_diagnostic','initial_diagnostics':initial_diagnostics,
        'results':results,'agreement_observational_equivalence_exact':True,
        'unobserved_switch_initial_preservation_exact':True,'parent_verdict_unchanged':True,
        'no_new_language_reports':True},traces


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','run','verify'),required=True)
    args=p.parse_args();torch.set_num_threads(2);manifest=prepare();check(manifest)
    if args.stage=='prepare':print('Prepared fixed eight-observation post-hoc disconnection probes');return
    for seed in SEEDS:
        record_path=ROOT/f'seed{seed}.json';trace_path=ROOT/f'seed{seed}_trace.pt.gz'
        if args.stage=='run':
            if record_path.exists() or trace_path.exists():raise SystemExit('refusing to overwrite diagnostic attempt')
            record_path.write_text(json.dumps({'seed':seed,'status':'reserved'})+'\n')
        result,trace=assess(seed)
        if args.stage=='run':
            save_trace(trace,trace_path)
            record={'status':'completed','result':result,'trace_sha256':digest(trace_path)}
            record_path.write_text(json.dumps(record,indent=2)+'\n')
        else:
            record=json.loads(record_path.read_text());assert record['status']=='completed'
            assert record['trace_sha256']==digest(trace_path)
            assert record['result']==json.loads(json.dumps(result));compare(trace,read_trace(trace_path))
        print(seed,'exactly replayed' if args.stage=='verify' else 'completed',flush=True)


if __name__=='__main__':main()
