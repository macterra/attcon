"""Audit what the existing uncertainty bridge changes; no new task or reporter."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import torch
from attcon.functional_model import buffer_contents
from run_functional_controls import SEEDS,digest,save_trace
from verify_functional_controls import compare

ROOT=Path('audits/functional_readout_audit_v1')
INTERFACE=Path('audits/functional_interface_v1')
CONTROL=Path('audits/functional_controls_v1')
SOURCES=('scripts/audit_functional_readout.py','src/attcon/functional_model.py',
    'src/attcon/functional_interface.py','docs/FUNCTIONAL_READOUT_AUDIT_PROTOCOL.md')


def read_trace(path):
    with gzip.open(path,'rb') as stream:return torch.load(io.BytesIO(stream.read()),weights_only=True)


def prepare():
    if (ROOT/'manifest.json').exists():return json.loads((ROOT/'manifest.json').read_text())
    paths=[INTERFACE/'summary.json']
    for seed in SEEDS:
        paths.extend((INTERFACE/f'seed{seed}_states.pt.gz',CONTROL/f'seed{seed}_trace.pt.gz'))
    manifest={'kind':'post_hoc_readout_dependency_audit','seeds':SEEDS,'conditions':[0,1,-1],
        'variants':['neutral','model_swap'],'rows_per_context':512,'output_source_buffer':0,
        'identification_threshold':.6,'rounding_decimal_places':5,'new_success_gate':None,
        'new_language_reports':0,'new_task_trials':0,'no_retraining':True,
        'source_sha256':{p:digest(p) for p in SOURCES},
        'dependency_sha256':{str(p):digest(p) for p in paths},'torch_version':torch.__version__}
    ROOT.mkdir(parents=True)
    (ROOT/'source_code.json').write_text(json.dumps({p:Path(p).read_text() for p in SOURCES},indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def rounded(tensor):
    # Same Python rounding as the archived reporter payloads, retained as float64.
    return torch.tensor([round(v,5) for v in tensor.flatten().tolist()],dtype=torch.float64).reshape(tensor.shape)


def identities(values,threshold=None):
    parts=[]
    for probs in (values[...,:4],values[...,4:]):
        maximum,index=probs.max(-1)
        if threshold is not None:index=torch.where(maximum>=threshold,index,torch.full_like(index,-1))
        parts.append(index)
    return torch.stack(parts,-1)


def varying(values):
    # [episode,command,position,color+shape label]
    return (values!=values[:,:1]).any(-1).any(1)


@torch.no_grad()
def assess():
    metrics=[];traces={}
    for seed in SEEDS:
        state=read_trace(INTERFACE/f'seed{seed}_states.pt.gz')
        world=read_trace(CONTROL/f'seed{seed}_trace.pt.gz')
        visual=state['visual'][:,0];base_id=identities(visual)
        truth=torch.stack((state['colors'],state['shapes']),-1)
        traces[seed]={'visual':visual.clone(),'truth':truth.clone(),'contexts':{}}
        for condition in (0,1,-1):
            physical=world['static'][condition]
            physical_current_q=physical['access'][:,-1,0,0]
            physical_future_q=physical['physical_recovery'][:,:,0]
            physical_future=buffer_contents(visual[:,None],physical_future_q)
            for variant in ('neutral','model_swap'):
                model=state['static'][condition][variant]
                current_q=model['access'][:,0,0];future_q=model['prospective_recovery'][:,:,0]
                current=buffer_contents(visual,current_q)
                future=buffer_contents(visual[:,None],future_q)
                rq_current,rq_future=rounded(current),rounded(future)
                raw_current=identities(current);raw_future=identities(future)
                round_current=identities(rq_current);round_future=identities(rq_future)
                identified_current=identities(rq_current,.6);identified_future=identities(rq_future,.6)
                threshold_vary=varying(identified_future);raw_vary=varying(raw_future)
                round_vary=varying(round_future)
                physical_identified=identities(rounded(physical_future),.6)
                peak=torch.stack((future[...,:4].max(-1).values,future[...,4:].max(-1).values),-1)
                m={'seed':seed,'condition':condition,'variant':variant,
                    'current_recovery_min':float(current_q.min()),'future_recovery_min':float(future_q.min()),
                    'raw_current_identity_matches_visual':float((raw_current==base_id).all(-1).double().mean()),
                    'raw_future_identity_matches_visual':float((raw_future==base_id[:,None]).all(-1).double().mean()),
                    'rounded_current_identity_matches_visual':float((round_current==base_id).all(-1).double().mean()),
                    'rounded_future_identity_matches_visual':float((round_future==base_id[:,None]).all(-1).double().mean()),
                    'raw_current_joint_identity_accuracy':float((raw_current==truth).all(-1).double().mean()),
                    'raw_command_varying_position_episodes':int(raw_vary.sum()),
                    'rounded_command_varying_position_episodes':int(round_vary.sum()),
                    'threshold_command_varying_position_episodes':int(threshold_vary.sum()),
                    'physical_threshold_command_varying_position_episodes':int(varying(physical_identified).sum()),
                    'position_episodes_total':int(threshold_vary.numel()),
                    'current_both_attributes_identified_fraction':float((identified_current>=0).all(-1).double().mean()),
                    'future_both_attributes_identified_fraction':float((identified_future>=0).all(-1).double().mean()),
                    'mean_command_peak_probability_range':float((peak.max(1).values-peak.min(1).values).double().mean()),
                    'physical_current_recovery_zero_positions':int((physical_current_q==0).sum())}
                metrics.append(m)
                traces[seed]['contexts'][f'{condition}_{variant}']={
                    'current_recovery':current_q.clone(),'future_recovery':future_q.clone(),
                    'physical_current_recovery':physical_current_q.clone(),
                    'physical_future_recovery':physical_future_q.clone(),
                    'current':current.clone(),'future':future.clone(),
                    'rounded_current':rq_current,'rounded_future':rq_future,
                    'raw_current_identities':raw_current,'raw_future_identities':raw_future,
                    'identified_current':identified_current,'identified_future':identified_future,
                    'physical_future_identified':physical_identified}
    return metrics,traces


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','run','verify'),required=True)
    args=p.parse_args();torch.set_num_threads(2);manifest=prepare()
    for path,h in {**manifest['source_sha256'],**manifest['dependency_sha256']}.items():assert digest(path)==h,path
    if args.stage=='prepare':print('Prepared post-hoc raw/thresholded readout audit');return
    summary_path=ROOT/'summary.json';trace_path=ROOT/'states.pt.gz'
    if args.stage=='run':
        if summary_path.exists() or trace_path.exists():raise SystemExit('refusing to overwrite readout audit')
        summary_path.write_text(json.dumps({'status':'reserved'})+'\n')
    metrics,traces=assess()
    result={'status':'completed','kind':'post_hoc_readout_dependency_audit','contexts':len(metrics),
        'metrics':metrics,'new_language_reports':0,'new_task_trials':0,
        'raw_command_varying_position_episodes':sum(m['raw_command_varying_position_episodes'] for m in metrics),
        'rounded_command_varying_position_episodes':sum(m['rounded_command_varying_position_episodes'] for m in metrics),
        'all_positive_modeled_recovery':all(m['current_recovery_min']>0 and m['future_recovery_min']>0 for m in metrics),
        'raw_identity_preservation_all_contexts':all(m['raw_current_identity_matches_visual']==1 and m['raw_future_identity_matches_visual']==1 for m in metrics),
        'native_category_decision_pipeline_established':False,'new_success_gate':None}
    if args.stage=='run':
        save_trace(traces,trace_path);result['trace_sha256']=digest(trace_path)
        summary_path.write_text(json.dumps(result,indent=2)+'\n')
        with (ROOT/'metrics.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    else:
        result['trace_sha256']=digest(trace_path)
        assert result==json.loads(summary_path.read_text());compare(traces,read_trace(trace_path))
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))


if __name__=='__main__':main()
