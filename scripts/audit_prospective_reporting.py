"""Fit independent reports and evaluate causal report/control coupling."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from attcon.prospective import make_splits,root_events
from attcon.prospective_learning import load_models
from attcon.prospective_reporting import labels_for,report_features,fit_report,scores,causal_audit
from attcon.regulation import weight_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('artifact',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    run=json.loads(a.artifact.read_text());saved,controllers=load_models(ROOT/run['checkpoint'])
    seed=run['seed'];agent=controllers['state'];before=weight_fingerprint(agent)
    if before!=run['training']['state']['selected_sha256']:raise ValueError('checkpoint mismatch')
    splits=make_splits(seed,run['task']);states={};features={};labels={}
    for name in ('report_fit','validation','test'):
        data=splits[name]
        if data.fingerprint()!=run['dataset_sha256'][name]:raise ValueError('data mismatch')
        with torch.no_grad():states[name]=agent.advance(root_events(data))
        features[name]=report_features(agent,states[name],data);labels[name]=labels_for(data)
    models={};selections={};reports={}
    for name in ('state','action','cue'):
        models[name],selections[name]=fit_report(features['report_fit'][name],labels['report_fit'],features['validation'][name],labels['validation'],seed+15000)
        with torch.no_grad():reports[name]=scores(models[name](features['test'][name]).argmax(-1),labels['test'])
    nulls=[];null_selections=[]
    for index in range(5):
        null_seed=seed+16000+index;order=torch.randperm(len(labels['report_fit']),generator=torch.Generator().manual_seed(null_seed))
        model,selection=fit_report(features['report_fit']['state'],labels['report_fit'][order],features['validation']['state'],labels['validation'],null_seed)
        with torch.no_grad():nulls.append(scores(model(features['test']['state']).argmax(-1),labels['test']))
        null_selections.append(selection)
        print(f'{a.out.stem}: null {index+1}/5',flush=True)
    primary=reports['state'];gates={'quality_accuracy':primary['quality_accuracy']>=.90,
        'verified_accuracy':primary['verified_value_accuracy']>=.90,'unverified_accuracy':primary['unverified_accuracy']>=.90,
        'fair_readout_advantage':primary['mean_quality_source_accuracy']-reports['cue']['mean_quality_source_accuracy']>=.02}
    causal=causal_audit(agent,models,states['report_fit'],splits['report_fit'],states['test'],splits['test'],seed+17000)
    if before!=weight_fingerprint(agent):raise ValueError('reporting changed controller')
    checkpoint=ROOT/'outputs/prospective'/(a.out.stem+'.pt')
    torch.save({'models':{name:m.state_dict() for name,m in models.items()},'agent_sha256':before,'seed':seed,'task':run['task'],'architecture':run['architecture']},checkpoint)
    sources=('src/attcon/prospective_reporting.py','scripts/audit_prospective_reporting.py','docs/COMPLETION_PROTOCOL.md')
    result={'audit':'prospective_reporting_coupling','task':run['task'],'architecture':run['architecture'],'seed':seed,
        'source':str(a.artifact),'source_artifact_sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'checkpoint':str(checkpoint.relative_to(ROOT)),'agent_sha256':before,'reports':reports,'selections':selections,
        'parameters':{name:sum(p.numel() for p in m.parameters()) for name,m in models.items()},
        'null_reports':nulls,'null_selections':null_selections,
        'null_mean_accuracy_p95':torch.tensor([v['mean_quality_source_accuracy'] for v in nulls]).quantile(.95).item(),
        'gates':gates,'all_gates_pass':all(gates.values()),'causal':causal,
        'boundary':'Post-training supervised measurements of environmental quality and source verification, not native introspection. Fair comparator has the observed cue. Five-fit null p95 is descriptive only. Earlier reporting and Stage 8 criteria remain unchanged.'}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(f'{a.out}: gates={gates}, causal_valid={causal["valid"]}',flush=True)
if __name__=='__main__':main()
