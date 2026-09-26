"""Correct the report comparator's omitted answer identity, retaining old results."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from attcon.prospective import make_splits,root_events
from attcon.prospective_learning import load_models
from attcon.prospective_reporting import labels_for,fit_report,scores
from attcon.history_reporting import pad_features
from attcon.regulation import weight_fingerprint


def identity_cue_features(agent,state,data):
    with torch.no_grad():return pad_features(torch.cat((agent.answer(state),data.quality[:,None]),1),128)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report_artifact',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);report=json.loads(a.report_artifact.read_text());source=ROOT/report['source'];run=json.loads(source.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=report['source_artifact_sha256']:raise ValueError('original source changed')
    _,models=load_models(ROOT/run['checkpoint']);agent=models['state'];before=weight_fingerprint(agent)
    if before!=report['agent_sha256']:raise ValueError('checkpoint mismatch')
    splits=make_splits(run['seed'],run['task']);features={};labels={}
    for name in ('report_fit','validation','test'):
        if splits[name].fingerprint()!=run['dataset_sha256'][name]:raise ValueError('data mismatch')
        with torch.no_grad():state=agent.advance(root_events(splits[name]))
        features[name]=identity_cue_features(agent,state,splits[name]);labels[name]=labels_for(splits[name])
    model,selection=fit_report(features['report_fit'],labels['report_fit'],features['validation'],labels['validation'],run['seed']+15000)
    with torch.no_grad():metrics=scores(model(features['test']).argmax(-1),labels['test'])
    gain=report['reports']['state']['mean_quality_source_accuracy']-metrics['mean_quality_source_accuracy']
    if before!=weight_fingerprint(agent):raise ValueError('controller mutated')
    checkpoint=ROOT/'outputs/prospective'/(a.out.stem+'.pt')
    torch.save({'model':model.state_dict(),'agent_sha256':before,'seed':run['seed'],'task':run['task'],'architecture':run['architecture']},checkpoint)
    sources=('scripts/audit_identity_cue_reporting.py','src/attcon/prospective_reporting.py','docs/COMPLETION_CORRECTIONS.md')
    result={'audit':'identity_cue_report_correction','task':run['task'],'architecture':run['architecture'],'seed':run['seed'],
        'original_report':str(a.report_artifact),'original_report_sha256':hashlib.sha256(a.report_artifact.read_bytes()).hexdigest(),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'checkpoint':str(checkpoint.relative_to(ROOT)),'agent_sha256':before,'parameters':sum(p.numel() for p in model.parameters()),
        'selection':selection,'identity_cue_report':metrics,'state_minus_identity_cue_gain':gain,'descriptive_margin_pass':gain>=.02,
        'boundary':'Post-registration correction: original confidence+cue reporter omitted answer identity. Full answer logits plus the same cue repair that input asymmetry. Reused tests make this a diagnostic, not independent confirmation. Original report gates remain archived and cannot alone establish fair advantage.'}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(f'{a.out}: state gain={gain:.4f}',flush=True)
if __name__=='__main__':main()
