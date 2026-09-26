"""Reserved-context delay and unannounced sensor-misspecification stress."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from attcon.prospective import make_splits,root_events
from attcon.prospective_learning import load_models,evaluate
from attcon.regulation import weight_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('artifact',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);run=json.loads(a.artifact.read_text());saved,models=load_models(ROOT/run['checkpoint'])
    before={name:weight_fingerprint(model) for name,model in models.items()}
    if any(before[name]!=run['training'][name]['selected_sha256'] for name in models):raise ValueError('checkpoint mismatch')
    splits=make_splits(run['seed'],run['task']);baseline=splits['stress'];groups=set(baseline.group.tolist())
    if baseline.fingerprint()!=run['dataset_sha256']['stress'] or any(groups & set(data.group.tolist()) for name,data in splits.items() if name!='stress'):raise ValueError('invalid reserved contexts')
    shifted=make_splits(run['seed'],run['task'],degradation=.15)['stress']
    if not torch.equal(root_events(baseline),root_events(shifted)):raise ValueError('misspecification changed initial observations')
    conditions={}
    for name,data,delay in (('baseline',baseline,1),('long_delay',baseline,5),('misspecified',shifted,1)):
        conditions[name]={'delay':delay,'dataset_sha256':data.fingerprint(),'evaluation':evaluate(models,data,run['seed'],delay),
            'actual_sensor_a_accuracy':(data.sample_a==data.value).float().mean().item(),'actual_sensor_b_accuracy':(data.sample_b==data.value).float().mean().item()}
    if before!={name:weight_fingerprint(model) for name,model in models.items()}:raise ValueError('stress mutated models')
    sources=('scripts/audit_prospective_stress.py','src/attcon/prospective.py','src/attcon/prospective_learning.py','docs/COMPLETION_PROTOCOL.md')
    result={'audit':'prospective_stress','task':run['task'],'architecture':run['architecture'],'seed':run['seed'],
        'source':str(a.artifact),'source_artifact_sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'disjoint_reserved_contexts':True,'reserved_context_count':len(groups),'conditions':conditions,
        'boundary':'No fitting on reserved contexts. Misspecified sensors are .15 less reliable than the displayed cue, which remains unchanged. Analytic comparator knows actual reliability and therefore has an information advantage under misspecification. Exact serial verification remains exact.'}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(str(a.out),flush=True)
if __name__=='__main__':main()
