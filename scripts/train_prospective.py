"""Run one registered prospective task/architecture/seed cell."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from attcon.prospective import make_splits
from attcon.prospective_learning import train_fitted,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('serial','routing'),required=True)
    p.add_argument('--architecture',choices=('gru','rnn'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    splits=make_splits(a.seed,a.task);models={};training={}
    for family in ('state','blind','cue'):
        models[family],training[family]=train_fitted(splits['train'],splits['validation'],a.seed,a.architecture,family)
    for field in ('initial_sha256','parameters','updates'):
        if len({v[field] for v in training.values()})!=1:raise ValueError('unmatched '+field)
    result=evaluate(models,splits['test'],a.seed)
    checkpoint=ROOT/'outputs/prospective'/(a.out.stem+'.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True)
    fingerprints={name:data.fingerprint() for name,data in splits.items()}
    torch.save({'seed':a.seed,'task':a.task,'architecture':a.architecture,'models':{name:model.state_dict() for name,model in models.items()},'training':training,'dataset_sha256':fingerprints},checkpoint)
    sources=('src/attcon/prospective.py','src/attcon/prospective_learning.py','scripts/train_prospective.py','docs/COMPLETION_PROTOCOL.md')
    artifact={'audit':'prospective_fitted_v1','seed':a.seed,'task':a.task,'architecture':a.architecture,
        'training':training,'test':result,'dataset_sha256':fingerprints,'checkpoint':str(checkpoint.relative_to(ROOT)),
        'source_sha256':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources},
        'boundary':'Fitted full-information environmental control; no reporting labels train agents. Primary comparator receives the same observed quality information. No introspective or Stage 8 conclusion from task success alone.'}
    a.out.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n');print(f'{a.out}: supported={result["all_gates_pass"]}',flush=True)
if __name__=='__main__':main()
