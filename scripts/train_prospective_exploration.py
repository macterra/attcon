"""Train registered native answer/decline and acquisition policies from rewards."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from attcon.prospective import make_splits
from attcon.prospective_exploration import train_exploration
from attcon.prospective_learning import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('serial','routing'),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    splits=make_splits(a.seed,a.task);models={};training={}
    for family in ('state','cue'):
        models[family],training[family]=train_exploration(splits['train'],splits['validation'],a.seed,family)
    for field in ('initial_sha256','parameters','updates','episodes_sampled'):
        if len({v[field] for v in training.values()})!=1:raise ValueError('unmatched '+field)
    test=evaluate(models,splits['test'],a.seed,native=True)
    checkpoint=ROOT/'outputs/prospective'/(a.out.stem+'.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True)
    fingerprints={name:batch.fingerprint() for name,batch in splits.items()}
    torch.save({'seed':a.seed,'task':a.task,'architecture':'gru','models':{name:m.state_dict() for name,m in models.items()},'training':training,'dataset_sha256':fingerprints},checkpoint)
    sources=('src/attcon/prospective_exploration.py','scripts/train_prospective_exploration.py','src/attcon/prospective.py','src/attcon/prospective_learning.py','docs/COMPLETION_PROTOCOL.md')
    result={'audit':'prospective_exploration_v1','task':a.task,'architecture':'gru','seed':a.seed,'training':training,'test':test,
        'checkpoint':str(checkpoint.relative_to(ROOT)),'dataset_sha256':fingerprints,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'boundary':'From-scratch epsilon-greedy learning uses experienced chosen-action rewards only. Decline earns0.30 and is a native communicative action, not a supervised access report. Finite simulator; no general autonomous-learning or conscious-access claim. Different objective/budget from fitted-control runs.'}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(f'{a.out}: supported={test["all_gates_pass"]}',flush=True)
if __name__=='__main__':main()
