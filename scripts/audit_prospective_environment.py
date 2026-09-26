"""Pretraining prospective environment audit; primary outcomes remain reserved."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from attcon.prospective import make_splits,analytic_policy,COSTS


def main():
    torch.set_num_threads(1)
    runs={}
    for task in ('serial','routing'):
        for seed in (2503,2521,2539):
            splits=make_splits(seed,task)
            groups={name:set(data.group.tolist()) for name,data in splits.items()}
            disjoint=all(not a & b for i,a in enumerate(groups.values()) for b in list(groups.values())[i+1:])
            if not disjoint: raise ValueError('overlapping contexts')
            data=splits['validation']; oracle=analytic_policy(data)
            runs[f'{task}_{seed}']={'disjoint':disjoint,'dataset_sha256':{name:batch.fingerprint() for name,batch in splits.items()},
                'contexts':{name:len(group) for name,group in groups.items()},
                'validation_oracle':{str(cost):{'realized_return':oracle['return'][data.cost==cost].mean().item(),'expected_return':oracle['expected_return'][data.cost==cost].mean().item()} for cost in COSTS}}
    sources=('src/attcon/prospective.py','scripts/audit_prospective_environment.py','docs/COMPLETION_PROTOCOL.md')
    result={'audit':'prospective_environment','runs':runs,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    (ROOT/'audits/prospective_environment.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Six environment partitions validated.')
if __name__=='__main__': main()
