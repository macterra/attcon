"""One entry point for the final study: smoke, archived replay, or full retraining."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2503,2521,2539)
TASKS = ('serial','routing')
ARCHITECTURES = ('gru','rnn')


def command_plan():
    stages = [('environment', [['scripts/audit_prospective_environment.py']])]
    fitted, exploration, reporting, identity, stress, summaries = [], [], [], [], [], []
    for task in TASKS:
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                base = f'{task}_{arch}_{seed}'
                artifact = f'audits/prospective_{base}.json'
                report = f'audits/prospective_reporting_{base}.json'
                fitted.append(['scripts/train_prospective.py','--task',task,'--architecture',arch,'--seed',str(seed),'--out',artifact])
                reporting.append(['scripts/audit_prospective_reporting.py',artifact,'--out',report])
                identity.append(['scripts/audit_identity_cue_reporting.py',report,'--out',f'audits/identity_reporting_{base}.json'])
                stress.append(['scripts/audit_prospective_stress.py',artifact,'--out',f'audits/prospective_stress_{base}.json'])
            for prefix in ('prospective','prospective_reporting'):
                summaries.append(['scripts/summarize_prospective.py',*[f'audits/{prefix}_{task}_{arch}_{seed}.json' for seed in SEEDS],'--out',f'audits/{prefix}_{task}_{arch}_summary.json'])
        for seed in SEEDS:
            exploration.append(['scripts/train_prospective_exploration.py','--task',task,'--seed',str(seed),'--out',f'audits/exploration_{task}_{seed}.json'])
        summaries.append(['scripts/summarize_prospective.py',*[f'audits/exploration_{task}_{seed}.json' for seed in SEEDS],'--out',f'audits/exploration_{task}_summary.json'])
    stages += [('fitted',fitted),('exploration',exploration),('reporting',reporting),('identity_correction',identity),('stress',stress),('summaries',summaries)]
    stages += [('final_audit',[['scripts/final_project_audit.py','--package','--replay']])]
    return stages


def smoke():
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from attcon.prospective import make_splits,root_events
    from attcon.prospective_learning import train_fitted,evaluate
    from attcon.prospective_exploration import train_exploration
    from attcon.prospective_reporting import report_features,labels_for,fit_report,scores,causal_audit
    torch.set_num_threads(1)
    results={}
    for task in TASKS:
        splits={name:batch.subset(torch.arange(108)) for name,batch in make_splits(2717,task).items()}
        for arch in ARCHITECTURES:
            models={}
            for family in ('state','blind','cue'):
                models[family],_=train_fitted(splits['train'],splits['validation'],2717,arch,family,epochs=2)
            measured=evaluate(models,splits['test'],2717)
            results[task+'_'+arch]={'costs':list(measured['costs']),'scientific_evidence':False}
            if arch=='gru':
                agent=models['state'];states={};features={};labels={}
                for name in ('report_fit','validation','test'):
                    with torch.no_grad():states[name]=agent.advance(root_events(splits[name]))
                    features[name]=report_features(agent,states[name],splits[name]);labels[name]=labels_for(splits[name])
                reporters={}
                for family in ('state','action','cue'):
                    reporters[family],_=fit_report(features['report_fit'][family],labels['report_fit'],features['validation'][family],labels['validation'],2717,steps=2)
                result=causal_audit(agent,reporters,states['report_fit'],splits['report_fit'],states['test'],splits['test'],2717)
                results[task+'_causal_smoke']={'valid':result['valid'],'scientific_evidence':False}
        native={}
        for family in ('state','cue'):
            native[family],_=train_exploration(splits['train'],splits['validation'],2717,family,updates=3)
        results[task+'_exploration']={'costs':list(evaluate(native,splits['test'],2717,native=True)['costs']),'scientific_evidence':False}
    path=ROOT/'outputs/final_smoke.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'audit':'smoke_only','results':results},indent=2,allow_nan=False)+'\n')
    print('Smoke pipeline completed; no scientific artifacts overwritten.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true');mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--verify',action='store_true',help='Restore archived checkpoints and replay all final metrics.')
    mode.add_argument('--run',action='store_true',help='Retrain the full fixed matrix and regenerate audits/archive.')
    parser.add_argument('--jobs',type=int,default=3,help='Concurrent independent runs;1–3, default3.')
    args=parser.parse_args()
    if not 1<=args.jobs<=3:parser.error('--jobs must be1–3')
    if args.plan:
        print(json.dumps(command_plan(),indent=2));return
    if args.smoke:
        smoke();return
    if args.verify:
        subprocess.run([sys.executable,'scripts/final_project_audit.py','--restore','--verify','--replay'],cwd=ROOT,check=True);return
    logs=ROOT/'outputs/reproduction_logs';logs.mkdir(parents=True,exist_ok=True)
    for stage,jobs in command_plan():
        def run(item):
            index,arguments=item;path=logs/f'{stage}_{index}.log'
            with path.open('w') as log:
                subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f'completed {stage} {index+1}/{len(jobs)}',flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(run,enumerate(jobs)))
    print('Full registered study, corrected comparator, archive, and replay audit completed.')
if __name__=='__main__':main()
