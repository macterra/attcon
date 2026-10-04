"""Export unchanged post-hoc probe outcomes and descriptive figures."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from run_control_probes import ROOT,SEEDS,check,read_trace
from run_functional_controls import digest


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    check(json.loads((ROOT/'manifest.json').read_text()))
    results={};records=[];cases=[];dependencies={}
    for seed in SEEDS:
        path=ROOT/f'seed{seed}.json';trace_path=ROOT/f'seed{seed}_trace.pt.gz'
        record=json.loads(path.read_text());assert record['status']=='completed'
        assert record['trace_sha256']==digest(trace_path)
        results[seed]=record['result'];trace=read_trace(trace_path)
        dependencies[str(path)]=digest(path);dependencies[str(trace_path)]=digest(trace_path)
        for route,r in record['result']['results'].items():
            for w in r['windows']:
                records.append({'seed':seed,'prior_controlled_channel':r['prior_controlled_channel'],
                    'policy':r['policy'],'continued_controlled_channel':r['continued_controlled_channel'],**w})
                snapshot=trace[route]['windows'][w['additional_observations']]
                assert w['control_accuracy']==float(snapshot['control_correct'].double().mean())
        # Per-episode outcomes restricted to the world that remains disconnected.
        for old,diagnostic in record['result']['initial_diagnostics'].items():
            original=read_trace(Path(f'audits/functional_controls_v1/seed{seed}_trace.pt.gz'))['revision'][f'{old}->-1']
            matches=(original['observed'][:,:,int(old)*4:(int(old)+1)*4].argmax(-1)==original['commands']).sum(1)
            for episode in diagnostic['incorrect_episode_ids']:
                for policy in ('random','agree','contradict'):
                    route=f'{old}_{policy}_-1'
                    for t in (1,2,4,8):
                        cases.append({'seed':seed,'prior_controlled_channel':int(old),'episode':episode,
                            'policy':policy,'additional_observations':t,
                            'original_command_allocation_matches':int(matches[episode]),
                            'original_command_allocation_disagreements':16-int(matches[episode]),
                            'correct_while_still_disconnected':bool(trace[route]['windows'][t]['control_correct'][episode])})
    write_csv(ROOT/'all_window_metrics.csv',records)
    write_csv(ROOT/'original_error_episode_outcomes.csv',cases)
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    styles={'random':('#666666','o'),'agree':('#D55E00','s'),'contradict':('#009E73','^')}
    times=(1,2,4,8)
    for old in (0,1):
        for column,new in enumerate((-1,old)):
            ax=axes[old,column]
            for policy,(color,marker) in styles.items():
                values=torch.tensor([[w['control_accuracy'] for w in results[seed]['results'][f'{old}_{policy}_{new}']['windows']] for seed in SEEDS])
                ax.plot(times,values.mean(0),marker=marker,color=color,label=policy)
                ax.fill_between(times,values.min(0).values,values.max(0).values,color=color,alpha=.15)
            ax.set_title(f'Prior buffer {old}; '+('still disconnected' if new==-1 else 'control restored'))
            ax.set_xscale('log',base=2);ax.set_xticks(times,labels=times)
            ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
    for ax in axes[:,0]:ax.set_ylabel('Correct control mask (episode fraction)')
    for ax in axes[-1]:ax.set_xlabel('Additional actual observations after original endpoint')
    axes[0,0].legend()
    fig.suptitle('Fixed-model command probes: mean of 3 models, shaded model range\nPost-hoc diagnostic; no new success gate or confidence interval')
    fig.tight_layout(rect=(0,0,1,.94))
    assets=Path('docs/assets');assets.mkdir(exist_ok=True)
    fig.savefig(assets/'control_disconnection_probes.svg');fig.savefig(ROOT/'control_probes.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for policy,(color,marker) in styles.items():
        r=results[2021]['results'][f'1_{policy}_-1']
        ax.plot(times,[w['original_error_subset_correct_in_current_world'] for w in r['windows']],
            marker=marker,color=color,label=policy)
    ax.set_xscale('log',base=2);ax.set_xticks(times,labels=times);ax.set_yticks(range(9));ax.set_ylim(-.3,8.3)
    ax.set_xlabel('Additional actual observations');ax.set_ylabel('Originally incorrect episodes now correct')
    ax.set_title('Eight retained seed-2021 errors; world remains disconnected')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(assets/'control_disconnection_original_errors.svg');plt.close(fig)
    finals=[r for r in records if r['additional_observations']==8]
    summary={'kind':'post_hoc_engineering_diagnostic_summary','contexts':36,'window_records':144,
        'parent_overall_verdict':'fail_unchanged','new_language_reports':0,'new_success_gate':None,
        'agreement_observational_equivalence_exact':all(r['agreement_observational_equivalence_exact'] for r in results.values()),
        'original_failed_route_seed2021_still_disconnected':{
            policy:{k:results[2021]['results'][f'1_{policy}_-1']['windows'][-1][k]
                for k in ('control_accuracy','incorrect_control_episodes','original_error_subset_correct_in_current_world')}
            for policy in styles},
        'eight_step_policy_ranges':{
            policy:{'control_accuracy_min':min(r['control_accuracy'] for r in finals if r['policy']==policy),
                    'control_accuracy_max':max(r['control_accuracy'] for r in finals if r['policy']==policy),
                    'effect_accuracy_min':min(r['effect_accuracy'] for r in finals if r['policy']==policy),
                    'effect_accuracy_max':max(r['effect_accuracy'] for r in finals if r['policy']==policy)} for policy in styles},
        'dependency_sha256':dependencies,'export_source_sha256':digest(Path(__file__))}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='dependency_sha256'},indent=2))


if __name__=='__main__':main()
