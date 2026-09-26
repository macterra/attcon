"""Validate and aggregate every registered cost and seed without gate relaxation."""
import argparse,hashlib,json
from pathlib import Path
SEEDS={2503,2521,2539}
COSTS={'0.1','0.25','0.4'}


def validate_control(run):
    expected_families={'state','blind','cue'} if run['audit']=='prospective_fitted_v1' else {'state','cue'}
    if set(run['training'])!=expected_families:raise ValueError('missing controller family')
    for field in ('initial_sha256','parameters','updates'):
        if len({v[field] for v in run['training'].values()})!=1:raise ValueError('unmatched '+field)
    if set(run['test']['costs'])!=COSTS:raise ValueError('missing registered cost')
    for cost,value in run['test']['costs'].items():
        metrics=value['policies'];intervals=value['state_minus_comparator_intervals']
        for name,interval in intervals.items():
            if abs(interval['mean']-(metrics['state']['return']-metrics[name]['return']))>1e-5:
                raise ValueError('inconsistent return difference')
        gates={'fresh_accuracy':metrics['state']['conditions']['fresh']['accuracy']>=.85,
            'forced_final_accuracy':value['forced_final_accuracy']>=.85,
            'gain_over_fair_cue':intervals['cue']['mean']>=.02,'positive_fair_bound':intervals['cue']['low']>0}
        gates.update({'gain_over_'+name:intervals[name]['mean']>=.02 for name in ('never','first','second')})
        if gates!=value['gates'] or all(gates.values())!=value['all_gates_pass']:raise ValueError('inconsistent gate')
    if run['test']['all_gates_pass']!=all(v['all_gates_pass'] for v in run['test']['costs'].values()):raise ValueError('inconsistent overall gate')


def validate_report(run):
    if set(run['reports'])!={'state','action','cue'} or len(set(run['parameters'].values()))!=1:raise ValueError('unmatched reporters')
    primary=run['reports']['state']
    gates={'quality_accuracy':primary['quality_accuracy']>=.90,'verified_accuracy':primary['verified_value_accuracy']>=.90,
        'unverified_accuracy':primary['unverified_accuracy']>=.90,
        'fair_readout_advantage':primary['mean_quality_source_accuracy']-run['reports']['cue']['mean_quality_source_accuracy']>=.02}
    if gates!=run['gates'] or all(gates.values())!=run['all_gates_pass']:raise ValueError('inconsistent report gate')
    if len(run['null_reports'])!=5:raise ValueError('wrong null count')
    if 'transplants' in run['causal']:
        for value in run['causal']['transplants'].values():
            expected=value['max_logit_residual']<=1e-5 and value['restoration_valid'] and all(value['comparator_report_invariance'].values())
            if expected!=value['valid']:raise ValueError('inconsistent causal validity')
        if all(v['valid'] for v in run['causal']['transplants'].values())!=run['causal']['valid']:raise ValueError('inconsistent causal summary')


def summarize(paths):
    runs=[json.loads(p.read_text()) for p in paths]
    if not runs or len({r['audit'] for r in runs})!=1:raise ValueError('mixed or missing audit types')
    reference=runs[0];groups={}
    for run in runs:
        if run['source_sha256']!=reference['source_sha256']:raise ValueError('incomparable sources')
        key=run['task']+'_'+run['architecture'];groups.setdefault(key,[]).append(run)
        (validate_report if run['audit']=='prospective_reporting_coupling' else validate_control)(run)
    def bounds(values):return {'min':min(values),'max':max(values)}
    summaries={}
    for key,group in groups.items():
        if len(group)!=3 or {r['seed'] for r in group}!=SEEDS:raise ValueError('missing/duplicate registered seeds')
        if reference['audit']=='prospective_reporting_coupling':
            summaries[key]={'all_seeds_supported':all(r['all_gates_pass'] for r in group),
                'gate_pass_counts':{gate:sum(r['gates'][gate] for r in group) for gate in group[0]['gates']},
                'report_ranges':{family:{metric:bounds([r['reports'][family][metric] for r in group]) for metric in group[0]['reports'][family]} for family in ('state','action','cue')},
                'all_causal_contrasts_valid':all(r['causal']['valid'] for r in group),
                'causal_ranges':{kind:{metric:bounds([r['causal']['transplants'][kind][metric] for r in group if 'transplants' in r['causal']]) for metric in ('quality_report_changed_rate','initial_policy_switch_rate','inspection_trajectory_switch_rate','joint_report_trajectory_change_rate','return_change')} for kind in ('quality','random')} if all('transplants' in r['causal'] for r in group) else None}
        else:
            summaries[key]={'all_seeds_supported':all(r['test']['all_gates_pass'] for r in group),
                'costs':{cost:{'return_ranges':{name:bounds([r['test']['costs'][cost]['policies'][name]['return'] for r in group]) for name in group[0]['test']['costs'][cost]['policies']},
                    'fair_gain_range':bounds([r['test']['costs'][cost]['state_minus_comparator_intervals']['cue']['mean'] for r in group]),
                    'state_answer_coverage_range':bounds([r['test']['costs'][cost]['policies']['state']['answer_coverage'] for r in group]),
                    'state_selective_accuracy_range':bounds([r['test']['costs'][cost]['policies']['state']['selective_accuracy'] for r in group if r['test']['costs'][cost]['policies']['state']['selective_accuracy'] is not None]) if any(r['test']['costs'][cost]['policies']['state']['selective_accuracy'] is not None for r in group) else None,
                    'gate_pass_counts':{gate:sum(r['test']['costs'][cost]['gates'][gate] for r in group) for gate in group[0]['test']['costs'][cost]['gates']}}
                    for cost in sorted(COSTS)}}
    return {'audit':reference['audit']+'_summary','sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'groups':summaries,'all_groups_supported':all(v['all_seeds_supported'] for v in summaries.values()),
        'boundary':'Registered seeds, costs, and fair comparators retained. Task or decoding success does not establish conscious access. Source-quality labels are environmental measurements. Stage 8 remains subject to its original criteria.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('artifacts',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=summarize(a.artifacts);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'groups':list(result['groups']),'all_groups_supported':result['all_groups_supported']}))
if __name__=='__main__':main()
