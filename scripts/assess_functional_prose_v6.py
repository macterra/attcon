"""Compare source-blind assertions with the supplied named internal state."""
import argparse
import collections
import json
from pathlib import Path
from extract_functional_prose_v1 import sentences


from assess_functional_prose import assess as assess_attributes
from extract_functional_process_v2 import valid_claim


def process_nodes(source, scope, command):
    if scope=='current' and command is None:
        return source['predicted_current']
    if scope=='command' and source['predicted_by_command'] is not None:
        trial=next((t for t in source['predicted_by_command'] if t['command']==command),None)
        return trial['nodes'] if trial else []
    return []


def expected_process(source):
    expected=set()
    slices=[('current',None,source['predicted_current'])]
    slices += [('command',t['command'],t['nodes']) for t in source['predicted_by_command'] or []]
    for scope,command,nodes in slices:
        for node in nodes:
            if 'selected_position' not in node: continue
            expected.add((scope,command,node['node'],None,'selection',node['selected_position']))
            if scope=='current':
                for obj in node['objects']:
                    expected.add((scope,command,node['node'],obj['position'],'recovery_trend',obj['unattended_trend']))
    return expected


def forecast_presence(source,scope,command,node):
    address=source['output_node'] if node=='output' else node
    current=next((n for n in source['predicted_current'] if n['node']==address),None)
    if current is None:return None
    if scope=='current':
        return 'selected_position' in current if command is None else None
    if scope!='command':return None
    if command is not None and command not in ('k0','k1','k2','k3'):return None
    trials=source['predicted_by_command']
    if trials is None:return False
    if command is not None:
        trial=next((t for t in trials if t['command']==command),None)
        if trial is None:return None
        found=next((n for n in trial['nodes'] if n['node']==address),None)
        return 'selected_position' in found if found else None
    values=[]
    for trial in trials:
        found=next((n for n in trial['nodes'] if n['node']==address),None)
        if found is None:return None
        values.append('selected_position' in found)
    return values[0] if values and all(v==values[0] for v in values) else None


def assess(source,report,extracted):
    result=assess_attributes(source,report,extracted)
    expected=expected_process(source);covered=set();seen=set();count=len(sentences(report))
    for c in extracted['process_claims']:
        node=source['output_node'] if c['node']=='output' else c['node']
        key=(c['scope'],c['command'],node,c['position'],c['dimension'],c['value'])
        if key in seen:continue
        seen.add(key)
        schema_valid=valid_claim(c)
        evidence=bool(c['evidence_ids']) and all(0<=i<count for i in c['evidence_ids'])
        resolved=False;correct=False
        if c['dimension']=='selection_forecast_presence':
            presence=forecast_presence(source,c['scope'],c['command'],c['node'])
            resolved=presence is not None
            correct=resolved and c['value']==('present' if presence else 'absent')
        else:
            nodes=process_nodes(source,c['scope'],c['command'])
            found=next((n for n in nodes if n['node']==node and 'selected_position' in n),None)
            resolved=found is not None
            if c['dimension']=='recovery_trend':
                resolved=resolved and any(o['position']==c['position'] for o in found['objects'])
            correct=resolved and key in expected
        resolved=resolved and schema_valid
        correct=bool(correct and schema_valid and evidence)
        if correct and key in expected:covered.add(key)
        result['facts'].append({'claim':c,'correct':correct,'primary':True,
            'resolved':bool(resolved),'valid_evidence':evidence,'schema_valid':schema_valid})
    result['process_coverage_correct']=len(covered)
    result['process_coverage_total']=len(expected)
    if source['predicted_by_command'] is None:
        # Describing missing selection-forecast information is not predicting a command effect.
        result['missing_relation_command_claims']+=sum(c['scope']=='command' and
            not (c['dimension']=='selection_forecast_presence' and c['value']=='absent')
            for c in extracted['process_claims'])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default='neutral_functional_pilot_v6');args=p.parse_args()
    root=Path('audits')/args.study;requests=json.loads((root/'requests.json').read_text())
    cases=[];groups=collections.defaultdict(list)
    for i,req in enumerate(requests):
        response=json.loads((root/(req['id']+'.json')).read_text())
        audit=json.loads((root/'extraction_v1'/f'r{i:03d}.json').read_text())
        if audit.get('response',{}).get('status')!='completed' or 'parsed' not in audit:
            raise SystemExit('incomplete audit; cannot score full run')
        process=json.loads((root/'process_extraction_v2'/f'r{i:03d}.json').read_text())
        if process.get('response',{}).get('status')!='completed' or 'parsed' not in process:
            raise SystemExit('incomplete process audit; cannot score full run')
        extracted={**audit['parsed'],**process['parsed']}
        result=assess(req['source'],response['report'],extracted)
        cases.append({'id':req['id'],'variant':req['variant'],**result});groups[req['variant']].append(result)
    metrics={}
    for variant,rows in groups.items():
        facts=[f for r in rows for f in r['facts']];primary=[f for f in facts if f['primary']]
        valid=[f for f in primary if f['resolved'] and f['valid_evidence']]
        checked=sum(f['correct'] for f in valid)/len(valid) if valid else 0
        precision=sum(f['correct'] for f in primary)/len(primary) if primary else 0
        cov=sum(r['coverage_correct'] for r in rows);total=sum(r['coverage_total'] for r in rows)
        coverage=cov/total if total else 1
        proc=sum(r['process_coverage_correct'] for r in rows); proc_total=sum(r['process_coverage_total'] for r in rows)
        proc_coverage=proc/proc_total if proc_total else 1
        metrics[variant]={'process_coverage_correct':proc,'process_coverage_total':proc_total,'process_coverage':proc_coverage,'claims':len(primary),'checked_accuracy':checked,'conservative_precision':precision,
            'coverage_correct':cov,'coverage_total':total,'coverage':coverage,
            'unresolved_or_invalid':sum(not f['resolved'] or not f['valid_evidence'] for f in primary),
            'incorrect_examples':[f for f in primary if not f['correct']],
            'features':{k:sum(bool(r['features'][k]) for r in rows) for k in rows[0]['features']},
            'characters':dict(collections.Counter(r['character'] for r in rows)),
            'factual_minima_pass':checked>=.98 and precision>=.95 and coverage>=.90 and proc_coverage>=.90}
    missing=groups['missing_relation']
    control=all(not r['missing_relation_command_claims'] and not r['missing_relation_positive_relation'] for r in missing)
    primary_groups=('neutral','remapped','conflicting_description','restored','model_swap')
    verdict=all(metrics[v]['factual_minima_pass'] for v in primary_groups) and control
    result={'study':args.study,'kind':'development_only','all_report_and_audit_attempts_complete':True,
            'primary_factual_minima_pass':verdict,'missing_relation_gate_pass':control,'metrics':metrics,'cases':cases,
            'limitations':'Selected extracted attribute, selection and two-step recovery-trend fields; negative exclusions and unextracted implications not checked. Auditor is source-blind but not independent human review.'}
    (root/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'primary_factual_minima_pass':verdict,'missing_relation_gate_pass':control,
        'metrics':{v:{k:x[k] for k in ('checked_accuracy','conservative_precision','coverage','process_coverage','factual_minima_pass')} for v,x in metrics.items()}},indent=2))


if __name__=='__main__':main()
