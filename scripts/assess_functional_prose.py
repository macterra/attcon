"""Compare source-blind assertions with the supplied named internal state."""
import argparse
import collections
import json
from pathlib import Path
from extract_functional_prose_v1 import sentences


def node_state(source, scope, command, node, position):
    if scope == 'current':
        if command is not None: return None
        nodes = source['predicted_current']
    elif scope == 'command' and source['predicted_by_command'] is not None:
        trial = next((t for t in source['predicted_by_command'] if t['command']==command),None)
        if trial is None: return None
        nodes = trial['nodes']
    else: return None
    address = source['output_node'] if node=='output' else node
    found=next((n for n in nodes if n['node']==address),None)
    return next((o for o in found['objects'] if o['position']==position),None) if found else None


def expected_coverage(source):
    expected=set()
    slices=[('current',None,source['predicted_current'])]
    slices += [('command',t['command'],t['nodes']) for t in source['predicted_by_command'] or []]
    for scope, command, nodes in slices:
        output=next(n for n in nodes if n['node']==source['output_node'])
        for o in output['objects']:
            for dimension in ('color','shape'):
                if o['identified_'+dimension] is not None:
                    expected.add((scope,command,o['position'],dimension,o['identified_'+dimension]))
    return expected


def assess(source, report, extracted):
    expected=expected_coverage(source);covered=set();facts=[];seen=set()
    count=len(sentences(report))
    for c in extracted['claims']:
        key=tuple(c.get(k) for k in ('node','position','scope','command','dimension','status','value'))
        if key in seen: continue
        seen.add(key)
        evidence=bool(c['evidence_ids']) and all(0<=i<count for i in c['evidence_ids'])
        obj=node_state(source,c['scope'],c['command'],c['node'],c['position'])
        correct=False;eligible=c['status']!='possible'
        if obj and evidence:
            dominant=obj['identified_'+c['dimension']]
            if c['status']=='identified': correct=c['value']==dominant and dominant is not None
            elif c['status']=='unidentified': correct=dominant is None and c['value'] is None
            else: correct=c['value'] in obj[c['dimension']+'_distribution'] and obj[c['dimension']+'_distribution'][c['value']]>0
            output=c['node']=='output' or c['node']==source['output_node']
            coverage_key=(c['scope'],c['command'],c['position'],c['dimension'],c['value'])
            if correct and output and c['status']=='identified' and coverage_key in expected:
                covered.add(coverage_key)
        facts.append({'claim':c,'correct':correct,'primary':eligible,'resolved':obj is not None,'valid_evidence':evidence})
    return {'facts':facts,'coverage_correct':len(covered),'coverage_total':len(expected),
            'missing_relation_command_claims':sum(c['scope']=='command' for c in extracted['claims']) if source['predicted_by_command'] is None else 0,
            'missing_relation_positive_relation':bool(extracted['features']['command_access_relation']) if source['predicted_by_command'] is None else False,
            'features':extracted['features'],'character':extracted['character']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default='neutral_functional_pilot_v2');args=p.parse_args()
    root=Path('audits')/args.study;requests=json.loads((root/'requests.json').read_text())
    cases=[];groups=collections.defaultdict(list)
    for i,req in enumerate(requests):
        response=json.loads((root/(req['id']+'.json')).read_text())
        audit=json.loads((root/'extraction_v1'/f'r{i:03d}.json').read_text())
        if audit.get('response',{}).get('status')!='completed' or 'parsed' not in audit:
            raise SystemExit('incomplete audit; cannot score full run')
        result=assess(req['source'],response['report'],audit['parsed'])
        cases.append({'id':req['id'],'variant':req['variant'],**result});groups[req['variant']].append(result)
    metrics={}
    for variant,rows in groups.items():
        facts=[f for r in rows for f in r['facts']];primary=[f for f in facts if f['primary']]
        valid=[f for f in primary if f['resolved'] and f['valid_evidence']]
        checked=sum(f['correct'] for f in valid)/len(valid) if valid else 0
        precision=sum(f['correct'] for f in primary)/len(primary) if primary else 0
        cov=sum(r['coverage_correct'] for r in rows);total=sum(r['coverage_total'] for r in rows)
        coverage=cov/total if total else 1
        metrics[variant]={'claims':len(primary),'checked_accuracy':checked,'conservative_precision':precision,
            'coverage_correct':cov,'coverage_total':total,'coverage':coverage,
            'unresolved_or_invalid':sum(not f['resolved'] or not f['valid_evidence'] for f in primary),
            'incorrect_examples':[f for f in primary if not f['correct']],
            'features':{k:sum(bool(r['features'][k]) for r in rows) for k in rows[0]['features']},
            'characters':dict(collections.Counter(r['character'] for r in rows)),
            'factual_minima_pass':checked>=.98 and precision>=.95 and coverage>=.90}
    missing=groups['missing_relation']
    control=all(not r['missing_relation_command_claims'] and not r['missing_relation_positive_relation'] for r in missing)
    primary_groups=('neutral','remapped','conflicting_description','restored')
    verdict=all(metrics[v]['factual_minima_pass'] for v in primary_groups) and control
    result={'study':args.study,'kind':'development_only','all_report_and_audit_attempts_complete':True,
            'primary_factual_minima_pass':verdict,'missing_relation_gate_pass':control,'metrics':metrics,'cases':cases,
            'limitations':'Selected extracted attribute fields; negative exclusions and unextracted implications not checked. Auditor is source-blind but not independent human review.'}
    (root/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'primary_factual_minima_pass':verdict,'missing_relation_gate_pass':control,
        'metrics':{v:{k:x[k] for k in ('checked_accuracy','conservative_precision','coverage','factual_minima_pass')} for v,x in metrics.items()}},indent=2))


if __name__=='__main__':main()
