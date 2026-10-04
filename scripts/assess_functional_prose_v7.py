"""Compare source-blind assertions with the supplied named internal state."""
import argparse
import collections
import json
from pathlib import Path
from extract_functional_prose_v1 import sentences


from assess_functional_prose import assess as assess_attributes
from extract_functional_process_v2 import valid_claim


from assess_functional_prose_v6 import process_nodes, expected_process, forecast_presence, assess


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default='neutral_functional_pilot_v7');args=p.parse_args()
    root=Path('audits')/args.study;requests=json.loads((root/'requests.json').read_text())
    cases=[];groups=collections.defaultdict(list)
    for i,req in enumerate(requests):
        response=json.loads((root/(req['id']+'.json')).read_text())
        audit=json.loads((root/'extraction_v1'/f'r{i:03d}.json').read_text())
        if audit.get('response',{}).get('status')!='completed' or 'parsed' not in audit:
            raise SystemExit('incomplete audit; cannot score full run')
        process=json.loads((root/'process_extraction_v3'/f'r{i:03d}.json').read_text())
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
