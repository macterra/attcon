"""Assess the frozen, bounded semantic challenge; no open-prose certification."""
import argparse,hashlib,json
from pathlib import Path
from build_integrated_report_pilot_v1 import ROOT,pointer,encoded

def assess(rows,cases,answers):
    by_case={x['id']:x for x in cases};expected={(x['case_id'],x['sentence_id']):x for x in answers}
    seen=set();errors=[];correct=0
    for r in rows:
        key=(r.get('case_id'),r.get('sentence_id'))
        if key not in expected or key in seen:
            errors.append({'key':key,'error':'unknown or repeated assertion'});continue
        seen.add(key);gold=expected[key];issues=[]
        if r.get('quote')!=gold['quote']:issues.append('quote mismatch')
        if r.get('outcome')!=gold['outcome']:issues.append('outcome mismatch')
        paths=r.get('source_paths',[])
        if not isinstance(paths,list) or not all(isinstance(x,str) for x in paths):issues.append('malformed source paths')
        else:
            for path in paths:
                try:pointer(by_case[key[0]]['payload'],path)
                except (KeyError,ValueError,IndexError,TypeError):issues.append('invalid source path '+path)
            if gold['outcome'] in ('entailed','contradicted') and not paths:issues.append('missing direct evidence')
        if not r.get('rationale','').strip():issues.append('missing rationale')
        if issues:errors.append({'key':key,'error':issues,'expected_outcome':gold['outcome'],'actual_outcome':r.get('outcome')})
        else:correct+=1
    missing=sorted(set(expected)-seen)
    return {'assertions':len(expected),'correct':correct,'missing':missing,'errors':errors,'passed':correct==len(expected) and not errors and not missing,'limit':'Exact bounded challenge only; quoted evidence and semantic gold agreement do not establish open-prose inventory completeness.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    cases=json.loads((ROOT/'qualification_cases.json').read_text());answers=json.loads((ROOT/'qualification_answers.json').read_text())
    reference=json.loads((ROOT/'reference_check/review.json').read_text())
    reviewed=reference['assertions'];refkeys={(r['case_id'],r['sentence_id']) for r in reviewed}
    required={(r['case_id'],r['sentence_id']) for r in answers}
    refpass=refkeys==required and len(reviewed)==48 and all(r['agreement'] is True and not r.get('ambiguous',False) for r in reviewed)
    results={}
    sources=['qualification_cases.json','qualification_answers.json','reference_check/review.json','qualification_source_first_receipts.json']
    receipt=json.loads((ROOT/'qualification_source_first_receipts.json').read_text())
    for label in ('A','B'):
        rel=f'auditor_{label}/qualification.json';rows=json.loads((ROOT/rel).read_text())
        results[label]=assess(rows,cases,answers);sources+=[rel,f'auditor_{label}/source_checklist.json',f'auditor_{label}/metadata.json']
        raw=(ROOT/f'auditor_{label}/source_checklist.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==receipt[label]['source_checklist_sha256']
    value={'kind':'bounded_synthetic_task_auditor_qualification','reference_key_review_passed':refpass,'auditors':results,'passed':refpass and all(r['passed'] for r in results.values()),'new_reporter_calls':0,'confirmation':False,'human_required':False,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target=ROOT/'qualification_assessment.json'
    if a.verify:assert json.loads(target.read_text())==value
    else:
        if target.exists():raise SystemExit('refusing overwrite')
        target.write_bytes(encoded(value))
    print(json.dumps({'passed':value['passed'],'reference_key_review_passed':refpass,'auditors':{k:{'correct':v['correct'],'assertions':v['assertions']} for k,v in results.items()}}))
if __name__=='__main__':main()
