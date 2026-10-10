"""Replay retained failed v1 verdict with canonical JSON tuple/list handling."""
import hashlib,json
from pathlib import Path
from assess_integrated_report_qualification_v1 import ROOT,assess

def main():
    saved=json.loads((ROOT/'qualification_assessment.json').read_text())
    for p,h in saved['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    code=Path('scripts/assess_integrated_report_qualification_v1.py')
    assert hashlib.sha256(code.read_bytes()).hexdigest()==saved['code_sha256']
    cases=json.loads((ROOT/'qualification_cases.json').read_text());gold=json.loads((ROOT/'qualification_answers.json').read_text())
    for label in ('A','B'):
        actual=assess(json.loads((ROOT/f'auditor_{label}/qualification.json').read_text()),cases,gold)
        assert json.loads(json.dumps(actual))==saved['auditors'][label]
    reference=json.loads((ROOT/'reference_check/review.json').read_text())['assertions']
    refpass=len(reference)==48 and {(r['case_id'],r['sentence_id']) for r in reference}=={(r['case_id'],r['sentence_id']) for r in gold} and all(r['agreement'] is True and not r.get('ambiguous',False) for r in reference)
    assert refpass==saved['reference_key_review_passed']
    assert saved['passed']==(refpass and all(r['passed'] for r in saved['auditors'].values()))
    print('Verified original v1 failure and all source/code pins; canonical JSON comparison only')
if __name__=='__main__':main()
