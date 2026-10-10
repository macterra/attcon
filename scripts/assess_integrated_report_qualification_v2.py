"""New supportedness gate; retain detailed label disagreement and v1 failure."""
import argparse,copy,hashlib,json
from pathlib import Path
from assess_integrated_report_qualification_v1 import assess
from build_integrated_report_pilot_v1 import encoded
ROOT=Path('audits/integrated_report_pilot_v2')
def evaluate(rows,cases,answers):
    fine=assess(rows,cases,answers)
    mapped=copy.deepcopy(rows);gold=copy.deepcopy(answers)
    for row in mapped:
        if row.get('outcome') in ('contradicted','unsupported','ambiguous'):row['outcome']='unsupported'
    for row in gold:row['outcome']='entailed' if row['supported'] else 'unsupported'
    result=assess(mapped,cases,gold)
    original={(r['case_id'],r['sentence_id']):r for r in answers}
    for row in rows:
        key=(row.get('case_id'),row.get('sentence_id'))
        if key in original and original[key]['outcome'] in ('entailed','contradicted') and not row.get('source_paths'):
            result['errors'].append({'key':key,'error':'direct source evidence required'})
    result['passed']=result['passed'] and not result['errors']
    result['fine_label_assessment']=fine
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    cases=json.loads((ROOT/'qualification_cases.json').read_text());answers=json.loads((ROOT/'qualification_answers_corrected.json').read_text())
    review=json.loads((ROOT/'reference_check/corrected_review.json').read_text());req={(r['case_id'],r['sentence_id']) for r in answers};items=review['assertions']
    refpass=len(items)==48 and {(r['case_id'],r['sentence_id']) for r in items}==req and all(r['agreement'] is True and not r['ambiguous'] for r in items)
    receipt=json.loads((ROOT/'qualification_source_first_receipts.json').read_text());results={};paths=['qualification_cases.json','qualification_answers.json','qualification_answers_corrected.json','qualification_answer_path_erratum.json','reference_check/corrected_review.json','qualification_source_first_receipts.json']
    for auditor in ('A2','B2'):
        folder=f'auditor_{auditor}';rows=json.loads((ROOT/folder/'qualification.json').read_text());results[auditor]=evaluate(rows,cases,answers)
        assert hashlib.sha256((ROOT/folder/'source_checklist.json').read_bytes()).hexdigest()==receipt[auditor]['source_checklist_sha256']
        paths+=[f'{folder}/qualification.json',f'{folder}/source_checklist.json',f'{folder}/metadata.json']
    data={'kind':'bounded_supportedness_qualification','reference_key_review_passed':refpass,'auditors':results,'passed':refpass and all(r['passed'] for r in results.values()),'version_1_failed_gate_unchanged':True,'fine_label_semantics_qualified':False,'open_prose_completeness_qualified':False,'new_reports':0,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target=ROOT/'qualification_assessment.json'
    if a.verify:assert json.loads(target.read_text())==data
    else:
        if target.exists():raise SystemExit('refusing overwrite')
        target.write_bytes(encoded(data))
    print(json.dumps({'passed':data['passed'],'reference_checked':refpass,'auditors':{k:{'support_correct':v['correct'],'fine_correct':v['fine_label_assessment']['correct']} for k,v in results.items()}}))
if __name__=='__main__':main()
