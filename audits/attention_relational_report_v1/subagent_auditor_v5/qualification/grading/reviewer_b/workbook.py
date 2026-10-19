import json,hashlib,sys
from pathlib import Path
BASE=Path('audits/attention_relational_report_v1/subagent_auditor_v5')
OUT=BASE/'qualification/grading/reviewer_b'
P=json.loads((BASE/'qualification/preparation/preparation.json').read_text())
C=json.loads((BASE/'qualification/collection/first_collection.json').read_text())
D=json.loads((BASE/'qualification/derived/first_derived.json').read_text())
K={}
for phase,path in [('short',Path('audits/attention_relational_report_v1/grounded_auditor_v4/qualification/first_materialization/hidden_atomic_key.json')),('long',BASE/'long_qualification/first_materialization/hidden_atomic_key.json')]:
 K.update({(phase,c['case_id']):c for c in json.loads(path.read_text())['cases']})
PUBLIC={(phase,c['case_id']):c for phase,v in P['public_cases'].items() for c in v['cases']}
REC={r['request_id']:r for r in C['records']}
ROWS=D['rows']
def show(r, sources=False):
 key=K[r['phase'],r['case_id']];public=PUBLIC[r['phase'],r['case_id']];ann=REC[r['request_id']]['annotation_file'].get('parsed')
 print('\nPOSITION',r['stream'],r['phase'],r['case_id'],r['request_id'],'VALID',r['mechanical_contract_established'],'clean',key.get('clean',key.get('clean_control')))
 print('ERRORS',r['derived_evidence'].get('errors'),r.get('artifact_issues'))
 print('REPORT',public.get('report_text',public.get('report')))
 print('GOLD')
 for a in key['mandatory_atomic_assertions']:
  print(a['assertion_id'],a['expected_verdict'],a['statement'],'OPERANDS',json.dumps(a['required_operands'],separators=(',',':')),'FACTS',json.dumps(a.get('required_packet_facts'),separators=(',',':')))
 print('ACTUAL')
 for a in ann['assertions'] if ann else []: print(a['assertion_id'],a['verdict'],a['paraphrase'],'REASON:',a['reason'],'CTX',a['source_context_ids'])
 if sources:
  for a in r['derived_evidence'].get('derived_assertions',[]):
   print('SOURCES',a['assertion']['assertion_id'],json.dumps(a['resolved_contexts'],separators=(',',':')),json.dumps(a['operand_contexts'],separators=(',',':')))
if __name__=='__main__':
 if sys.argv[1]=='summary':
  for i,r in enumerate(ROWS):
   k=K[r['phase'],r['case_id']];print(i,r['stream'],r['phase'],r['case_id'],'valid',r['mechanical_contract_established'],'atoms',len(k['mandatory_atomic_assertions']),'keyfields',list(k) if i==0 else '', 'errors',r['derived_evidence'].get('errors'))
 else:
  for i in map(int,sys.argv[1].split(',')):show(ROWS[i],len(sys.argv)>2)
