import json,sys
from pathlib import Path
B=Path('audits/attention_relational_report_v1/grounded_auditor_v4/qualification')
def read(p):return json.loads((B/p).read_text())
d=read('derived/first_derived.json');k={c['case_id']:c for c in read('first_materialization/hidden_atomic_key.json')['cases']}; p={c['case_id']:c for c in read('first_materialization/public_cases.json')['cases']}
for r in d['rows']:
 n=int(r['case_id'][-3:])
 if not (int(sys.argv[1])<=n<=int(sys.argv[2])):continue
 print('\nREQUEST',r['auditor'],r['case_id'],r['request_id'],'CLEAN',k[r['case_id']]['fully_supported_control'])
 print('REPORT',p[r['case_id']]['report_text'])
 e=r['derived_evidence'];print('FLAGS',e['complete_assertion_inventory'],e['unresolved_ambiguity'])
 for da in e['derived_assertions']:
  a=da['assertion'];print('ASSERT',a['assertion_id'],a['verdict'],'PARA:',a['paraphrase'],'CTX',','.join(a['source_context_ids']),'REASON:',a['reason'])
 cs={c['context_id']:c for da in e['derived_assertions'] for c in da['resolved_contexts']+da['operand_contexts']}
 for cid,c in cs.items():print('CONTEXT',cid,json.dumps(c,separators=(',',':')))
