import json,pathlib,hashlib,math
BASE=pathlib.Path('audits/attention_relational_report_v1/subagent_auditor_v5'); OUT=BASE/'qualification/grading/reviewer_a'
P=json.loads((BASE/'qualification/preparation/preparation.json').read_text());C=json.loads((BASE/'qualification/collection/first_collection.json').read_text());D=json.loads((BASE/'qualification/derived/first_derived.json').read_text())
KEYS={}
for phase,path in [('short',pathlib.Path('audits/attention_relational_report_v1/grounded_auditor_v4/qualification/first_materialization/hidden_atomic_key.json')),('long',BASE/'long_qualification/first_materialization/hidden_atomic_key.json')]:
 KEYS[phase]={x['case_id']:x for x in json.loads(path.read_text())['cases']}
REQUESTS={r['id']:json.loads(r['input'][r['input'].index('{"packet"'):]) for b in P['public_batches'].values() for r in b['requests']}
RECORDS={r['request_id']:r for r in C['records']}
CASES={phase:{x['case_id']:x for x in P['public_cases'][phase]['cases']} for phase in ['short','long']}
def stats(m):
 peak=max(map(max,m));return {'peak_probability':peak,'peaks':[[i,j]for i,row in enumerate(m)for j,v in enumerate(row)if v==peak],'entropy_nats':-sum(v*math.log(v)for row in m for v in row if v),'expected_row':sum(i*v for i,row in enumerate(m)for v in row),'expected_column':sum(j*v for row in m for j,v in enumerate(row))}
def compact(ctx):
 return {k:v for k,v in ctx.items() if k not in ['packet_path','supplied_derived','supplied_relation']}
def ann(row):return RECORDS[row['request_id']]['annotation_file']['parsed']
def contexts(row,a):
 cs={c['context_id']:c for c in REQUESTS[row['request_id']]['source_contexts']};out=[]
 for cid in a['source_context_ids']:
  if cid in cs:
   out.append(compact(cs[cid]));out.extend(compact(cs[oid])for oid in cs[cid].get('operands',[])if oid in cs)
  else:out.append({'missing':cid})
 return out
if __name__=='__main__':
 print('REQUEST shape',next(iter(REQUESTS.values())).keys())
 for r in D['rows'][48:]:print(r['stream'],r['case_id'],r['mechanical_contract_established'],r['derived_evidence']['errors'],len(ann(r)['assertions']))
