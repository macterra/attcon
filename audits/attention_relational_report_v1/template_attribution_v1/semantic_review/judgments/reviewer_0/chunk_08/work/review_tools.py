import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_08.json'
SHA='9d42eb4604fb29619be14b21681e582d1d9864dc34aa4d37a64030ebf89e0cf3'
WORK=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_08/work')
c=w.bound_chunk(CHUNK,SHA)
def view(n,path,offset=0,limit=100):
 v=w.source_view(c,c['ordered_template_ids'][n],path,offset,limit)
 print(json.dumps({k:x for k,x in v.items() if k!='value'}));print(json.dumps(v['value'],ensure_ascii=False) if not isinstance(v['value'],str) else v['value'])
def base(n):
 r=c['records'][n]
 assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
 return {k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
def save(n,v):
 p=WORK/('draft_'+str(n)+'.json')
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
 print(w.save_first_record(c,v['id'],WORK,p.read_bytes()))
def unestablished(n):
 r=c['records'][n];assert not r['reference_established'];assert not r['components'] and not r['evidence'] and not r['bindings']
 v=base(n);v.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics retained without semantic salvage: '+json.dumps(r['reference_measurement']['derivation'],sort_keys=True))
 save(n,v)
if __name__=='__main__':
 for n in [0,1]:unestablished(n)
