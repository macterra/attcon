import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_00.json'
SHA='1634476fd3bd669ab467dad2137fbe29b50b2a6064d3b412c13d003578ccc3f6'
W=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_00/work')
c=w.bound_chunk(C,SHA)
def view(i,p,lim=4000):
 id=c['ordered_template_ids'][i]; off=0
 while True:
  v=w.source_view(c,id,p,off,lim)
  print(json.dumps(v,ensure_ascii=False))
  if v['next_offset'] is None:break
  off=v['next_offset']
def save(i,record):
 r=c['records'][i]
 raw=(json.dumps(record,indent=2,ensure_ascii=False)+'\n').encode()
 print(w.save_first_record(c,r['id'],W,raw))
def invalid(i):
 r=c['records'][i]
 assert not r['reference_established']
 save(i,dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics retained verbatim: '+json.dumps(r['reference_measurement'],sort_keys=True)))
if __name__=='__main__':
 i=int(sys.argv[1]); cmd=sys.argv[2]
 if cmd=='text':view(i,['text'])
 elif cmd=='state':
  for p in [['reference_established'],['review_state'],['reference_measurement']]:view(i,p)
 elif cmd=='semantic':
  for p in [['canonical_first_annotation'],['components'],['evidence'],['bindings']]:view(i,p,2)
 elif cmd=='view':view(i,json.loads(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 4)
 elif cmd=='invalid':invalid(i)
