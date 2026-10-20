import sys,json,hashlib
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_10.json'
H='fbca36e08a91b90feb57c6b8acaa50249f7752bf16c8a36e77312d53b13f7dad'
W='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_10/work'
c=h.bound_chunk(C,H)
r=c['records'][int(sys.argv[2])];i=r['id']
assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
def view(p):
 v=h.source_view(c,i,p,int(sys.argv[3]) if len(sys.argv)>3 else 0,10000)
 print(json.dumps(v,ensure_ascii=False))
if sys.argv[1]=='text':view(['text'])
elif sys.argv[1]=='details':
 for p in [['reference_established'],['components'],['evidence'],['bindings']]:view(p)
 if not r['reference_established']:
  for p in [['review_state'],['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation'],['reference_measurement','derivation','quote_diagnostics']]:view(p)
elif sys.argv[1]=='save-unestablished':
 assert r['reference_established'] is False and r['review_state']=='unestablished' and r['components']==[] and r['evidence']==[] and r['bindings']==[]
 j={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
 j.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='Unestablished original annotation; semantic inventories intentionally empty and no labels salvaged. Original diagnostics: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=False,sort_keys=True))
 print(h.save_first_record(c,i,W,(json.dumps(j,ensure_ascii=False,indent=2)+'\n').encode()))
