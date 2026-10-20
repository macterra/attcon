import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_10.json'
SHA='fbca36e08a91b90feb57c6b8acaa50249f7752bf16c8a36e77312d53b13f7dad'
W='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_10/work'
c=h.bound_chunk(C,SHA);r=c['records'][int(sys.argv[1])]
if sys.argv[2]=='view':
 for arg in sys.argv[3:]:print(json.dumps(h.source_view(c,r['id'],json.loads(arg),0,10000)))
elif sys.argv[2]=='unestablished':
 assert r['reference_established'] is False and r['review_state']=='unestablished'
 assert r['components']==[] and r['evidence']==[] and r['bindings']==[]
 d=r['reference_measurement']['derivation']
 out={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
 out.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics preserved: '+json.dumps(d,ensure_ascii=True,sort_keys=True))
 print(json.dumps(h.save_first_record(c,r['id'],W,(json.dumps(out,ensure_ascii=True,indent=2)+'\n').encode())))
