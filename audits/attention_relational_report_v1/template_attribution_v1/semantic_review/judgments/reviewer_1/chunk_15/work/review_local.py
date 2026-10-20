import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_15.json'
S='c0d951e9d053c8952703280c254ff6c7619a920d2261a820975e40a10d106d45'
W=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_15/work')
c=h.bound_chunk(C,S)
i=int(sys.argv[1]); ident=c['ordered_template_ids'][i]
def v(path,limit=10000,offset=0):return h.source_view(c,ident,path,offset=offset,limit=limit)
if sys.argv[2]=='view':
 path=json.loads(sys.argv[3]);print(json.dumps(v(path,int(sys.argv[4]) if len(sys.argv)>4 else 10000,int(sys.argv[5]) if len(sys.argv)>5 else 0),indent=2))
elif sys.argv[2]=='diagnostics':
 for p in [['id'],['unit_id'],['text_sha256'],['review_state'],['reference_established'],['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation'],['components'],['evidence'],['bindings']]:print(json.dumps(v(p)))
elif sys.argv[2]=='save-unestablished':
 assert v(['reference_established'])['value'] is False
 for p in ['components','evidence','bindings']:assert v([p])['value']==[]
 d=v(['reference_measurement','derivation'])['value']
 out=dict(id=ident,unit_id=v(['unit_id'])['value'],text_sha256=v(['text_sha256'])['value'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original derivation diagnostics preserved without semantic salvage: '+json.dumps(d,sort_keys=True))
 print(json.dumps(h.save_first_record(c,ident,W,json.dumps(out,indent=2).encode()+b'\n')))
