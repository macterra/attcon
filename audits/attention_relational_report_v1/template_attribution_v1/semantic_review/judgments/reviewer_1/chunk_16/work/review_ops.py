import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_16.json'
S='1ba8f3a46e99d796822029fae67f6b8a1bfad4501de7131fd67364f20113f37f'
W=Path(__file__).parent
c=w.bound_chunk(C,S);n=int(sys.argv[1]);i=c['ordered_template_ids'][n]
def v(p,offset=0,limit=10000):return w.source_view(c,i,p,offset,limit)
a=sys.argv[2]
if a=='text':print(json.dumps(v(['text'])))
elif a=='meta':
 for p in [['reference_established'],['review_state'],['components'],['evidence'],['bindings'],['reference_measurement','derivation']]:print(json.dumps(v(p)))
elif a=='save-unestablished':
 assert v(['reference_established'])['value'] is False
 assert v(['review_state'])['value']=='unestablished'
 for p in ['components','evidence','bindings']:assert v([p])['value']==[]
 d=v(['reference_measurement','derivation'])['value']
 r=dict(id=i,unit_id=v(['unit_id'])['value'],text_sha256=v(['text_sha256'])['value'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; '+sys.argv[3]+' Original derivation diagnostics retained without semantic salvage: '+json.dumps(d,separators=(',',':')))
 raw=(json.dumps(r,indent=2)+'\n').encode();print(json.dumps(w.save_first_record(c,i,W,raw)))
