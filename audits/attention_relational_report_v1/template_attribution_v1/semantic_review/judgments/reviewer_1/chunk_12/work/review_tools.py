import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
CP='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_12.json'
CS='ae80eb6032940985f0e44b2993e800da8f95bfe7ec26edff50e1d569c3a2026b'
WORK=Path(__file__).parent
x=w.bound_chunk(CP,CS); r=x['records'][int(sys.argv[1])-1];i=r['id'];action=sys.argv[2]
if action=='text':
 assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
 print(json.dumps(w.source_view(x,i,['text'],0,10000)))
elif action=='meta':
 for k in ['id','unit_id','text_sha256','reference_established','review_state','reference_measurement','components','evidence','bindings']:
  print(json.dumps(w.source_view(x,i,[k],0,10000)))
elif action=='unest':
 assert r['reference_established'] is False and r['review_state']=='unestablished'
 assert r['components']==[] and r['evidence']==[] and r['bindings']==[]
 review=dict(id=i,unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics preserved without semantic salvage: '+json.dumps(r['reference_measurement'],sort_keys=True))
 raw=(json.dumps(review,indent=2)+'\n').encode();print(json.dumps(w.save_first_record(x,i,WORK,raw)))
elif action=='view':
 print(json.dumps(w.source_view(x,i,json.loads(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 0,int(sys.argv[5]) if len(sys.argv)>5 else 10)))
