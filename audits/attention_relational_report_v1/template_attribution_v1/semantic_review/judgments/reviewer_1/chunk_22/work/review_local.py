import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_22.json'
SHA='45748ba13e82811ac3b495f913a9ef3e2f30824dc5303f5deea085b7f0316395'
WORK=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_22/work')
c=w.bound_chunk(CHUNK,SHA)
n=int(sys.argv[1]);r=c['records'][n];mode=sys.argv[2]
if mode=='view':
 p=json.loads(sys.argv[3]);o=int(sys.argv[4]) if len(sys.argv)>4 else 0;l=int(sys.argv[5]) if len(sys.argv)>5 else 20
 print(json.dumps(w.source_view(c,r['id'],p,o,l)))
elif mode=='save-unestablished':
 assert r['reference_established'] is False
 rec=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original annotation is not established. No semantic salvage. Original diagnostics: '+json.dumps(r['reference_measurement'],sort_keys=True))
 raw=(json.dumps(rec,ensure_ascii=True,indent=2)+'\n').encode()
 print(json.dumps(w.save_first_record(c,r['id'],WORK,raw)))
elif mode=='inventory':
 for k in ['id','unit_id','text_sha256','reference_established','review_state']:print(k,json.dumps(w.source_view(c,r['id'],[k],0,100)))
 for k in ['components','evidence','bindings']:print(k,len(r[k]))
