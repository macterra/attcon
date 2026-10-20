import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
C=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_20.json')
SHA='dce8ec6b7058f0946a47c90706ae9815542e2ec0f0471fc6e85f01281cf64d0d'
c=w.bound_chunk(C,SHA)
r=c['records'][int(sys.argv[1])]
mode=sys.argv[2]
def view(p,o=0,n=10000):
 print(json.dumps(w.source_view(c,r['id'],p,o,n),ensure_ascii=False))
if mode=='report':
 view(['text'],int(sys.argv[3]) if len(sys.argv)>3 else 0,int(sys.argv[4]) if len(sys.argv)>4 else 10000)
elif mode=='meta':
 for p in ['id','unit_id','text_sha256','reference_established','review_state']:view([p])
 view(['reference_measurement','derivation','errors'])
 view(['reference_measurement','derivation','semantic_validation'])
 print('inventories',[(k,len(r[k])) for k in ['components','evidence','bindings']])
 q=w.source_view(c,r['id'],['reference_measurement','derivation','quote_diagnostics'],0,10000)
 print(json.dumps(dict(total=q['total_items'],complete=q['complete'],quote_errors=[dict(path=v['path'],errors=v['errors']) for v in q['value']])))
elif mode=='view':view(json.loads(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 0,int(sys.argv[5]) if len(sys.argv)>5 else 20)
elif mode=='save-unestablished':
 assert r['reference_established'] is False and r['review_state']=='unestablished'
 assert not r['components'] and not r['evidence'] and not r['bindings']
 d=r['reference_measurement']['derivation']
 notes='unestablished; original frozen annotation is not established. No semantic labels salvaged. Original derivation diagnostics: '+json.dumps(d,ensure_ascii=False,sort_keys=True)
 out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes=notes)
 raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode()
 print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,raw)))
