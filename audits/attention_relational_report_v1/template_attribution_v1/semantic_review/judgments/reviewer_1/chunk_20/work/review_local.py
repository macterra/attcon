import sys,json,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as h
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_20.json'
S='dce8ec6b7058f0946a47c90706ae9815542e2ec0f0471fc6e85f01281cf64d0d'
W=Path(__file__).parent
c=h.bound_chunk(C,S)
i=int(sys.argv[1]); r=c['records'][i]; mode=sys.argv[2]
assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
if mode=='text':
 print('POSITION',i,'ID',r['id'])
 print(json.dumps(h.source_view(c,r['id'],['text'],0,10000)))
elif mode=='path':
 print(json.dumps(h.source_view(c,r['id'],json.loads(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 0,int(sys.argv[5]) if len(sys.argv)>5 else 20)))
elif mode=='details':
 for p in [['reference_established'],['review_state'],['unit_id'],['text_sha256'],['components'],['evidence'],['bindings'],['reference_measurement','derivation']]:
  print(json.dumps(h.source_view(c,r['id'],p,0,10000)))
elif mode=='save-unestablished':
 assert r['reference_established'] is False and r['review_state']=='unestablished'
 assert r['components']==r['evidence']==r['bindings']==[]
 v=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original annotation is not eligible for semantic salvage. Original derivation diagnostics: '+json.dumps(r['reference_measurement']['derivation'],sort_keys=True))
 p=W/f'input_{i:02}.json'
 with p.open('x') as f: json.dump(v,f,indent=2)
 print(h.save_first_record(c,r['id'],W,p.read_bytes()))
