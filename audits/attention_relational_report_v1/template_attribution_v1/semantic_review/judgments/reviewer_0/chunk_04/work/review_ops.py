import sys,json,hashlib
from pathlib import Path
H=Path('scripts/attention_relational_template_review_workspace_v1.py')
assert hashlib.sha256(H.read_bytes()).hexdigest()=='20834fcb9c47ff0376088d5800196cbbd42763348a10f329f96a689fad89060b'
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_04.json'
S='13699cc07b4f19a63c6400dec8326be2257bfa016408084f0d6bb58af34741b3'
W=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_04/work')
c=h.bound_chunk(C,S); i=int(sys.argv[2]); ident=c['ordered_template_ids'][i]; r=c['records'][i]
def view(path,offset=0,limit=20):
 v=h.source_view(c,ident,path,offset,limit); print(json.dumps(v,ensure_ascii=False))
if sys.argv[1]=='view':
 view(json.loads(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 0,int(sys.argv[5]) if len(sys.argv)>5 else 20)
elif sys.argv[1]=='text':
 for o in range(0,len(r['text']),4000): view(['text'],o,4000)
 for p in [['reference_established'],['review_state'],['unit_id'],['text_sha256']]:view(p,0,100)
 if not r['reference_established']:
  for p in [['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation']]:view(p,0,100)
 else:
  view(['components'],0,12)
  print('EVIDENCE_COUNT',len(r['evidence']),'BINDING_COUNT',len(r['bindings']))
elif sys.argv[1]=='save_unestablished':
 assert not r['reference_established']
 notes=sys.argv[3]
 d={'id':ident,'unit_id':r['unit_id'],'text_sha256':r['text_sha256'],'reference_established':False,'component_reviews':[],'evidence_reviews':[],'binding_reviews':[],'additional_concerns':[], 'notes':notes+' Original diagnostics: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=False,sort_keys=True)}
 raw=json.dumps(d,ensure_ascii=False,indent=2).encode()+b'\n'
 p=W/('authored_'+str(i)+'.json');p.write_bytes(raw)
 print(h.save_first_record(c,ident,W,raw))
