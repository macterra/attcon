import sys,json,hashlib,pathlib
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_21.json'
S='1f945495ca5a50790b1e50665e0c5539a2e5f87d10417833fff68d136e4fd479'
c=w.bound_chunk(C,S); n=int(sys.argv[1]);r=c['records'][n]; i=r['id']; stage=sys.argv[2]
def view(p,limit=100): print(json.dumps(w.source_view(c,i,p,0,limit),ensure_ascii=False))
if stage=='text':
 for offset in range(0,len(r['text']),10000): print(json.dumps(w.source_view(c,i,['text'],offset,10000),ensure_ascii=False))
elif stage=='diagnostic':
 for p in [['id'],['unit_id'],['text_sha256'],['reference_established'],['review_state'],['canonical_first_annotation'],['components'],['evidence'],['bindings'],['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation']]:view(p)
elif stage=='save-unestablished':
 assert r['reference_established'] is False and r['canonical_first_annotation'] is None and r['components']==[] and r['evidence']==[] and r['bindings']==[]
 assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
 a={'id':r['id'],'unit_id':r['unit_id'],'text_sha256':r['text_sha256'],'reference_established':False,'component_reviews':[],'evidence_reviews':[],'binding_reviews':[],'additional_concerns':['Original annotation remains unestablished; no semantic labels are salvaged.'],'notes':'Unestablished original. Original diagnostics: '+json.dumps(r['reference_measurement']['derivation'],sort_keys=True)}
 p=pathlib.Path(__file__).parent;q=p/f'input_{n:02}.json';q.write_text(json.dumps(a));print(json.dumps(w.save_first_record(c,i,p,q.read_bytes())))
