import json,sys,pathlib,hashlib
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_21.json'; H='1f945495ca5a50790b1e50665e0c5539a2e5f87d10417833fff68d136e4fd479'; W=pathlib.Path(__file__).parent
c=w.bound_chunk(C,H); n=int(sys.argv[1]);i=c['ordered_template_ids'][n];r=c['records'][n]
mode=sys.argv[2]
if mode=='text':
 print(json.dumps(w.source_view(c,i,['text'],int(sys.argv[3]) if len(sys.argv)>3 else 0,10000)))
elif mode=='meta':
 for p in [['id'],['unit_id'],['text_sha256'],['reference_established'],['review_state'],['components'],['bindings'],['evidence']]:print(json.dumps(w.source_view(c,i,p,0,100)))
 if not r['reference_established']:
  for p in [['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation']]:print(json.dumps(w.source_view(c,i,p,0,100)))
elif mode=='view': print(json.dumps(w.source_view(c,i,json.loads(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 0,int(sys.argv[5]) if len(sys.argv)>5 else 20)))
elif mode=='unestablished':
 assert r['reference_established'] is False
 judgment=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics retained without semantic salvage: '+json.dumps(r['reference_measurement']['derivation'],sort_keys=True))
 raw=(json.dumps(judgment,ensure_ascii=True,indent=2)+'\n').encode();print(json.dumps(w.save_first_record(c,i,W,raw)))
