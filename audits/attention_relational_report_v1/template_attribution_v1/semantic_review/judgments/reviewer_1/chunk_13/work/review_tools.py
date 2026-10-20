import sys,json,pathlib,hashlib
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
SHA='cda7fcb6e29aa7076403202a25c76a764aab8ea322c91eaa90dc38fb06448bf0'
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_13.json',SHA)
work=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_13/work')
def view(n,mode):
 r=c['records'][n-1];i=r['id']
 if mode=='text':
  offset=0
  while True:
   v=w.source_view(c,i,['text'],offset,10000);print(json.dumps(v))
   if v['next_offset'] is None: break
   offset=v['next_offset']
 elif mode=='detail':
  for k in ['reference_established','canonical_first_annotation','components','evidence','bindings']:
   print(json.dumps(w.source_view(c,i,[k],0,100)))
  if r['reference_established']:
   ids={x['unit_id'] for e in r['evidence'] for x in e['geometry']['intersections']}
   for j,p in enumerate(r['property_table']):
    if p['unit_id'] in ids:print(json.dumps(w.source_view(c,i,['property_table',j],0,100)))
  else:print(json.dumps(w.source_view(c,i,['reference_measurement'],0,100)))
 elif mode=='save-unestablished':
  assert r['reference_established'] is False
  assert r['canonical_first_annotation'] is None and not r['components'] and not r['evidence'] and not r['bindings']
  assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
  d=r['reference_measurement']['derivation']
  out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics retained without semantic salvage: '+json.dumps(d,sort_keys=True))
  p=work/('draft_%02d.json'%n);p.write_text(json.dumps(out,indent=2)+'\n');print(w.save_first_record(c,r['id'],work,p.read_bytes()))
if __name__=='__main__':view(int(sys.argv[1]),sys.argv[2])
