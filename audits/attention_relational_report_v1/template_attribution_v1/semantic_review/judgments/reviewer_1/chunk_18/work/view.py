import sys,json,pathlib
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk(pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_18.json'),'00f7c30a0463a31224f1d86edb5794dd76a4b87538d05fbddbb568b7c13d2ac1');r=c['records'][int(sys.argv[1])];i=r['id']
def view(p,offset=0,limit=100):
 v=h.source_view(c,i,p,offset,limit);print(json.dumps({k:v[k] for k in ['id','source_path','offset','total_items','page_end','next_offset','value']}))
m=sys.argv[2]
if m=='text':view(['text'],int(sys.argv[3]) if len(sys.argv)>3 else 0,10000)
elif m=='meta':
 for p in [['reference_established'],['unit_id'],['text_sha256'],['canonical_first_annotation'],['components'],['bindings']]:view(p)
 if not r['reference_established']:
  for p in [['review_state'],['evidence'],['reference_measurement']]:view(p)
elif m=='evidence':view(['evidence'],int(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 4)
elif m=='props':
 ids={t['unit_id'] for e in r['evidence'] for t in e['geometry']['intersections']}
 if len(sys.argv)>3:ids.update(sys.argv[3:])
 for ix,t in enumerate(r['property_table']):
  if t['unit_id'] in ids:view(['property_table',ix])
