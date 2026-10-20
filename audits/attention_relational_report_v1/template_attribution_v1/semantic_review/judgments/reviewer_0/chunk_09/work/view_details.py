import sys,json
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_09.json','d57c72746c294a14f1b2f09a203a1aa9195dc2be051f6d3569084267f5da56a0');r=c['records'][int(sys.argv[1])]
for key in ['id','unit_id','text_sha256','canonical_first_annotation','components','bindings']:
 v=h.source_view(c,r['id'],[key],limit=10000);print(key,'TOTAL',v['total_items'],'NEXT',v['next_offset']);print(json.dumps(v['value'],ensure_ascii=False))
for i,e in enumerate(r['evidence']):
 v=h.source_view(c,r['id'],['evidence',i],limit=100);print('EVIDENCE',i,'TOTAL',v['total_items'],'NEXT',v['next_offset']);print(json.dumps(v['value'],ensure_ascii=False))
unit_ids={x['unit_id'] for e in r['evidence'] for x in e['geometry']['intersections']}
for i,u in enumerate(r['property_table']):
 if u['unit_id'] in unit_ids:
  v=h.source_view(c,r['id'],['property_table',i],limit=100);print('PROPERTY',i,'TOTAL',v['total_items'],'NEXT',v['next_offset']);print(json.dumps(v['value'],ensure_ascii=False))
