import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_01.json','5e5797c6c195d8a7e569f94e3d6455a34e81423f1c2fc5feb7da7c8215fd84fb');r=c['records'][int(sys.argv[1])]
seen=set()
for i in range(int(sys.argv[2]),min(int(sys.argv[3]),len(r['evidence']))):
 v=w.source_view(c,r['id'],['evidence',i],0,10000);print(json.dumps(v))
 for x in v['value']['geometry']['intersections']:
  uid=x['unit_id']
  if uid not in seen:
   seen.add(uid);j=next(j for j,p in enumerate(r['property_table']) if p['unit_id']==uid)
   print(json.dumps(w.source_view(c,r['id'],['property_table',j],0,10000)))
print('evidence total',len(r['evidence']))
