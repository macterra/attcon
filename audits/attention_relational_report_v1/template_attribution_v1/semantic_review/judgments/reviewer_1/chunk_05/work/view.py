import sys,json
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_05.json','9821af4a85f4f4f5c5b4fd00d5516a46fcb0e0142af0b0629c8a5fda898b2bc9')
i=c['ordered_template_ids'][int(sys.argv[1])-1]
for p in sys.argv[2:]:
 a=json.loads(p);path=a[0];offset=a[1];limit=a[2];print(json.dumps(w.source_view(c,i,path,offset,limit)))
