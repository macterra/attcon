import json,sys
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_11.json','861514d2ea78d2073085b44a0042c46c17bc60a0205e9640bb087b5067d1d6a2')
i=c['ordered_template_ids'][int(sys.argv[1])]
for p in sys.argv[2:]:print(json.dumps(w.source_view(c,i,json.loads(p),0,10000),ensure_ascii=False,separators=(',',':')))
