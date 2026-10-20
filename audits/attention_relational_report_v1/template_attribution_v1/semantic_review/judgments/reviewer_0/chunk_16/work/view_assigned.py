import sys,json
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_16.json','1ba8f3a46e99d796822029fae67f6b8a1bfad4501de7131fd67364f20113f37f')
i=c['ordered_template_ids'][int(sys.argv[1])]
for spec in sys.argv[2:]:
 p=json.loads(spec);r=h.source_view(c,i,p,0,10000)
 print(json.dumps(r,ensure_ascii=True))
