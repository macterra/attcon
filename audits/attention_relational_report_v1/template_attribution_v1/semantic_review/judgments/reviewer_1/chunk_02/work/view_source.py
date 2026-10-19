import json,sys
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_02.json','f542f1185f57db3825509f6572db4c40cf4f54c0b859b2511f95eaa1acecf7da')
i=c['ordered_template_ids'][int(sys.argv[1])]
p=json.loads(sys.argv[2]); off=int(sys.argv[3]) if len(sys.argv)>3 else 0; lim=int(sys.argv[4]) if len(sys.argv)>4 else 20
print(json.dumps(h.source_view(c,i,p,off,lim),ensure_ascii=False,indent=2))
