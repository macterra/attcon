import json,sys
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_01.json'
H='5e5797c6c195d8a7e569f94e3d6455a34e81423f1c2fc5feb7da7c8215fd84fb'
c=w.bound_chunk(C,H);r=c['records'][int(sys.argv[1])]
paths=json.loads(sys.argv[2])
for p in paths:
 v=w.source_view(c,r['id'],p,0,10000)
 print(json.dumps(v,ensure_ascii=False))
