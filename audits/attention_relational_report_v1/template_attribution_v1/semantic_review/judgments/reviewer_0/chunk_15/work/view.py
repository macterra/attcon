import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_15.json'
SHA='c0d951e9d053c8952703280c254ff6c7619a920d2261a820975e40a10d106d45'
c=w.bound_chunk(C,SHA);i=c['ordered_template_ids'][int(sys.argv[1])]
for s in sys.argv[2:]:
 p=json.loads(s);off=0;limit=10000
 if isinstance(p,dict):off=p.get('offset',0);limit=p.get('limit',10000);p=p['path']
 print(json.dumps(w.source_view(c,i,p,off,limit)))
