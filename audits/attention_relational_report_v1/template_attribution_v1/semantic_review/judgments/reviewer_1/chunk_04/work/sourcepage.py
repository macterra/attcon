import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_04.json'
SHA='13699cc07b4f19a63c6400dec8326be2257bfa016408084f0d6bb58af34741b3'
c=h.bound_chunk(CHUNK,SHA)
n=int(sys.argv[1]);i=c['ordered_template_ids'][n]
p=json.loads(sys.argv[2]);off=int(sys.argv[3]) if len(sys.argv)>3 else 0;lim=int(sys.argv[4]) if len(sys.argv)>4 else 20
v=h.source_view(c,i,p,off,lim)
print(json.dumps(v,ensure_ascii=False))
