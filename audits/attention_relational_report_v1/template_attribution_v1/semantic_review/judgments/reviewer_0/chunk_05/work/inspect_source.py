import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_05.json','9821af4a85f4f4f5c5b4fd00d5516a46fcb0e0142af0b0629c8a5fda898b2bc9')
i=c['ordered_template_ids'][int(sys.argv[1])]
def show(p,lim=100,off=0):
 v=w.source_view(c,i,p,off,lim)
 print(json.dumps({'path':p,'offset':off,'total':v['total_items'],'next':v['next_offset'],'value':v['value']},ensure_ascii=False))
if sys.argv[2]=='text':show(['text'],10000)
elif sys.argv[2]=='metadata':
 for p in [['id'],['unit_id'],['text_sha256'],['reference_established'],['review_state']]:show(p)
 if not c['records'][int(sys.argv[1])]['reference_established']:
  for p in [['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation'],['canonical_first_annotation'],['components'],['evidence'],['bindings']]:show(p)
 else:
  show(['components'],12);show(['bindings'],4);show(['evidence'],1)
else:
 show(json.loads(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 1,int(sys.argv[4]) if len(sys.argv)>4 else 0)
