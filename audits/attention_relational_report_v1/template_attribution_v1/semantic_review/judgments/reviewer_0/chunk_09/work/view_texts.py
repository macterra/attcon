import sys
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_09.json','d57c72746c294a14f1b2f09a203a1aa9195dc2be051f6d3569084267f5da56a0')
for i in range(int(sys.argv[1]),int(sys.argv[2])):
 r=c['records'][i];print('RECORD',i,r['id'])
 for offset in range(0,len(r['text']),5000):
  v=h.source_view(c,r['id'],['text'],offset=offset,limit=5000);print('PAGE',offset,v['page_end'],'TOTAL',v['total_items'],'NEXT',v['next_offset']);print(v['value'])
