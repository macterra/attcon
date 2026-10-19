import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_02.json','f542f1185f57db3825509f6572db4c40cf4f54c0b859b2511f95eaa1acecf7da');n=int(sys.argv[1]);i=c['ordered_template_ids'][n]
r=next(x for x in c['records'] if x['id']==i)
assert r['reference_established'] is False and r['review_state']=='unestablished'
assert r['components']==[] and r['evidence']==[] and r['bindings']==[]
v={'id':i,'unit_id':r['unit_id'],'text_sha256':r['text_sha256'],'reference_established':False,'component_reviews':[],'evidence_reviews':[],'binding_reviews':[],'additional_concerns':[],'notes':'unestablished. No semantic salvage or feature negatives inferred. Original diagnostics preserved: '+json.dumps(r['reference_measurement'],ensure_ascii=False,sort_keys=True)}
w='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_02/work'
p=w+f'/authored_{n:02}.json';raw=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();open(p,'xb').write(raw)
print(h.save_first_record(c,i,w,raw))
