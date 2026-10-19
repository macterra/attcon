import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_05.json'
S='9821af4a85f4f4f5c5b4fd00d5516a46fcb0e0142af0b0629c8a5fda898b2bc9'
W='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_05/work'
c=w.bound_chunk(C,S)
n=int(sys.argv[1]);i=c['ordered_template_ids'][n-1]
def v(path): return w.source_view(c,i,path,0,10000)['value']
assert v(['reference_established']) is False
r={'id':i,'unit_id':v(['unit_id']),'text_sha256':v(['text_sha256']),'reference_established':False,'component_reviews':[],'evidence_reviews':[],'binding_reviews':[],'additional_concerns':['Original unestablished annotation retained without semantic salvage.'],'notes':sys.argv[2]+' Original review_state='+v(['review_state'])+'. Original derivation diagnostics='+json.dumps(v(['reference_measurement','derivation']),ensure_ascii=True,separators=(',',':'))}
raw=(json.dumps(r,ensure_ascii=True,indent=2)+'\n').encode()
print(json.dumps(w.save_first_record(c,i,W,raw)))
