import sys,json
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_16.json','1ba8f3a46e99d796822029fae67f6b8a1bfad4501de7131fd67364f20113f37f');i=c['ordered_template_ids'][int(sys.argv[1])]
def v(p):return h.source_view(c,i,p,0,10000)['value']
assert v(['reference_established']) is False
assert v(['review_state'])=='unestablished'
assert v(['components'])==v(['evidence'])==v(['bindings'])==[]
r=dict(id=i,unit_id=v(['unit_id']),text_sha256=v(['text_sha256']),reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='Unestablished original; no semantic salvage. Original derivation diagnostics retained verbatim as JSON: '+json.dumps(v(['reference_measurement','derivation']),ensure_ascii=True,sort_keys=True))
print(h.save_first_record(c,i,'audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_16/work',json.dumps(r,ensure_ascii=True,indent=2).encode()+b'\n'))
