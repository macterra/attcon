import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_04.json','13699cc07b4f19a63c6400dec8326be2257bfa016408084f0d6bb58af34741b3')
r=c['records'][int(sys.argv[1])]
assert r['reference_established'] is False
assert not r['components'] and not r['evidence'] and not r['bindings']
d=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; '+sys.argv[2]+' Original diagnostics preserved without semantic salvage: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=False))
print(h.save_first_record(c,r['id'],'audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_04/work',(json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode()))
