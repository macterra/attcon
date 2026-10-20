import sys,json,pathlib
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk(pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_18.json'),'00f7c30a0463a31224f1d86edb5794dd76a4b87538d05fbddbb568b7c13d2ac1')
r=c['records'][int(sys.argv[1])];assert r['reference_established'] is False and r['review_state']=='unestablished'
v=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics retained unchanged below. No semantic inventory or salvaged label. '+json.dumps(r['reference_measurement'],sort_keys=True,ensure_ascii=True))
print(h.save_first_record(c,r['id'],pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_18/work'),(json.dumps(v,indent=2)+'\n').encode()))
