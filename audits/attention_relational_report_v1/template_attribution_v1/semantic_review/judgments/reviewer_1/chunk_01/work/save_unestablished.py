import json,sys
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_01.json';H='5e5797c6c195d8a7e569f94e3d6455a34e81423f1c2fc5feb7da7c8215fd84fb';D='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_01/work'
c=w.bound_chunk(C,H);r=c['records'][int(sys.argv[1])]
assert r['reference_established'] is False and r['review_state']=='unestablished'
assert r['components']==[] and r['evidence']==[] and r['bindings']==[]
review={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
review.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes=sys.argv[2]+' Original derivation diagnostics: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=False,sort_keys=True))
raw=(json.dumps(review,ensure_ascii=False,indent=2)+'\n').encode()
print(json.dumps(w.save_first_record(c,r['id'],D,raw)))
