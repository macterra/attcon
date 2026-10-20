import sys,json,pathlib,hashlib
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_11.json','861514d2ea78d2073085b44a0042c46c17bc60a0205e9640bb087b5067d1d6a2');r=c['records'][int(sys.argv[1])]
assert r['reference_established'] is False and r['canonical_first_annotation'] is None
assert r['components']==[] and r['evidence']==[] and r['bindings']==[]
assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
assert r['reference_measurement']['id']==r['id'] and r['reference_measurement']['unit_id']==r['unit_id'] and r['reference_measurement']['text_sha256']==r['text_sha256']
x={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']};x.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; matching original derivation diagnostics preserved without salvaging labels: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=False,sort_keys=True))
p=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_11/work')/('record'+sys.argv[1]+'_input.json');p.write_text(json.dumps(x,indent=2)+'\n');print(w.save_first_record(c,r['id'],p.parent,p.read_bytes()))
