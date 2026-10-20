import json,pathlib,subprocess,sys
idx=int(sys.argv[1]); chunk='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_07.json';r=json.load(open(chunk))['records'][idx]
assert r['reference_established'] is False and r['canonical_first_annotation'] is None
w=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_07/work')
y=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; full report inspected. Original derivation diagnostics preserved without semantic salvage: '+json.dumps(r['reference_measurement']['derivation'],sort_keys=True))
p=w/('authored_%02d.json'%idx)
with p.open('x') as f:f.write(json.dumps(y,ensure_ascii=True,indent=2)+'\n')
subprocess.run(['python3','scripts/attention_relational_template_review_workspace_v1.py','--chunk',chunk,'--chunk-sha256','75ae0b350529484b0168488f74a4d9270567afe75d8ec1c47cd93598dcf8527e','save-first-record','--id',r['id'],'--input',str(p),'--workspace',str(w)],check=True)
