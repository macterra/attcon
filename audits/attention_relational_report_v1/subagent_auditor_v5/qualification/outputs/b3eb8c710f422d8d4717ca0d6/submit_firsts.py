import importlib.util,json,hashlib
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/b3eb8c710f422d8d4717ca0d6')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/b3eb8c710f422d8d4717ca0d6.json')
spec=importlib.util.spec_from_file_location('submission_helper','scripts/attention_relational_subagent_submit_v5.py'); helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
b=json.loads(BATCH.read_text())
for position,request in enumerate(b['requests']):
 raw=(ROOT/(request['id']+'.manual.json')).read_bytes()
 receipt=helper.submit(BATCH,ROOT,request['id'],json.loads(raw),raw_bytes=raw)
 print('Submitted position',position,request['id'],receipt['annotation_sha256'])
artifacts=[ROOT/name for name in ['calculate.py','calculations.json','calculation_display.txt','prepare_annotations.py','submit_firsts.py']]
manifest=helper.finalize(BATCH,ROOT,calculator_artifacts=artifacts,limitations=[])
p=ROOT/'first_batch.json'
print('FINAL',str(p),hashlib.sha256(p.read_bytes()).hexdigest())
print('COUNTS',sum(x['annotation'] is not None for x in manifest['positions']),manifest['requested_count'])
