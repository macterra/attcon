"""Unchanged complete-interface inputs with newly qualified process auditor."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import torch
import neutral_functional_reports as v1
from neutral_functional_reports_v3 import complete_record, GLOSSARY, WORD_LIMIT

ROOT=Path('audits/neutral_functional_pilot_v6')


def build_requests(samples,stored):
    requests=[]
    for sample in samples:
        if sample['row']!=3:continue
        for variant in v1.VARIANTS:
            source=complete_record(sample,stored,variant)
            requests.append({'id':f"{sample['visual_seed']}_r3_c{sample['physical_owner']}_{variant}",
                'visual_seed':sample['visual_seed'],'row':3,'physical_owner':sample['physical_owner'],
                'variant':variant,'source':source,'input':GLOSSARY+'\n'+json.dumps(source,separators=(',',':'))})
    assert len(requests)==36
    return requests


def prepare():
    path=ROOT/'requests.json'
    if path.exists():return json.loads(path.read_text())
    torch.set_num_threads(2)
    samples=json.loads((v1.MODEL_ROOT/'samples.json').read_text())
    stored=torch.load(v1.MODEL_ROOT/'states.pt',weights_only=True)
    requests=build_requests(samples,stored)
    ROOT.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(requests,indent=2)+'\n')
    paths=('scripts/neutral_functional_reports_v6.py','scripts/neutral_functional_reports_v3.py',
        'scripts/neutral_functional_reports_v2.py','scripts/neutral_functional_reports.py',
        'src/attcon/functional_model.py','scripts/extract_functional_prose_v3.py','scripts/extract_functional_prose_v1.py',
        'scripts/extract_functional_process_v1.py','scripts/extract_functional_process_v2.py',
        'scripts/assess_functional_prose_v6.py','scripts/assess_functional_prose.py',
        'docs/NEUTRAL_FUNCTIONAL_PILOT_V6.md')
    sources={p:Path(p).read_text() for p in paths}
    (ROOT/'source_code.json').write_text(json.dumps(sources,indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps({'model':v1.MODEL,'reasoning':'medium',
        'max_output_tokens':8192,'word_limit':WORD_LIMIT,'max_attempts':36,'no_retries':True,
        'source_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()},
        'parent_states_sha256':hashlib.sha256((v1.MODEL_ROOT/'states.pt').read_bytes()).hexdigest(),
        'parent_samples_sha256':hashlib.sha256((v1.MODEL_ROOT/'samples.json').read_bytes()).hexdigest(),
        'requests_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'development_row':3,'reporter_glossary_sha256':hashlib.sha256(GLOSSARY.encode()).hexdigest()},indent=2)+'\n')
    return requests


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','generate'),required=True)
    args=p.parse_args();requests=prepare()
    if args.stage=='generate':
        from extract_functional_process_v2 import assess_fixtures
        if not assess_fixtures(Path('audits/functional_process_extractor_v2_fixtures')):
            raise SystemExit('fixture failures block generation')
        from extract_functional_prose_v1 import assess_fixtures as assess_attributes
        if not assess_attributes(Path('audits/functional_extractor_v1_fixtures')):
            raise SystemExit('attribute fixture failures block generation')
        v1.ROOT=ROOT;asyncio.run(v1.generate(requests))
    else:print('Prepared',len(requests),'fresh-row complete-interface requests')


if __name__=='__main__':main()
