"""Exact archive replay, source blindness and pre-data provenance checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
import neutral_functional_reports as v1
from neutral_functional_reports_v6 import ROOT, build_requests, GLOSSARY
from verify_neutral_functional_v2 import raw_text
from extract_functional_process_v2 import assess_fixtures
from extract_functional_prose_v1 import assess_fixtures as attribute_fixtures


def verify_prepared():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    sources=json.loads((ROOT/'source_code.json').read_text())
    for p,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(sources[p].encode()).hexdigest()==digest,p
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
    for filename,key in [('samples.json','parent_samples_sha256'),('states.pt','parent_states_sha256')]:
        assert hashlib.sha256((v1.MODEL_ROOT/filename).read_bytes()).hexdigest()==manifest[key]
    assert hashlib.sha256((ROOT/'requests.json').read_bytes()).hexdigest()==manifest['requests_sha256']
    assert manifest['development_row']==3
    assert manifest['reporter_glossary_sha256']==hashlib.sha256(GLOSSARY.encode()).hexdigest()
    samples=json.loads((v1.MODEL_ROOT/'samples.json').read_text())
    stored=torch.load(v1.MODEL_ROOT/'states.pt',weights_only=True)
    requests=json.loads((ROOT/'requests.json').read_text())
    assert requests==build_requests(samples,stored)
    print('Verified 36 prepared inputs and frozen source/state hashes')
    return manifest,requests



def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);manifest,requests=verify_prepared()
    if args.prepared_only:return
    assert assess_fixtures(Path('audits/functional_process_extractor_v2_fixtures'))
    assert attribute_fixtures(Path('audits/functional_extractor_v1_fixtures'))
    auditor_specs=[('extraction_v1','scripts/extract_functional_prose_v1.py'),
        ('process_extraction_v2','scripts/extract_functional_process_v2.py')]
    for dirname,source_path in auditor_specs:
        root=ROOT/dirname;m=json.loads((root/'manifest.json').read_text())
        assert hashlib.sha256(m['source'].encode()).hexdigest()==m['source_sha256']
        assert m['source']==Path(source_path).read_text()
        audit_requests=json.loads((root/'requests.json').read_text())
        for i,req in enumerate(requests):
            response=json.loads((ROOT/(req['id']+'.json')).read_text())
            assert response['response']['status']=='completed' and response['response']['model']==manifest['model']
            assert response['report']==raw_text(response['response'])
            assert audit_requests[i]=={'id':f'r{i:03d}','report':response['report']}
            audit=json.loads((root/f'r{i:03d}.json').read_text())
            assert audit['response']['status']=='completed' and audit['response']['model']==m['model']
            assert audit['parsed']==json.loads(raw_text(audit['response']))
    from extract_functional_process_v2 import valid_claim
    for i in range(len(requests)):
        process=json.loads((ROOT/'process_extraction_v2'/f'r{i:03d}.json').read_text())
        assert all(valid_claim(c) for c in process['parsed']['process_claims'])
    before=(ROOT/'assessment.json').read_bytes()
    subprocess.run([sys.executable,'scripts/assess_functional_prose_v6.py'],check=True,stdout=subprocess.DEVNULL)
    assert (ROOT/'assessment.json').read_bytes()==before
    print('Verified 36 raw reports, both sets of 36 source-blind audits, qualifications and exact scoring replay')


if __name__=='__main__':main()
