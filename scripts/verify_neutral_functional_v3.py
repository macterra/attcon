"""Exact archive replay, source blindness and pre-data provenance checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
import neutral_functional_reports as v1
from neutral_functional_reports_v3 import ROOT, build_requests
from verify_neutral_functional_v2 import raw_text
from extract_functional_prose_v2 import assess_fixtures


def verify_prepared():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    sources=json.loads((ROOT/'source_code.json').read_text())
    for p,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(sources[p].encode()).hexdigest()==digest,p
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
    for filename,key in [('samples.json','parent_samples_sha256'),('states.pt','parent_states_sha256')]:
        assert hashlib.sha256((v1.MODEL_ROOT/filename).read_bytes()).hexdigest()==manifest[key]
    assert hashlib.sha256((ROOT/'requests.json').read_bytes()).hexdigest()==manifest['requests_sha256']
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
    fixture_root=Path('audits/functional_extractor_v2_fixtures')
    assert assess_fixtures(fixture_root)
    audit_root=ROOT/'extraction_v2'
    audit_manifest=json.loads((audit_root/'manifest.json').read_text())
    assert hashlib.sha256(audit_manifest['source'].encode()).hexdigest()==audit_manifest['source_sha256']
    assert audit_manifest['source']==Path('scripts/extract_functional_prose_v2.py').read_text()
    audit_requests=json.loads((audit_root/'requests.json').read_text())
    for i,req in enumerate(requests):
        response=json.loads((ROOT/(req['id']+'.json')).read_text())
        assert response['response']['status']=='completed'
        assert response['response']['model']==manifest['model']
        assert response['report']==raw_text(response['response'])
        assert audit_requests[i]=={'id':f'r{i:03d}','report':response['report']}
        audit=json.loads((audit_root/f'r{i:03d}.json').read_text())
        assert audit['response']['status']=='completed' and audit['response']['model']==audit_manifest['model']
        assert audit['parsed']==json.loads(raw_text(audit['response']))
    before=(ROOT/'assessment.json').read_bytes()
    subprocess.run([sys.executable,'scripts/assess_functional_prose_v3.py'],check=True,stdout=subprocess.DEVNULL)
    assert (ROOT/'assessment.json').read_bytes()==before
    print('Verified 36 raw reports, 36 source-blind extractions, fixture pass and exact scoring replay')


if __name__=='__main__':main()
