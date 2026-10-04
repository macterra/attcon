"""Verify retained amendment, named inputs, audit blindness and raw responses."""
import hashlib
import json
from pathlib import Path
import torch
import neutral_functional_reports as v1
from neutral_functional_reports_v2 import ROOT, GLOSSARY, named
from extract_functional_prose_v1 import assess_fixtures


def raw_text(response):
    return ''.join(c['text'] for item in response['output'] if item['type']=='message'
                   for c in item['content'] if c['type']=='output_text')


def main():
    torch.set_num_threads(2)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    snapshots=json.loads((ROOT/'source_code.json').read_text())
    for p,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(snapshots[p].encode()).hexdigest()==digest,p
    amendment=json.loads((ROOT/'renderer_amendment.json').read_text())
    assert hashlib.sha256(amendment['corrected_source'].encode()).hexdigest()==amendment['corrected_source_sha256']
    assert hashlib.sha256(Path('scripts/neutral_functional_reports_v2.py').read_bytes()).hexdigest()==amendment['corrected_source_sha256']
    assert hashlib.sha256((ROOT/'requests.json').read_bytes()).hexdigest()==manifest['requests_sha256']==amendment['unchanged_requests_sha256']
    assert hashlib.sha256((v1.MODEL_ROOT/'samples.json').read_bytes()).hexdigest()==manifest['parent_samples_sha256']
    samples=json.loads((v1.MODEL_ROOT/'samples.json').read_text())
    states=torch.load(v1.MODEL_ROOT/'states.pt',weights_only=True)
    requests=json.loads((ROOT/'requests.json').read_text())
    audit_manifest=json.loads((ROOT/'extraction_v1'/'manifest.json').read_text())
    assert hashlib.sha256(audit_manifest['source'].encode()).hexdigest()==audit_manifest['source_sha256']
    audit_requests=json.loads((ROOT/'extraction_v1'/'requests.json').read_text())
    fixture_root=Path('audits/functional_extractor_v1_fixtures')
    assert assess_fixtures(fixture_root)
    for i,req in enumerate(requests):
        sample=next(s for s in samples if s['row']==1 and s['visual_seed']==req['visual_seed'] and s['physical_owner']==req['physical_owner'])
        source=named(v1.variants(sample,states)[req['variant']])
        assert source==req['source']
        assert req['input']==GLOSSARY+'\n'+json.dumps(source,separators=(',',':'))
        response=json.loads((ROOT/(req['id']+'.json')).read_text())
        assert response['response']['status']=='completed'
        assert response['response']['model']==manifest['model']
        assert response['report']==raw_text(response['response'])
        assert audit_requests[i]=={'id':f'r{i:03d}','report':response['report']}
        audit=json.loads((ROOT/'extraction_v1'/f'r{i:03d}.json').read_text())
        assert audit['response']['status']=='completed' and audit['response']['model']==audit_manifest['model']
        assert audit['parsed']==json.loads(raw_text(audit['response']))
    print('Verified 36 named inputs/reports, 36 source-blind extractions, fixture pass and renderer amendment')


if __name__=='__main__':main()
