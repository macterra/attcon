"""Offline replay of pilot rendering and complete API output archives."""
import hashlib
import json
from pathlib import Path
import torch
from neutral_functional_reports import ROOT, MODEL_ROOT, GLOSSARY, variants


def main():
    torch.set_num_threads(2)
    manifest = json.loads((ROOT/'manifest.json').read_text())
    snapshots = json.loads((ROOT/'source_code.json').read_text())
    for path, digest in manifest['source_sha256'].items():
        assert hashlib.sha256(snapshots[path].encode()).hexdigest() == digest, path
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256((ROOT/'requests.json').read_bytes()).hexdigest() == manifest['requests_sha256']
    assert hashlib.sha256((MODEL_ROOT/'samples.json').read_bytes()).hexdigest() == manifest['parent_samples_sha256']
    requests = json.loads((ROOT/'requests.json').read_text())
    samples = json.loads((MODEL_ROOT/'samples.json').read_text())
    states = torch.load(MODEL_ROOT/'states.pt', weights_only=True)
    source_lookup = {(s['visual_seed'], s['physical_owner']): s for s in samples if s['row'] == 0}
    for req in requests:
        sample = source_lookup[req['visual_seed'], req['physical_owner']]
        source = variants(sample, states)[req['variant']]
        assert source == req['source']
        assert GLOSSARY+'\n'+json.dumps(source, separators=(',', ':')) == req['input']
        result = json.loads((ROOT/(req['id']+'.json')).read_text())
        response = result['response']
        assert response['status'] == 'completed' and response['model'] == manifest['model']
        raw = ''.join(c['text'] for item in response['output'] if item['type'] == 'message'
                      for c in item['content'] if c['type'] == 'output_text')
        assert raw == result['report'], req['id']
    print('All 36 pilot requests and complete raw reports exactly verified')


if __name__ == '__main__':
    main()
