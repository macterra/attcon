#!/usr/bin/env python3
"""Offline exact source replay, full-response validation, and archive checksums."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

ROOT=Path('audits/bound_content')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def same_tensors(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_tensors(a[k],b[k]) for k in a)
    return a==b


def response_text(response):
    return ''.join(c.get('text','') for output in response['output'] if output['type']=='message' for c in output['content'] if c['type']=='output_text')


def verify_study(root):
    manifest=json.loads((root/'manifest.json').read_text());sources=json.loads((root/'source_code.json').read_text())
    for name,text in sources.items():assert hashlib.sha256(text.encode()).hexdigest()==manifest['source_sha256'][name]
    for path,expected in manifest['checkpoint_sha256'].items():assert digest(Path(path))==expected,path
    assert digest(root/'requests.json')==manifest['requests_sha256']
    assert digest(root/'source_states.pt')==manifest['states_sha256']
    requests=json.loads((root/'requests.json').read_text());assert len(requests)==manifest['config']['max_attempts']
    # Execute only this project's archived generator, locally, with no API entry point.
    with tempfile.TemporaryDirectory() as tmp:
        source_path=Path(tmp)/'frozen_generator.py';source_path.write_text(sources['scripts/bound_reports.py'])
        spec=importlib.util.spec_from_file_location('bound_frozen_generator',source_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config_path=next(Path(p) for p in sources if p.startswith('configs/'))
        fresh_root=Path(tmp)/'replay'
        regenerated=module.prepare(manifest['config'],config_path,fresh_root)
        assert regenerated==requests,root
        assert same_tensors(torch.load(root/'source_states.pt',weights_only=True),torch.load(fresh_root/'source_states.pt',weights_only=True)),root
    counts={'attempts':len(requests),'complete':0,'errors':0,'incomplete':0,'input_tokens':0,'output_tokens':0}
    for req in requests:
        result=json.loads((root/(req['id']+'.json')).read_text());assert result['id']==req['id']
        if result['status']=='error':counts['errors']+=1;continue
        assert result['status']=='received'
        response=result['response'];assert response['model']==manifest['config']['model']
        assert response_text(response)==result['report']
        counts['complete' if response['status']=='completed' else 'incomplete']+=1
        for key in ('input_tokens','output_tokens'):counts[key]+=response['usage'][key]
    for folder in root.glob('extraction*'):
        if not folder.is_dir():continue
        judge=json.loads((folder/'manifest.json').read_text());assert hashlib.sha256(judge['source'].encode()).hexdigest()==judge['source_sha256']
        judge_requests=json.loads((folder/'requests.json').read_text());assert len(judge_requests)<=judge['max_attempts']
        judge_counts={'attempts':len(judge_requests),'complete':0,'errors':0,'incomplete':0,'input_tokens':0,'output_tokens':0}
        for req in judge_requests:
            original=json.loads((root/(req['id']+'.json')).read_text());assert req['report']==original['report']
            result=json.loads((folder/(req['id']+'.json')).read_text())
            if result['status']=='error':judge_counts['errors']+=1;continue
            assert result['status']=='received', ('unfinished extraction',req['id'])
            assert result['response']['model']==judge['model']
            if 'parsed' in result:assert json.loads(response_text(result['response']))==result['parsed']
            judge_counts['complete' if result['response']['status']=='completed' and 'parsed' in result else 'incomplete']+=1
            for key in ('input_tokens','output_tokens'):judge_counts[key]+=result['response']['usage'][key]
        counts[folder.name]=judge_counts
    return counts


p=argparse.ArgumentParser();p.add_argument('--write-manifest',action='store_true');args=p.parse_args()
subprocess.run([sys.executable,'scripts/verify_bound_mechanism.py'],check=True,stdout=subprocess.DEVNULL)
counts={}
for root in sorted(ROOT.glob('language_*')):
    if (root/'requests.json').exists():counts[root.name]=verify_study(root)
manifest_path=ROOT/'verification.json'
if args.write_manifest:
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=manifest_path)
    manifest_path.write_text(json.dumps({'counts':counts,'files_sha256':{str(p):digest(p) for p in paths}},indent=2)+'\n')
else:
    manifest=json.loads(manifest_path.read_text());assert counts==manifest['counts']
    for path,expected in manifest['files_sha256'].items():assert digest(Path(path))==expected,path
print(json.dumps(counts,indent=2))
