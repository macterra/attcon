"""Replay every saved three-way forecast, matched target and engineering gate."""
import argparse
import gzip
import io
import json
from pathlib import Path
import torch
from attcon.predictive_attention import PredictiveAttention
from run_functional_controls import ROOT, SEEDS, assess_model, check_sources, digest


def compare(expected,actual,path='trace'):
    if isinstance(expected,torch.Tensor):
        assert isinstance(actual,torch.Tensor) and torch.equal(expected,actual),path
    elif isinstance(expected,dict):
        assert expected.keys()==actual.keys(),path
        for key in expected:compare(expected[key],actual[key],path+'/'+str(key))
    else:
        assert expected==actual,path


def verify_prepared():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    sources=json.loads((ROOT/'source_code.json').read_text())
    check_sources(manifest)
    import hashlib
    for p,h in manifest['source_sha256'].items():
        assert hashlib.sha256(sources[p].encode()).hexdigest()==h,p
    print('Frozen three-way source/protocol hashes verified',flush=True)
    return manifest


def verify(seed,manifest):
    record=json.loads((ROOT/f'seed{seed}.json').read_text())
    assert record['status']=='completed' and record['seed']==seed
    assert record['source_sha256']==manifest['source_sha256']
    checkpoint=ROOT/f'seed{seed}.pt';trace_path=ROOT/f'seed{seed}_trace.pt.gz'
    assert digest(checkpoint)==record['checkpoint_sha256']
    assert digest(trace_path)==record['trace_sha256']
    weights=torch.load(checkpoint,weights_only=True)
    assert weights['seed']==seed and weights['updates']==manifest['updates']
    model=PredictiveAttention();model.load_state_dict(weights['state_dict']);model.eval()
    assessment,expected=assess_model(model,seed)
    assert json.loads(json.dumps(assessment))==record['assessment']
    with gzip.open(trace_path,'rb') as stream:
        actual=torch.load(io.BytesIO(stream.read()),weights_only=True)
    compare(expected,actual)
    print(f'{seed}: every static/revision tensor, common target, metric and gate exactly replayed',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared-only',action='store_true')
    p.add_argument('--seed',type=int,choices=SEEDS);args=p.parse_args()
    torch.set_num_threads(2);manifest=verify_prepared()
    if args.prepared_only:return
    for seed in (args.seed,) if args.seed is not None else SEEDS:verify(seed,manifest)


if __name__=='__main__':main()
