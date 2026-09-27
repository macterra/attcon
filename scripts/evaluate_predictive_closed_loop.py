#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import torch
from attcon.predictive_attention import PredictiveAttention
from attcon.predictive_closed_loop import rollout

p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, default=811); args = p.parse_args()
torch.set_num_threads(2)
root = Path('audits/predictive_attention/closed_loop_v1'); root.mkdir(parents=True, exist_ok=True)
path = root / f'seed{args.seed}.json'
if path.exists():
    raise SystemExit('refusing to overwrite closed-loop assessment')
checkpoint = Path(f'audits/predictive_attention/pilot_v1/seed{args.seed}.pt')
model = PredictiveAttention(); model.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict']); model.eval()
metrics, traces = {}, {}
for condition in ('none', 'rotate_effects', 'restore', 'shuffle_effects'):
    metrics[condition], traces[condition] = rollout(model, 960000000 + args.seed, intervention=condition)
metrics['restoration_entire_trace_exact'] = all(torch.equal(traces['none'][k], traces['restore'][k]) for k in traces['none'])
archive = root / f'seed{args.seed}_traces.pt'; torch.save(traces, archive)
record = {'status': 'development closed-loop extension', 'seed': args.seed, 'metrics': metrics,
          'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'trace_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'source_sha256': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in
                           ['src/attcon/predictive_closed_loop.py', 'scripts/evaluate_predictive_closed_loop.py']}}
path.write_text(json.dumps(record, indent=2) + '\n'); print(json.dumps(metrics, indent=2))
