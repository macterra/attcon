#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import torch
from attcon.predictive_attention import PredictiveAttention
from attcon.predictive_closed_loop import rollout

torch.set_num_threads(2)
root = Path('audits/predictive_attention/closed_loop_confirmation'); root.mkdir(parents=True, exist_ok=True)
for seed in (901, 911, 921):
    path = root / f'seed{seed}.json'
    if path.exists():
        raise SystemExit('refusing to overwrite confirmation')
    checkpoint = Path(f'audits/predictive_attention/confirmation_v1/seed{seed}.pt')
    model = PredictiveAttention(); model.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict']); model.eval()
    metrics, traces = {}, {}
    for condition in ('none', 'rotate_effects', 'restore', 'shuffle_effects'):
        metrics[condition], traces[condition] = rollout(model, 980000000 + seed, count=1024, intervention=condition)
    gates = {'controlled_selection': metrics['none']['controlled_query_selected'] >= .99,
             'rotation': metrics['rotate_effects']['controlled_query_selected'] <= .01,
             'restoration': all(torch.equal(traces['none'][k], traces['restore'][k]) for k in traces['none'])}
    archive = root / f'seed{seed}_traces.pt'; torch.save(traces, archive)
    sources = ['src/attcon/predictive_closed_loop.py', 'scripts/confirm_predictive_closed_loop.py', 'docs/PREDICTIVE_CLOSED_LOOP_CONFIRMATION.md']
    record = {'seed': seed, 'metrics': metrics, 'gates': gates, 'all_gates_pass': all(gates.values()),
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'trace_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'source_sha256': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}}
    path.write_text(json.dumps(record, indent=2) + '\n'); print(seed, gates, flush=True)
