#!/usr/bin/env python3
"""Frozen internal model confirmation; no report-character inference."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from attcon.predictive_attention import PredictiveAttention, Forecast, simulate, prediction_loss, evaluate

ROOT = Path('audits/predictive_attention/confirmation_v1')
SEEDS = (901, 911, 921)


@torch.no_grad()
def assess(model, seed):
    e = simulate(seed + 950000000, 1024, 16)
    metrics = evaluate(model, e)
    predictions, _ = model(e.observations)
    f = Forecast(predictions.allocation[:, 7], predictions.access[:, 7], predictions.effects[:, 7])
    query = torch.arange(1024) % 4
    command = f.command(e.controlled, query)
    altered = f.intervene('effects', f.effects.roll(1, 1))
    restored = altered.intervene('effects', f.effects)
    metrics.update({
        'policy_query_hit': (command == query).double().mean().item(),
        'effect_rotation_following': (altered.command(e.controlled, query) == (command + 1) % 4).double().mean().item(),
        'restoration_exact': torch.equal(restored.command(e.controlled, query), command),
        'preservation_exact': torch.equal(altered.access, f.access) and torch.equal(altered.allocation, f.allocation),
        'access_only_command_invariant': torch.equal(f.intervene('access', f.access.flip(-1)).command(e.controlled, query), command),
        'allocation_only_command_invariant': torch.equal(f.intervene('allocation', f.allocation.roll(1, -1)).command(e.controlled, query), command),
    })
    gates = {name: metrics[name] >= .99 for name in ('allocation_accuracy', 'effect_accuracy', 'controlled_channel_accuracy', 'policy_query_hit', 'effect_rotation_following')}
    gates['access_mae'] = metrics['access_mae'] <= .04
    gates['reconstruction_calibration'] = metrics['reconstruction_brier'] - metrics['oracle_reconstruction_brier'] <= .005
    gates.update({k: metrics[k] for k in ('restoration_exact', 'preservation_exact', 'access_only_command_invariant', 'allocation_only_command_invariant')})
    return {'metrics': metrics, 'gates': gates, 'all_gates_pass': all(gates.values())}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, choices=SEEDS, required=True); p.add_argument('--replay', action='store_true'); args = p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    ROOT.mkdir(parents=True, exist_ok=True)
    result_path = ROOT / f'seed{args.seed}.json'; checkpoint = ROOT / f'seed{args.seed}.pt'
    model = PredictiveAttention()
    if args.replay:
        recorded = json.loads(result_path.read_text())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == recorded['checkpoint_sha256']
        for path, digest in recorded['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        model.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict']); model.eval()
        assert assess(model, args.seed) == recorded['assessment']
        print(f'seed {args.seed}: exact replay verified', flush=True)
        return
    if result_path.exists() or checkpoint.exists():
        raise SystemExit('refusing to overwrite confirmation')
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for update in range(1200):
        e = simulate(args.seed * 100000 + update, 128)
        f, _ = model(e.observations); loss = prediction_loss(f, e)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if (update + 1) % 400 == 0:
            print(args.seed, update + 1, float(loss.detach()), flush=True)
    model.eval()
    assessment = assess(model, args.seed)
    torch.save({'state_dict': model.state_dict(), 'seed': args.seed, 'updates': 1200}, checkpoint)
    sources = ['src/attcon/predictive_attention.py', 'scripts/confirm_predictive_attention.py', 'docs/PREDICTIVE_CONFIRMATION_PROTOCOL.md']
    record = {'seed': args.seed, 'evaluation_seed': args.seed + 950000000, 'assessment': assessment,
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'source_sha256': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}}
    result_path.write_text(json.dumps(record, indent=2) + '\n'); print(json.dumps(record['assessment'], indent=2), flush=True)


if __name__ == '__main__':
    main()
