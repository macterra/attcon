#!/usr/bin/env python3
"""Retained development experiment; never a confirmatory or qualia verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from attcon.predictive_attention import PredictiveAttention, Forecast, simulate, prediction_loss, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=811)
    p.add_argument('--updates', type=int, default=1200)
    p.add_argument('--output', default='audits/predictive_attention/pilot_v1')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f'seed{args.seed}.json'
    if result_path.exists():
        raise SystemExit('refusing to overwrite retained pilot')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model = PredictiveAttention()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    history = []
    for update in range(args.updates):
        episode = simulate(args.seed * 100000 + update, 128)
        forecast, _ = model(episode.observations)
        loss = prediction_loss(forecast, episode)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (update + 1) % 200 == 0:
            entry = {'update': update + 1, 'loss': loss.item()}
            history.append(entry)
            print(json.dumps(entry), flush=True)
    model.eval()
    evaluation_seed = args.seed + 900000000
    e = simulate(evaluation_seed, 512, 16)
    metrics = evaluate(model, e)
    with torch.no_grad():
        forecasts, _ = model(e.observations)
        f = Forecast(forecasts.allocation[:, 7], forecasts.access[:, 7], forecasts.effects[:, 7])
        query = torch.arange(512) % 4
        commands = f.command(e.controlled, query)
        changed = f.intervene('effects', f.effects.roll(1, dims=1))
        changed_commands = changed.command(e.controlled, query)
        restored = changed.intervene('effects', f.effects)
        metrics.update({
            'policy_hits_queried_slot': (commands == query).float().mean().item(),
            'effect_intervention_command_change': (changed_commands != commands).float().mean().item(),
            'effect_intervention_shift_following': (changed_commands == (commands + 1) % 4).float().mean().item(),
            'restoration_exact': torch.equal(restored.command(e.controlled, query), commands),
            'unintervened_access_exact': torch.equal(changed.access, f.access),
        })
    checkpoint = out / f'seed{args.seed}.pt'
    torch.save({'state_dict': model.state_dict(), 'seed': args.seed, 'updates': args.updates}, checkpoint)
    paths = ['src/attcon/predictive_attention.py', 'scripts/predictive_attention_pilot.py']
    result = {'status': 'development pilot; not confirmation', 'seed': args.seed,
              'evaluation_seed': evaluation_seed, 'evaluation_count': 512, 'evaluation_steps': 16,
              'burn_in': 4, 'history': history, 'metrics': metrics,
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'source_sha256': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in paths}}
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
