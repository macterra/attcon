"""Probe frozen recurrent feedback without changing prior studies."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from attcon.predictive_attention import PredictiveAttention, Forecast

ROOT = Path('audits/attention_consistency_v1')
COUNT, STEPS, SWITCH = 512, 24, 8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(seed, condition):
    g = torch.Generator().manual_seed(seed)
    owner = torch.randint(2, (COUNT,), generator=g)
    direction = torch.randint(2, (COUNT,), generator=g) * 2 - 1
    replay = torch.randint(4, (COUNT,), generator=g)
    commands = torch.randint(4, (COUNT, STEPS), generator=g)
    quality = .1 + .9 * torch.rand(COUNT, STEPS, 2, 1, generator=g)
    rows = torch.arange(COUNT)
    observations, tables, owners = [], [], []
    # Table[t] describes the actual result of every command at step t.
    for t in range(STEPS + 1):
        replay = (replay + direction) % 4
        own = 1 - owner if condition == 'owner_swap' and t >= SWITCH else owner
        offset = 2 if condition == 'command_offset' and t >= SWITCH else 0
        table = F.one_hot(replay[:, None, None].expand(-1, 4, 2), 4).float()
        for c in range(4):
            table[rows, c, own] = F.one_hot(torch.full((COUNT,), (c + offset) % 4), 4).float()
        tables.append(table)
        owners.append(own)
        if t < STEPS:
            allocation = table[rows, commands[:, t]]
            glimpses = allocation * quality[:, t]
            observations.append(torch.cat((allocation.flatten(1), glimpses.flatten(1),
                                           F.one_hot(commands[:, t], 4).float()), -1))
    return torch.stack(observations, 1), tables, owners, commands


@torch.no_grad()
def run(model, observations, tables, owners, commands, treatment):
    h, previous = None, None
    rows = torch.arange(COUNT)
    output = []
    for t in range(STEPS):
        if t == SWITCH:
            if treatment == 'reset':
                h = torch.zeros_like(h)
            elif treatment == 'shuffle':
                h = h.roll(1, 1)
        pred, h = model(observations[:, t:t+1], h)
        current = Forecast(pred.allocation[:, 0], pred.access[:, 0], pred.effects[:, 0])
        target = tables[t+1]
        query = rows % 4
        chosen = current.command(owners[t+1], query)
        actual = tables[t][rows, commands[:, t]]
        entry = {
            'step': t,
            'next_effect_accuracy': float((current.effects.argmax(-1) == target.argmax(-1)).double().mean()),
            'controlled_query_success': float((target[rows, chosen, owners[t+1]].argmax(-1) == query).double().mean()),
            'executed_prediction_mse_before_observation': None if previous is None else
                float((previous.effects[rows, commands[:, t]] - actual).square().double().mean()),
            'allocation_mse_after_observation': float((current.allocation - actual).square().double().mean()),
        }
        output.append(entry)
        previous = current
    return output


def main():
    if (ROOT / 'results.json').exists():
        raise SystemExit('refusing to overwrite retained diagnostic')
    torch.set_num_threads(2)
    records = []
    checkpoints = {}
    for seed in (901, 911, 921):
        path = Path(f'audits/predictive_attention/confirmation_v1/seed{seed}.pt')
        checkpoints[str(path)] = digest(path)
        model = PredictiveAttention()
        model.load_state_dict(torch.load(path, weights_only=True)['state_dict'])
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for condition in ('unchanged', 'owner_swap', 'command_offset'):
            x, tables, owners, commands = process(seed + 290900000, condition)
            outcomes = {kind: run(model, x, tables, owners, commands, kind)
                        for kind in ('intact', 'reset', 'shuffle')}
            assert all(outcomes[kind][:SWITCH] == outcomes['intact'][:SWITCH]
                       for kind in ('reset', 'shuffle'))
            with torch.no_grad():
                forecast, h = model(x[:, :SWITCH])
                a, _ = model(x[:, SWITCH:SWITCH+1], h.clone())
                forecast.effects.copy_(forecast.effects.roll(1, -1))
                b, _ = model(x[:, SWITCH:SWITCH+1], h.clone())
                invariant = all(torch.equal(getattr(a, k), getattr(b, k))
                                for k in ('allocation', 'access', 'effects'))
                assert invariant
            records.append({'seed': seed, 'condition': condition, 'treatments': outcomes,
                            'forecast_corruption_next_prediction_invariant': invariant,
                            'observations_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest()})
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    sources = ['scripts/diagnose_attention_consistency.py', 'src/attcon/predictive_attention.py',
               'src/attcon/predictive_closed_loop.py', 'src/attcon/bound_content.py',
               'docs/CONSISTENCY_DIAGNOSTIC_PROTOCOL.md']
    result = {'count_per_seed': COUNT, 'steps': STEPS, 'switch_before_step': SWITCH,
              'torch_version': torch.__version__, 'checkpoint_sha256': checkpoints,
              'source_sha256': {p: digest(p) for p in sources}, 'records': records,
              'parameter_preservation_exact': True}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Wrote', ROOT / 'results.json')


if __name__ == '__main__':
    main()
