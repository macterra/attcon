"""Matched-input interventions in existing command-predictive hidden states."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from attcon.predictive_attention import PredictiveAttention

ROOT = Path('audits/hidden_expectation_v1')
COUNT, PREFIX, SUFFIX = 512, 8, 4


def paired_streams(seed, count=COUNT):
    g = torch.Generator().manual_seed(seed)
    commands = torch.randint(4, (count, PREFIX + SUFFIX + 1), generator=g)
    replay = torch.randint(4, (count,), generator=g)
    direction = torch.randint(2, (count,), generator=g) * 2 - 1
    quality = .1 + .9 * torch.rand(count, PREFIX + SUFFIX + 1, 2, 1, generator=g)
    streams, targets = [[], []], [[], []]
    rows = torch.arange(count)
    for t in range(PREFIX + SUFFIX + 1):
        replay = (replay + direction) % 4
        for owner in (0, 1):
            table = F.one_hot(replay[:, None, None].expand(-1, 4, 2), 4).float()
            for command in range(4):
                table[:, command, owner] = F.one_hot(torch.full((count,), command), 4).float()
            allocation = table[rows, commands[:, t]]
            streams[owner].append(torch.cat((allocation.flatten(1),
                (allocation * quality[:, t]).flatten(1), F.one_hot(commands[:, t], 4).float()), -1))
            targets[owner].append(table)
    return [torch.stack(x, 1) for x in streams], [torch.stack(x, 1) for x in targets]


def head_projector(weight):
    _, singular, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    keep = singular > singular.max() * 1e-10
    basis = vh[keep]
    return (basis.T @ basis).to(weight.dtype), int(keep.sum())


def decode(model, h):
    z = h[0]
    return (model.effect_head(z).reshape(-1, 4, 2, 4).softmax(-1),
            model.allocation_head(z).reshape(-1, 2, 4).softmax(-1),
            model.access_head(z).sigmoid())


@torch.no_grad()
def assess(model, seed):
    streams, tables = paired_streams(seed + 290910000)
    histories = [model(x[:, :PREFIX])[1] for x in streams]
    projector, rank = head_projector(model.effect_head.weight)
    records = []
    for owner in (0, 1):
        base, donor = histories[owner], histories[1-owner]
        delta = donor - base
        row_delta = delta @ projector
        null_delta = delta - row_delta
        g = torch.Generator().manual_seed(seed + owner + 790000)
        random_delta = torch.randn(delta.shape, generator=g)
        random_delta *= row_delta.norm(dim=-1, keepdim=True) / random_delta.norm(dim=-1, keepdim=True)
        variants = {'matched': base, 'mismatched': donor, 'effect_row': base + row_delta,
                    'effect_null': base + null_delta, 'random_matched_norm': base + random_delta}
        logits = lambda h: model.effect_head(h[0])
        null_error = float((logits(variants['effect_null']) - logits(base)).abs().max())
        row_error = float((logits(variants['effect_row']) - logits(donor)).abs().max())
        assert null_error < 1e-4 and row_error < 1e-4
        baseline = decode(model, base)
        for name, initial in variants.items():
            pred = decode(model, initial)
            entry = {'owner': owner, 'treatment': name,
                'null_logit_max_error': null_error, 'row_donor_logit_max_error': row_error,
                'perturbation_norm': float((initial-base).norm(dim=-1).mean()),
                'pre_effect_change_mae': float((pred[0]-baseline[0]).abs().mean()),
                'pre_allocation_change_mae': float((pred[1]-baseline[1]).abs().mean()),
                'pre_access_change_mae': float((pred[2]-baseline[2]).abs().mean()), 'steps': []}
            h = initial.clone()
            for offset in range(SUFFIX):
                t = PREFIX + offset
                prev = h
                f, h = model(streams[owner][:, t:t+1], h)
                truth = tables[owner][:, t+1]
                entry['steps'].append({'observations': offset+1,
                    'next_effect_accuracy': float((f.effects[:, 0].argmax(-1) == truth.argmax(-1)).double().mean()),
                    'allocation_accuracy': float((f.allocation[:, 0].argmax(-1) == streams[owner][:, t, :8].reshape(-1, 2, 4).argmax(-1)).double().mean()),
                    'hidden_update_norm': float((h-prev).norm(dim=-1).mean())})
            records.append(entry)
    return {'seed': seed, 'head_rank': rank, 'records': records,
            'observation_sha256': [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in streams]}


def main():
    if (ROOT / 'results.json').exists():
        raise SystemExit('refusing to overwrite retained diagnostic')
    torch.set_num_threads(2)
    results, checkpoints = [], {}
    for seed in (901, 911, 921):
        path = Path(f'audits/predictive_attention/confirmation_v1/seed{seed}.pt')
        checkpoints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        model = PredictiveAttention()
        model.load_state_dict(torch.load(path, weights_only=True)['state_dict'])
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        results.append(assess(model, seed))
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    paths = ['scripts/diagnose_hidden_expectation.py', 'src/attcon/predictive_attention.py',
             'docs/HIDDEN_EXPECTATION_PROTOCOL.md']
    result = {'count_per_seed': COUNT, 'prefix': PREFIX, 'suffix': SUFFIX,
              'torch_version': torch.__version__, 'checkpoint_sha256': checkpoints,
              'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
              'parameter_preservation_exact': True, 'results': results}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Wrote', ROOT / 'results.json')


if __name__ == '__main__':
    main()
