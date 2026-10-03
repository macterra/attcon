#!/usr/bin/env python3
"""Explicit self-access representation: an own-content-by-command table.

The table is the system's prediction, from its own learned attention model rolled
one step forward under each command, of how strongly it will hold each object's
color and shape. Content is gated as in self_coupled_v1. Conditions:
  coupled_table      coupled content; table from the counterfactual rollout
  independent_table  content gated by another episode's weights (independent of
                     this episode's attention); the accurate table is flat
  coupled_table_swapped  coupled content; table replaced by a flat table
                     (the self-model says no command changes own content)
  coupled_no_table   coupled content, no table (self_coupled_v1 replicate)
Stages: --stage prepare | generate | fixtures | extract.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from attcon.predictive_attention import PredictiveAttention, simulate
from specificity_reports import build_states, render, generate, COMMANDS
from self_coupled_reports import gate, blocking
import importlib

ROOT = Path('audits/bound_content')
TABLE_GLOSSARY = ('\nOwn_content_by_command is grouped by command: for each possible command it gives the system\'s prediction, '
                  'from its own attention model, of how strongly it will hold each location\'s color and shape after that command '
                  '(1 = fully, 0 = not at all).')


@torch.no_grad()
def counterfactual_table(pair, config):
    """[episode, command, view, slot] forecast access now after each alternative command."""
    model = PredictiveAttention()
    model.load_state_dict(torch.load(f"audits/predictive_attention/confirmation_v1/seed{pair['attention_seed']}.pt", weights_only=True)['state_dict']); model.eval()
    e = simulate(pair['process_seed'], pair['count'], 12); t = config['step']; x = e.observations
    actual = x[:, t+1]; allocation = actual[:, :8].reshape(-1, 2, 4); quality = actual[:, 8:16].reshape(-1, 2, 4).sum(-1)
    rows = torch.arange(len(x)); tables = []
    for command in range(4):
        a = allocation.clone(); a[rows, e.controlled] = F.one_hot(torch.full((len(x),), command), 4).float()
        alt = torch.cat((a.flatten(1), (a*quality[..., None]).flatten(1), F.one_hot(torch.full((len(x),), command), 4).float()), -1)
        forecast, _ = model(torch.cat((x[:, :t+1], alt[:, None]), 1))
        tables.append(forecast.access[:, t+1, 0])
    return torch.stack(tables, 1)


def conditions(base, table):
    w = base.attention.access[:, 0]; independent = w.roll(1, 0)
    flat = lambda weights: weights[:, None].expand(-1, 4, -1, -1).clone()
    coupled = gate(base, w)
    return {'coupled_table': (coupled, table), 'independent_table': (gate(base, independent), flat(independent)),
            'coupled_table_swapped': (coupled, flat(w)), 'coupled_no_table': (coupled, None)}


def with_table(state, table, i, config):
    source, presented, prompt = render(state, i, 'model', config)
    if table is None: return source, presented, prompt
    for view_index, (s, p) in enumerate(zip(source, presented)):
        rows = {COMMANDS[c]: {COMMANDS[slot]: round(float(table[i, c, view_index, slot]), 5) for slot in range(4)} for c in range(4)}
        s['own_content_by_command'] = rows; p['own_content_by_command'] = json.loads(json.dumps(rows))
    head, _ = prompt.split('\n[{', 1); glossary, rest = head.split('\nUse at most', 1)
    return source, presented, glossary+TABLE_GLOSSARY+'\nUse at most'+rest+'\n'+json.dumps(presented)


def prepare(config, config_path, root):
    root.mkdir(parents=True, exist_ok=True)
    request_path = root/'requests.json'
    if request_path.exists(): return json.loads(request_path.read_text())
    torch.set_num_threads(2); records = []; tensors = {}; checkpoints = {}
    for pair in config['pairs']:
        base, tensors[pair['visual_seed']], hashes = build_states(pair, config); checkpoints.update(hashes)
        table = counterfactual_table(pair, config); conds = conditions(base, table)
        tensors[pair['visual_seed']].update({'table': table, **{f'visual_{k}': v[0].visual for k, v in conds.items()}})
        for i in range(pair['count']):
            for condition in config['conditions']:
                state, t = conds[condition]
                source, presented, prompt = with_table(state, t, i, config)
                records.append({'id': f"{pair['visual_seed']}_{i}_{condition}", 'seed': pair['visual_seed'], 'episode': i,
                                'condition': condition, 'source': source, 'presented': presented, 'input': prompt})
    assert len(records) == config['max_attempts']
    request_path.write_text(json.dumps(records, indent=2)+'\n'); torch.save(tensors, root/'source_states.pt')
    source_paths = ['scripts/self_access_table_reports.py', 'scripts/self_coupled_reports.py', 'scripts/specificity_reports.py',
                    'scripts/bound_reports.py', f"scripts/extract_bound_prose_{config.get('extractor', 'v11')}.py", 'scripts/extract_bound_prose_v11.py', 'scripts/extract_bound_prose_v10.py',
                    'src/attcon/bound_content.py', 'src/attcon/predictive_attention.py', str(config_path)]
    sources = {s: Path(s).read_text() for s in source_paths}
    (root/'source_code.json').write_text(json.dumps(sources, indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps({'config': config, 'checkpoint_sha256': checkpoints,
        'source_sha256': {s: hashlib.sha256(v.encode()).hexdigest() for s, v in sources.items()},
        'requests_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
        'states_sha256': hashlib.sha256((root/'source_states.pt').read_bytes()).hexdigest()}, indent=2)+'\n')
    return records


def extractor_for(config):
    """v1 used extractor v11; later versions name theirs in the config."""
    version = config.get('extractor', 'v11')
    return version, importlib.import_module(f'extract_bound_prose_{version}')


def blocking_v12(results):
    """Self-coupled and v12-specific fixtures block; v10 claim fixtures do not."""
    return [r['id'] for r in results if not r['passed'] and r['kind'] != 'v10_claim']


async def extract(root, fixture_root, limit, config):
    version, extractor = extractor_for(config)
    results = json.loads((fixture_root/'assessment.json').read_text())
    failed = blocking(results) if version == 'v11' else blocking_v12(results)
    if failed: raise SystemExit(f'blocking fixtures failed: {failed}')
    requests = []
    for req in json.loads((root/'requests.json').read_text()):
        response = json.loads((root/(req['id']+'.json')).read_text())
        if response.get('response', {}).get('status') == 'completed' and response.get('report'):
            requests.append({'id': req['id'], 'report': response['report']})
    await extractor.run_requests(requests, root/f'extraction_{version}', limit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/bound_content/self_access_table_v1.json')
    p.add_argument('--stage', choices=['prepare', 'generate', 'fixtures', 'extract'], required=True)
    args = p.parse_args()
    path = Path(args.config); config = json.loads(path.read_text()); root = ROOT/config['name']
    version, extractor = extractor_for(config)
    fixture_root = ROOT/(config['name']+f'_extractor_fixtures_{version}')
    if args.stage == 'prepare': print(len(prepare(config, path, root)), 'requests')
    elif args.stage == 'generate': asyncio.run(generate(config, prepare(config, path, root), root))
    elif args.stage == 'fixtures': asyncio.run(extractor.fixtures(fixture_root))
    else: asyncio.run(extract(root, fixture_root, config['max_attempts'], config))


if __name__ == '__main__': main()
