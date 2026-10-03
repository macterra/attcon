#!/usr/bin/env python3
"""Self-coupled access: content certainty gated by the attention model's own forecast.

Stages: --stage prepare | generate | fixtures | extract. In `coupled` conditions the
system's color/shape distributions are mixed toward uniform by its forecast access
now; `decoupled_matched` applies the same weights permuted across locations within
each view, so uncertainty is matched but no longer tracks modeled access. The
attention-model values are identical in both. Scoring uses each condition's own
canonical v5-format source record.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import torch
from specificity_reports import build_states, render, generate
import extract_bound_prose_v11 as extractor

ROOT = Path('audits/bound_content')


def gate(state, weights):
    """Own content = w * encoder distribution + (1 - w) * uniform, per object."""
    w = weights.clamp(0, 1)[..., None]  # [batch, view, slot]; binding is identity
    return state.replace_visual(w * state.visual + (1 - w) * .25)


def states(base):
    access_now = base.attention.access[:, 0]
    intervened = base.replace_attention(base.attention.intervene('access', base.attention.access.flip(-1)))
    coupled = gate(base, access_now)
    decoupled = gate(base, access_now.roll(1, -1))
    return {'coupled': coupled, 'decoupled_matched': decoupled,
            'coupled_access_intervention': gate(intervened, intervened.attention.access[:, 0]),
            'coupled_opaque': coupled, 'decoupled_opaque': decoupled}


def label(condition):
    """Rendering variant: opaque replicates use the specificity_v1 neutral labels."""
    return 'opaque' if condition.endswith('_opaque') else 'model'


def prepare(config, config_path, root):
    root.mkdir(parents=True, exist_ok=True)
    request_path = root/'requests.json'
    if request_path.exists(): return json.loads(request_path.read_text())
    torch.set_num_threads(2); records = []; tensors = {}; checkpoints = {}
    for pair in config['pairs']:
        base, tensors[pair['visual_seed']], hashes = build_states(pair, config); checkpoints.update(hashes)
        conditions = states(base)
        tensors[pair['visual_seed']].update({f'visual_{k}': v.visual for k, v in conditions.items()})
        for i in range(pair['count']):
            for condition in config['conditions']:
                source, presented, prompt = render(conditions[condition], i, label(condition), config)
                records.append({'id': f"{pair['visual_seed']}_{i}_{condition}", 'seed': pair['visual_seed'], 'episode': i,
                                'condition': condition, 'source': source, 'presented': presented, 'input': prompt})
    assert len(records) == config['max_attempts']
    request_path.write_text(json.dumps(records, indent=2)+'\n'); torch.save(tensors, root/'source_states.pt')
    source_paths = ['scripts/self_coupled_reports.py', 'scripts/specificity_reports.py', 'scripts/bound_reports.py',
                    'scripts/extract_bound_prose_v11.py', 'scripts/extract_bound_prose_v10.py', 'src/attcon/bound_content.py', str(config_path)]
    sources = {s: Path(s).read_text() for s in source_paths}
    (root/'source_code.json').write_text(json.dumps(sources, indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps({'config': config, 'checkpoint_sha256': checkpoints,
        'source_sha256': {s: hashlib.sha256(v.encode()).hexdigest() for s, v in sources.items()},
        'requests_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
        'states_sha256': hashlib.sha256((root/'source_states.pt').read_bytes()).hexdigest()}, indent=2)+'\n')
    return records


def blocking(results):
    """Frozen rule: any self-coupled fixture failure blocks; v10 claim fixtures do not."""
    return [r['id'] for r in results if not r['passed'] and r['kind'] == 'self_coupled']


async def extract(root, fixture_root, limit):
    results = json.loads((fixture_root/'assessment.json').read_text())
    if blocking(results): raise SystemExit(f'self-coupled fixtures failed: {blocking(results)}')
    requests = []
    for req in json.loads((root/'requests.json').read_text()):
        response = json.loads((root/(req['id']+'.json')).read_text())
        if response.get('response', {}).get('status') == 'completed' and response.get('report'):
            requests.append({'id': req['id'], 'report': response['report']})
    await extractor.run_requests(requests, root/'extraction_v11', limit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/bound_content/self_coupled_v1.json')
    p.add_argument('--stage', choices=['prepare', 'generate', 'fixtures', 'extract'], required=True)
    args = p.parse_args()
    path = Path(args.config); config = json.loads(path.read_text()); root = ROOT/config['name']
    fixture_root = ROOT/(config['name']+'_extractor_fixtures_v11')
    if args.stage == 'prepare': print(len(prepare(config, path, root)), 'requests')
    elif args.stage == 'generate': asyncio.run(generate(config, prepare(config, path, root), root))
    elif args.stage == 'fixtures': asyncio.run(extractor.fixtures(fixture_root))
    else: asyncio.run(extract(root, fixture_root, config['max_attempts']))


if __name__ == '__main__': main()
