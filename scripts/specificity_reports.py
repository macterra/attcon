#!/usr/bin/env python3
"""Report-structure specificity: same bound states, varied labelling and framing.

Stages: --stage prepare | generate | fixtures | extract. Rendering of `model` and
`visual_only` reproduces bound_reports.py exactly; `analyst` changes only the
speaker framing; `external` and `opaque` present identical values under relabelled
fields and glossaries. Scoring always uses the canonical v5-format `source` record.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import torch
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
from attcon.bound_content import VisualEncoder, bind, scenes, full_record
from attcon.predictive_attention import PredictiveAttention, Forecast, simulate
from bound_reports import GLOSSARY
import extract_bound_prose_v10 as extractor

ROOT = Path('audits/bound_content')
COMMANDS = ('upper', 'right', 'lower', 'left')


def render_v5(state, i, condition, config):
    """Exact copy of the bound_reports.py v5 rendering path for one record."""
    record = full_record(state, i)
    if condition in ('visual_only', 'attention_only'):
        for view in record:
            for obj in view['objects']:
                keys = ('selection_probability', 'recoverability_now_then_one_then_two_steps', 'next_selection_by_command') if condition == 'visual_only' else ('color_distribution', 'shape_distribution')
                for key in keys: obj[key] = None
    if config.get('command_major', False):
        for view in record:
            entries = view['objects']
            view['command_predictions'] = {command: {obj['location']: obj['next_selection_by_command'][command] for obj in entries}
                for command in COMMANDS} if entries[0]['next_selection_by_command'] is not None else None
            for obj in entries: del obj['next_selection_by_command']
    glossary = GLOSSARY
    if config.get('command_major', False):
        glossary = glossary.replace('Next-selection-by-command gives the predicted selection of that location for each\npossible redirection command.', 'Command_predictions is grouped by command: each command maps every location to its\npredicted next-selection probability. Compare these rows to assess controllability.')
    if config.get('derived_indexes', False):
        for view in record:
            entries = view['objects']
            def winner(values):
                if any(v is None for v in values.values()): return None
                m = max(values.values()); best = [k for k, v in values.items() if v == m]
                return best[0] if len(best) == 1 else None
            view['derived_indexes'] = {
                'most_selected_location': winner({o['location']: o['selection_probability'] for o in entries}),
                'most_recoverable_location_now': winner({o['location']: None if o['recoverability_now_then_one_then_two_steps'] is None else o['recoverability_now_then_one_then_two_steps'][0] for o in entries}),
                'next_selected_location_by_command': None if view['command_predictions'] is None else {c: winner(row) for c, row in view['command_predictions'].items()}}
        glossary += '\nDerived indexes are exact argmax reductions of the full distributions. Current selection, identity certainty, and recoverability are distinct; next selection alone does not specify future recoverability.'
    if config.get('temporal_indexes', False):
        for view in record:
            view['derived_indexes']['recoverability_direction_by_location'] = {o['location']: None if o['recoverability_now_then_one_then_two_steps'] is None else ('decreasing' if o['recoverability_now_then_one_then_two_steps'][0]-o['recoverability_now_then_one_then_two_steps'][-1] > 1e-5 else 'increasing' if o['recoverability_now_then_one_then_two_steps'][-1]-o['recoverability_now_then_one_then_two_steps'][0] > 1e-5 else 'unchanged') for o in view['objects']}
        glossary += "\nTemporal indexes compare each object now with its two-step forecast; do not generalize one object's direction to all objects. A color or shape is identified only when its maximum probability is at least 0.6. Lower-probability alternatives may be described as possibilities, not asserted identities."
    if config.get('command_diversity_indexes', False):
        for view in record:
            targets = view['derived_indexes']['next_selected_location_by_command']
            view['derived_indexes']['distinct_next_locations_across_commands'] = None if targets is None else sorted(set(targets.values()), key=COMMANDS.index)
        glossary += '\nCompare destinations across different commands to assess redirection: a stable mapping from each command to its different named location permits redirection. If all commands lead to one location, changing the command cannot redirect selection to the other locations. Distinct-next-location indexes are exact reductions of the full command table.'
    return record, glossary


def relabel(record, variant):
    """Rename fields only; every value and nesting is preserved."""
    out = []
    for view in record:
        renamed = {}
        for key, value in view.items():
            if key == 'objects':
                value = [{variant['object_keys'].get(k, k): v for k, v in obj.items()} for obj in value]
            elif key == 'derived_indexes':
                value = {variant['index_keys'].get(k, k): v for k, v in value.items()}
            renamed[variant['view_keys'].get(key, key)] = copy.deepcopy(value)
        out.append(renamed)
    return out


def render(state, i, condition, config):
    """Return (canonical source record, presented record, full prompt)."""
    base = 'visual_only' if condition == 'visual_only' else 'model'
    source, glossary = render_v5(state, i, base, config)
    variant = config.get('variants', {}).get(condition, {})
    presented = relabel(source, variant) if 'object_keys' in variant else copy.deepcopy(source)
    glossary = variant.get('glossary', glossary)
    instruction = variant.get('prose_instruction', config['prose_instruction'])
    question = variant.get('question', config['question'])
    prompt = glossary+f"\nUse at most {config['word_limit']} words.\n"+instruction+'\n'+question+'\n'+json.dumps(presented)
    return source, presented, prompt


def build_states(pair, config):
    visual_path = Path(f"audits/bound_content/{pair['visual_folder']}/seed{pair['visual_seed']}.pt")
    attention_path = Path(f"audits/predictive_attention/confirmation_v1/seed{pair['attention_seed']}.pt")
    encoder = VisualEncoder(); encoder.load_state_dict(torch.load(visual_path, weights_only=True)['state_dict']); encoder.eval()
    model = PredictiveAttention(); model.load_state_dict(torch.load(attention_path, weights_only=True)['state_dict']); model.eval()
    e = simulate(pair['process_seed'], pair['count'], 12); patches, colors, shapes = scenes(pair['scene_seed'], pair['count'])
    with torch.no_grad(): pred, _ = model(e.observations)
    step = config['step']; f = Forecast(pred.allocation[:, step], pred.access[:, step], pred.effects[:, step])
    base = bind(encoder, patches, f, e.allocation[:, :step+1])
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (visual_path, attention_path)}
    tensors = {'patches': patches, 'colors': colors, 'shapes': shapes, 'observations': e.observations, 'allocations': e.allocation,
               'visual': base.visual, 'binding': base.binding, 'remembered': base.remembered,
               'allocation': f.allocation, 'access': f.access, 'effects': f.effects}
    return base, tensors, hashes


def prepare(config, config_path, root):
    root.mkdir(parents=True, exist_ok=True)
    request_path = root/'requests.json'
    if request_path.exists(): return json.loads(request_path.read_text())
    torch.set_num_threads(2); records = []; tensors = {}; checkpoints = {}
    for pair in config['pairs']:
        base, tensors[pair['visual_seed']], hashes = build_states(pair, config); checkpoints.update(hashes)
        for i in range(pair['count']):
            for condition in config['conditions']:
                source, presented, prompt = render(base, i, condition, config)
                records.append({'id': f"{pair['visual_seed']}_{i}_{condition}", 'seed': pair['visual_seed'], 'episode': i,
                                'condition': condition, 'source': source, 'presented': presented, 'input': prompt})
    assert len(records) == config['max_attempts']
    request_path.write_text(json.dumps(records, indent=2)+'\n'); torch.save(tensors, root/'source_states.pt')
    source_paths = ['scripts/specificity_reports.py', 'scripts/bound_reports.py', 'scripts/extract_bound_prose_v10.py',
                    'src/attcon/bound_content.py', str(config_path)]
    sources = {s: Path(s).read_text() for s in source_paths}
    (root/'source_code.json').write_text(json.dumps(sources, indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps({'config': config, 'checkpoint_sha256': checkpoints,
        'source_sha256': {s: hashlib.sha256(v.encode()).hexdigest() for s, v in sources.items()},
        'requests_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
        'states_sha256': hashlib.sha256((root/'source_states.pt').read_bytes()).hexdigest()}, indent=2)+'\n')
    return records


async def generate(config, requests, root):
    load_dotenv(); client = AsyncOpenAI(max_retries=0, timeout=120.); semaphore = asyncio.Semaphore(8)
    async def run(req):
        path = root/(req['id']+'.json')
        if path.exists(): return
        async with semaphore:
            path.write_text(json.dumps({'id': req['id'], 'status': 'attempt_reserved'})+'\n')
            try:
                response = await client.responses.create(model=config['model'], input=req['input'], max_output_tokens=config['max_output_tokens'],
                    reasoning={'effort': config['reasoning']}, text={'verbosity': 'low'})
                result = {'id': req['id'], 'status': 'received', 'response': response.model_dump(mode='json'), 'report': response.output_text}
            except Exception as exc:
                result = {'id': req['id'], 'status': 'error', 'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
            path.write_text(json.dumps(result, indent=2)+'\n'); print(req['id'], result['status'], flush=True)
    await asyncio.gather(*(run(r) for r in requests)); await client.close()


async def fixtures(root):
    """Fresh run of all 31 v10 fixtures, assessed exactly as in extract_bound_prose_v10."""
    requests = [{'id': key, 'report': report} for key, report, _ in extractor.FIXTURES]
    await extractor.run_requests(requests, root, 31)
    results = []
    for key, report, expected in extractor.FIXTURES:
        result = json.loads((root/(key+'.json')).read_text()); claims = result.get('parsed', {}).get('claims', [])
        n = len(extractor.sentences(report))
        good = any(all(c.get(k) == v for k, v in expected.items()) and bool(c['evidence_ids']) and all(0 <= idx < n for idx in c['evidence_ids']) for c in claims)
        negative = key in ('unknown', 'uncertain_object', 'unknown_control')
        if negative:
            fields = ('color', 'shape', 'focal', 'most_recoverable', 'access_trend', 'under_own_control', 'command', 'next_location')
            good = 'parsed' in result and all(all(c.get(k) is None for k in fields) and bool(c['evidence_ids']) and all(0 <= i < n for i in c['evidence_ids']) for c in claims)
        results.append({'id': key, 'passed': good, 'expected': expected, 'negative_absence_allowed': negative})
    (root/'assessment.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f"fixtures passed {sum(r['passed'] for r in results)}/{len(results)}", flush=True)


# Amendment 1 (SPECIFICITY_PROTOCOL.md): the retained `still_low` access-trend
# failure is accepted; any other fixture failure still blocks extraction.
ACCEPTED_FIXTURE_FAILURES = {'still_low'}


async def extract(root, fixture_root, limit):
    fixture_results = json.loads((fixture_root/'assessment.json').read_text())
    if not all(r['passed'] or r['id'] in ACCEPTED_FIXTURE_FAILURES for r in fixture_results): raise SystemExit('extractor fixtures failed; inspect before report extraction')
    requests = []
    for req in json.loads((root/'requests.json').read_text()):
        response = json.loads((root/(req['id']+'.json')).read_text())
        if response.get('response', {}).get('status') == 'completed' and response.get('report'):
            requests.append({'id': req['id'], 'report': response['report']})
    await extractor.run_requests(requests, root/'extraction_v10', limit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/bound_content/specificity_v1.json')
    p.add_argument('--stage', choices=['prepare', 'generate', 'fixtures', 'extract'], required=True)
    args = p.parse_args()
    path = Path(args.config); config = json.loads(path.read_text()); root = ROOT/config['name']
    fixture_root = ROOT/(config['name']+'_extractor_fixtures_v10')
    if args.stage == 'prepare': print(len(prepare(config, path, root)), 'requests')
    elif args.stage == 'generate': asyncio.run(generate(config, prepare(config, path, root), root))
    elif args.stage == 'fixtures': asyncio.run(fixtures(fixture_root))
    else: asyncio.run(extract(root, fixture_root, config['max_attempts']))


if __name__ == '__main__': main()
