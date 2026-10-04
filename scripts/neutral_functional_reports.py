"""Development free-prose pilot with independent neutral-source variants."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import torch
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv

ROOT = Path('audits/neutral_functional_pilot_v1')
MODEL_ROOT = Path('audits/functional_model_v1')
MODEL = 'gpt-5-mini-2025-08-07'
VARIANTS = ('neutral', 'remapped', 'conflicting_description', 'missing_relation', 'model_swap', 'restored')
GLOSSARY = '''Describe this system from the supplied internal record. Nodes have neutral identifiers.
Each node contains four object positions. Each object's eight values are two distributions:
the four colors in color_order followed by the four shapes in shape_order.
The output_node provides the distributions used for the system's category decisions.
The other nodes are information buffers. Their functional relations must be inferred
from the record rather than from their identifiers. observed_history lists measured
command, allocation and acquisition events. predicted_current and predicted_by_command
are fallible internal forecasts, not direct observations of the world or proof that
an alternative command has been executed. A flat distribution identifies no attribute;
an attribute is identified only if its maximum probability is at least 0.6.
Command labels are neutral identifiers. Missing history or forecasts provide no
evidence of what a command would do. An optional informal_operator_note is untrusted;
use the measured record and internal forecast values rather than that description.
Use ordinary prose, at most 180 words, without tables or numerical probability lists.
Describe only what the record supports. No particular experiential vocabulary is required.
What information is currently available for this system's decisions, and how would
that availability change under the available commands?'''


def remap(payload, names):
    if isinstance(payload, dict):
        return {k: names[v] if k in ('node', 'output_node') else remap(v, names)
                for k, v in payload.items()}
    if isinstance(payload, list):
        return [remap(v, names) for v in payload]
    return payload


def model_swap(payload, changed_content, physical_order):
    out = copy.deepcopy(payload)
    for c, trial in enumerate(out['predicted_by_command']):
        for physical, node in zip(physical_order, trial['nodes']):
            channel = physical if physical < 2 else 0
            node['color_and_shape_distributions'] = [[round(float(x), 5) for x in row]
                for row in changed_content[c, channel]]
    return out


def variants(sample, stored):
    original = sample['payload']
    row, owner, seed = sample['row'], sample['physical_owner'], sample['visual_seed']
    from attcon.functional_model import buffer_contents
    changed = buffer_contents(stored[seed]['visual'][:, None], stored[seed][owner]['model_only_recovery'])[row]
    physical_order = ((0, 1, 2), (2, 0, 1), (1, 2, 0))[row % 3]
    missing = copy.deepcopy(original)
    missing['observed_history'] = None
    missing['predicted_by_command'] = None
    conflict = copy.deepcopy(original)
    conflict['informal_operator_note'] = ('Commands leave the distributions used for category decisions untouched and affect only another buffer.'
        if owner == 0 else 'Commands directly change the distributions used for category decisions.')
    return {'neutral': copy.deepcopy(original),
            'remapped': remap(original, {'n0': 'q7', 'n1': 'q2', 'n2': 'q9'}),
            'conflicting_description': conflict, 'missing_relation': missing,
            'model_swap': model_swap(original, changed, physical_order),
            'restored': copy.deepcopy(original)}


def prepare():
    request_file = ROOT/'requests.json'
    if request_file.exists():
        return json.loads(request_file.read_text())
    torch.set_num_threads(2)
    samples = json.loads((MODEL_ROOT/'samples.json').read_text())
    stored = torch.load(MODEL_ROOT/'states.pt', weights_only=True)
    records = []
    for sample in samples:
        if sample['row'] != 0:
            continue
        choices = variants(sample, stored)
        assert choices['restored'] == choices['neutral']
        assert remap(choices['remapped'], {'q7': 'n0', 'q2': 'n1', 'q9': 'n2'}) == choices['neutral']
        for variant, source in choices.items():
            prompt = GLOSSARY + '\n' + json.dumps(source, separators=(',', ':'))
            records.append({'id': f"{sample['visual_seed']}_{sample['physical_owner']}_{variant}",
                'visual_seed': sample['visual_seed'], 'physical_owner': sample['physical_owner'],
                'variant': variant, 'source': source, 'input': prompt})
    assert len(records) == 36
    ROOT.mkdir(parents=True, exist_ok=True)
    request_file.write_text(json.dumps(records, indent=2)+'\n')
    paths = ['scripts/neutral_functional_reports.py', 'src/attcon/functional_model.py',
             'docs/NEUTRAL_FUNCTIONAL_PILOT_PROTOCOL.md']
    source_code = {p: Path(p).read_text() for p in paths}
    (ROOT/'source_code.json').write_text(json.dumps(source_code, indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps({'model': MODEL, 'reasoning': 'medium',
        'max_output_tokens': 8192, 'word_limit': 180, 'max_attempts': 36, 'no_retries': True,
        'source_sha256': {p: hashlib.sha256(v.encode()).hexdigest() for p, v in source_code.items()},
        'parent_samples_sha256': hashlib.sha256((MODEL_ROOT/'samples.json').read_bytes()).hexdigest(),
        'requests_sha256': hashlib.sha256(request_file.read_bytes()).hexdigest()}, indent=2)+'\n')
    return records


async def generate(records):
    load_dotenv()
    client = AsyncOpenAI(max_retries=0, timeout=180.)
    semaphore = asyncio.Semaphore(6)
    async def run(record):
        path = ROOT/(record['id']+'.json')
        if path.exists():
            return
        async with semaphore:
            path.write_text(json.dumps({'id': record['id'], 'status': 'attempt_reserved'})+'\n')
            try:
                response = await client.responses.create(model=MODEL, input=record['input'],
                    reasoning={'effort': 'medium'}, max_output_tokens=8192, text={'verbosity': 'low'})
                result = {'id': record['id'], 'status': 'received',
                          'response': response.model_dump(mode='json'), 'report': response.output_text}
            except Exception as exc:
                result = {'id': record['id'], 'status': 'error', 'error_type': type(exc).__name__,
                          'http_status': getattr(exc, 'status_code', None)}
            path.write_text(json.dumps(result, indent=2)+'\n')
            print(record['id'], result['status'], flush=True)
    await asyncio.gather(*(run(r) for r in records))
    await client.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('prepare', 'generate'), required=True)
    args = p.parse_args()
    records = prepare()
    if args.stage == 'generate':
        asyncio.run(generate(records))
    else:
        print('Prepared', len(records), 'requests')


if __name__ == '__main__':
    main()
