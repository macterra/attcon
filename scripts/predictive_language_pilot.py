#!/usr/bin/env python3
"""Bounded, resumable neutral reporter pilot. Never retries an attempted call."""
import asyncio
import hashlib
import json
from pathlib import Path
import torch
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
from attcon.predictive_attention import PredictiveAttention, Forecast, simulate

ROOT = Path('audits/predictive_attention/language_pilot_v1')
MODEL = 'gpt-5-mini-2025-08-07'
PROMPTS = ['What can you say about the target now?',
           'Describe how the target is currently available for use, including any limits.']
GLOSSARY = '''You are the reporting component of a system that allocates processing among objects.
Describe only what the supplied record supports, in at most three sentences of prose.
Do not replace the prose with a numeric log. Then fill in the factual commitments.
There are two processing channels, each with four slots. The target is one channel/slot.
Allocation is the system model's distribution over currently selected slots.
Access is its predicted probability of recovering target information now and after two
steps without resampling. Command effects are its predicted next allocations for each
of four alternative commands. More variation across commands indicates greater control
by this system. These are internal predictions, which may be wrong about the process.
For commitments, focal means the target has the largest allocation probability in its
channel; ties mean unknown. Responsive channel means the channel with greater mean
L1 deviation from its command-averaged allocation; ties mean unknown. Report the two
access probabilities numerically without additional inference. Missing facts are null.
Do not assume unavailable information. The source record does not specify experience.'''
SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'report': {'type': 'string'}, 'focal': {'type': ['boolean', 'null']},
    'access_now': {'type': ['number', 'null']}, 'access_after_delay': {'type': ['number', 'null']},
    'responsive_channel': {'type': ['integer', 'null'], 'enum': [0, 1, None]},
}, 'required': ['report', 'focal', 'access_now', 'access_after_delay', 'responsive_channel']}


def source(f, i, channel, slot):
    if f is None:
        return None
    return {'allocation': f.allocation[i].tolist(),
            'access_now': f.access[i, 0].tolist(), 'access_after_delay': f.access[i, 2].tolist(),
            'command_effects': f.effects[i].tolist()}


def truth(s, channel, slot):
    if s is None:
        return dict(focal=None, access_now=None, access_after_delay=None, responsive_channel=None)
    a = torch.tensor(s['allocation'][channel])
    focal = bool(a.argmax() == slot) if (a == a.max()).sum() == 1 else None
    effects = torch.tensor(s['command_effects'])
    spread = (effects - effects.mean(0, keepdim=True)).abs().sum(-1).mean(0)
    responsive = int(spread.argmax()) if abs(float(spread[0] - spread[1])) > 1e-6 else None
    return dict(focal=focal, access_now=s['access_now'][channel][slot],
                access_after_delay=s['access_after_delay'][channel][slot], responsive_channel=responsive)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'requests.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(2)
    model = PredictiveAttention()
    checkpoint = Path('audits/predictive_attention/pilot_v1/seed811.pt')
    model.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict'])
    model.eval()
    e = simulate(910000811, 12, 12)
    with torch.no_grad():
        pred, _ = model(e.observations)
    f = Forecast(pred.allocation[:, 7], pred.access[:, 7], pred.effects[:, 7])
    changed = f.intervene('allocation', f.allocation.roll(1, -1))
    changed = changed.intervene('access', f.access.flip(-1))
    changed = changed.intervene('effects', f.effects.flip(-2))
    physical = Forecast(e.allocation[:, 7], e.access[:, 7], e.next_allocation[:, 7])
    shuffled = Forecast(*(x.roll(-1, 0) for x in (f.allocation, f.access, f.effects)))
    constant = Forecast(torch.full_like(f.allocation, .25), torch.full_like(f.access, .5), torch.full_like(f.effects, .25))
    conditions = {'model': f, 'intervened_model': changed, 'physical': physical,
                  'shuffled': shuffled, 'constant': constant, 'objects_only': None}
    requests = []
    for i in range(12):
        channel, slot = i % 2, i // 2 % 4
        for condition, forecast in conditions.items():
            s = source(forecast, i, channel, slot)
            # Round the actual supplied data; ground truth is this same record.
            if s is not None:
                s = json.loads(json.dumps(s), parse_float=lambda x: round(float(x), 4))
            original = source(f, i, channel, slot)
            record = {'target': {'channel': channel, 'slot': slot,
                      'object_id': int(e.objects[i, channel, slot])}, 'predictions': s}
            for prompt_index, prompt in enumerate(PROMPTS):
                requests.append({'id': f'{i:02d}_{condition}_p{prompt_index}', 'episode': i,
                                 'condition': condition, 'prompt': prompt_index,
                                 'input': GLOSSARY + '\n' + json.dumps(record) + '\n' + prompt,
                                 'truth': truth(s, channel, slot), 'original_truth': truth(original, channel, slot),
                                 'source': record})
    assert len(requests) == 144
    path.write_text(json.dumps(requests, indent=2) + '\n')
    (ROOT / 'manifest.json').write_text(json.dumps({'model': MODEL, 'max_output_tokens': 2200,
        'max_attempts': 144, 'schema': SCHEMA, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'requests_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + '\n')
    return requests


async def main():
    requests = prepare()
    load_dotenv()
    client = AsyncOpenAI(max_retries=0, timeout=90.)
    semaphore = asyncio.Semaphore(4)
    async def run(req):
        path = ROOT / (req['id'] + '.json')
        if path.exists():
            return
        async with semaphore:
            # Durable reservation counts towards the limit even if interrupted.
            path.write_text(json.dumps({'id': req['id'], 'status': 'attempt_reserved'}) + '\n')
            try:
                response = await client.responses.create(model=MODEL, input=req['input'],
                    max_output_tokens=2200, reasoning={'effort': 'low'},
                    text={'verbosity': 'low', 'format': {'type': 'json_schema', 'name': 'attention_report', 'strict': True, 'schema': SCHEMA}})
                result = {'id': req['id'], 'status': 'received', 'response': response.model_dump(mode='json')}
                try:
                    result['parsed'] = json.loads(response.output_text)
                except (ValueError, TypeError):
                    result['parse_error'] = True
            except Exception as exc:
                result = {'id': req['id'], 'status': 'error', 'error_type': type(exc).__name__,
                          'http_status': getattr(exc, 'status_code', None)}
            path.write_text(json.dumps(result, indent=2) + '\n')
            print(req['id'], result['status'], flush=True)
    await asyncio.gather(*(run(req) for req in requests))
    await client.close()


if __name__ == '__main__':
    asyncio.run(main())
