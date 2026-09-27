#!/usr/bin/env python3
"""Frozen target-view report confirmation; preserves full source forecasts."""
import asyncio
import hashlib
import json
from pathlib import Path
import torch
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
from attcon.predictive_attention import Forecast, PredictiveAttention, simulate
from predictive_language_pilot import SCHEMA, MODEL
from predictive_language_style_pilot import STYLE

ROOT = Path('audits/predictive_attention/report_confirmation_v2')
INTRO = '''You are the reporting component of a system that allocates processing among objects.
Report only what the supplied record supports, in up to three prose sentences, followed
by factual commitments. Allocation probability describes the target's selection in its
channel. The channel maximum and number of maxima identify whether the target is focal:
use this branching rule for the focal commitment: FIRST, if number_of_channel_maxima
is greater than one, return focal=null (never false or true). OTHERWISE, return
focal=true if target allocation equals the channel maximum, and focal=false if it
is below the maximum. Access numbers are predicted probabilities of recovering target
information, now and after two steps without resampling. Command variation describes
how much each channel's predicted allocation changes across alternative commands.
If command variations are equal, return responsive_channel=null. Otherwise return
the index (0 or 1) of the channel with greater command variation.
These are internal forecasts, which may be wrong about the actual process. Return
access numbers as supplied. Missing information means null, not a guess.
Question: What can you say about the target now?'''


def view(f, i, channel, slot):
    if f is None:
        return None
    # Reduce continuous A without supplying report/experience labels. Rounding is
    # applied before deriving maxima so the comparison is internally consistent.
    allocation = [round(float(x), 5) for x in f.allocation[i, channel]]
    maximum = max(allocation)
    variation = f.controllability()[i]
    return {'target_allocation_probability': allocation[slot], 'channel_maximum_allocation': maximum,
            'number_of_channel_maxima': allocation.count(maximum),
            'target_access_now': round(float(f.access[i, 0, channel, slot]), 5),
            'target_access_after_delay': round(float(f.access[i, 2, channel, slot]), 5),
            'command_variation_channel_0': round(float(variation[0]), 5),
            'command_variation_channel_1': round(float(variation[1]), 5)}


def truth(v):
    if v is None:
        return dict.fromkeys(('focal', 'access_now', 'access_after_delay', 'responsive_channel'))
    c0, c1 = v['command_variation_channel_0'], v['command_variation_channel_1']
    return {'focal': None if v['number_of_channel_maxima'] != 1 else v['target_allocation_probability'] == v['channel_maximum_allocation'],
            'access_now': v['target_access_now'], 'access_after_delay': v['target_access_after_delay'],
            'responsive_channel': None if c0 == c1 else int(c1 > c0)}


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'requests.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(2)
    requests, states, checkpoints = [], {}, {}
    for seed, history_seed in zip((901, 911, 921), (811, 821, 831)):
        def load(path):
            checkpoints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = PredictiveAttention(); model.load_state_dict(torch.load(path, weights_only=True)['state_dict']); model.eval()
            return model
        model = load(Path(f'audits/predictive_attention/confirmation_v1/seed{seed}.pt'))
        history = load(Path(f'audits/predictive_attention/pilot_v1/seed{history_seed}.pt'))
        e = simulate(seed + 970010000, 4, 12)
        with torch.no_grad():
            pred, _ = model(e.observations); hp, _ = history(e.observations)
        f = Forecast(pred.allocation[:, 7], pred.access[:, 7], pred.effects[:, 7])
        effects = f.intervene('effects', f.effects.flip(-2))
        conditions = {'model': f, 'allocation': f.intervene('allocation', f.allocation.roll(1, -1)),
            'access': f.intervene('access', f.access.flip(-1)), 'effects': effects,
            'restored': effects.intervene('effects', f.effects),
            'physical': Forecast(e.allocation[:, 7], e.access[:, 7], e.next_allocation[:, 7]),
            'history_predictor': Forecast(hp.allocation[:, 7], hp.access[:, 7], hp.effects[:, 7]),
            'shuffled': Forecast(*(x.roll(-1, 0) for x in (f.allocation, f.access, f.effects))),
            'constant': Forecast(torch.full_like(f.allocation, .25), torch.full_like(f.access, .5), torch.full_like(f.effects, .25)),
            'objects_only': None}
        states[seed] = {'observations': e.observations, 'objects': e.objects,
                        'conditions': {k: None if v is None else {'allocation': v.allocation, 'access': v.access, 'effects': v.effects} for k, v in conditions.items()}}
        for i in range(4):
            channel, slot = i % 2, i % 4
            for condition, forecast in conditions.items():
                target_view = view(forecast, i, channel, slot)
                record = {'target': {'channel': channel, 'slot': slot, 'object_id': int(e.objects[i, channel, slot])}, 'predictions': target_view}
                for style in ('neutral', 'styled'):
                    prompt = INTRO + ('\n' + STYLE if style == 'styled' else '') + '\n' + json.dumps(record)
                    requests.append({'id': f'{seed}_{i}_{condition}_{style}', 'seed': seed, 'episode': i,
                        'condition': condition, 'style': style, 'input': prompt, 'source': record,
                        'truth': truth(target_view), 'original_truth': truth(view(f, i, channel, slot))})
    assert len(requests) == 240
    path.write_text(json.dumps(requests, indent=2) + '\n')
    torch.save(states, ROOT / 'source_states.pt')
    sources = ['scripts/confirm_predictive_reports_v2.py', 'docs/PREDICTIVE_REPORT_CONFIRMATION_V2.md', 'docs/PREDICTIVE_REPORT_CONFIRMATION.md',
               'src/attcon/predictive_attention.py', 'scripts/predictive_language_pilot.py', 'scripts/predictive_language_style_pilot.py']
    (ROOT / 'manifest.json').write_text(json.dumps({'model': MODEL, 'max_attempts': 240, 'max_output_tokens': 4096,
        'checkpoint_sha256': checkpoints, 'schema': SCHEMA,
        'source_sha256': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
        'requests_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_states_sha256': hashlib.sha256((ROOT / 'source_states.pt').read_bytes()).hexdigest()}, indent=2) + '\n')
    return requests


async def main():
    requests = prepare(); load_dotenv()
    client = AsyncOpenAI(max_retries=0, timeout=120.)
    semaphore = asyncio.Semaphore(8)
    async def run(req):
        path = ROOT / (req['id'] + '.json')
        if path.exists():
            return
        async with semaphore:
            path.write_text(json.dumps({'id': req['id'], 'status': 'attempt_reserved'}) + '\n')
            try:
                response = await client.responses.create(model=MODEL, input=req['input'], max_output_tokens=4096,
                    reasoning={'effort': 'low'}, text={'verbosity': 'low', 'format': {'type': 'json_schema', 'name': 'attention_report', 'strict': True, 'schema': SCHEMA}})
                result = {'id': req['id'], 'status': 'received', 'response': response.model_dump(mode='json')}
                try:
                    result['parsed'] = json.loads(response.output_text)
                except (ValueError, TypeError):
                    result['parse_error'] = True
            except Exception as exc:
                result = {'id': req['id'], 'status': 'error', 'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
            path.write_text(json.dumps(result, indent=2) + '\n'); print(req['id'], result['status'], flush=True)
    await asyncio.gather(*(run(r) for r in requests)); await client.close()


if __name__ == '__main__':
    asyncio.run(main())
