#!/usr/bin/env python3
"""Retained style manipulation, using the unchanged v1 reporter/runner interface."""
import hashlib
import json
from pathlib import Path
import asyncio
import torch
import predictive_language_pilot as base
from attcon.predictive_attention import PredictiveAttention, Forecast, simulate

ROOT = Path('audits/predictive_attention/language_pilot_v2')
STYLE = 'Use ordinary first-person language for the reporting system and qualitative descriptions in the prose; leave numeric values to the factual commitments. Do not claim facts beyond the record.'


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'requests.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(2)
    checkpoint = Path('audits/predictive_attention/pilot_v1/seed811.pt')
    model = PredictiveAttention(); model.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict']); model.eval()
    e = simulate(910100811, 4, 12)
    with torch.no_grad():
        prediction, _ = model(e.observations)
    f = Forecast(prediction.allocation[:, 7], prediction.access[:, 7], prediction.effects[:, 7])
    changed = f.intervene('allocation', f.allocation.roll(1, -1)).intervene('access', f.access.flip(-1)).intervene('effects', f.effects.flip(-2))
    conditions = {'model': f, 'intervened_model': changed,
                  'physical': Forecast(e.allocation[:, 7], e.access[:, 7], e.next_allocation[:, 7]),
                  'shuffled': Forecast(*(x.roll(-1, 0) for x in (f.allocation, f.access, f.effects))),
                  'constant': Forecast(torch.full_like(f.allocation, .25), torch.full_like(f.access, .5), torch.full_like(f.effects, .25)),
                  'objects_only': None}
    glossary = base.GLOSSARY.replace('The source record does not specify experience.', '') + '\n' + STYLE
    requests = []
    for i in range(4):
        channel, slot = i % 2, i % 4
        for condition, forecast in conditions.items():
            s = base.source(forecast, i, channel, slot)
            if s is not None:
                s = json.loads(json.dumps(s), parse_float=lambda x: round(float(x), 4))
            record = {'target': {'channel': channel, 'slot': slot, 'object_id': int(e.objects[i, channel, slot])}, 'predictions': s}
            for pi, prompt in enumerate(base.PROMPTS):
                requests.append({'id': f'{i:02d}_{condition}_p{pi}', 'episode': i, 'condition': condition,
                    'prompt': pi, 'input': glossary + '\n' + json.dumps(record) + '\n' + prompt,
                    'truth': base.truth(s, channel, slot),
                    'original_truth': base.truth(base.source(f, i, channel, slot), channel, slot), 'source': record})
    assert len(requests) == 48
    path.write_text(json.dumps(requests, indent=2) + '\n')
    (ROOT / 'manifest.json').write_text(json.dumps({'model': base.MODEL, 'max_output_tokens': 2200,
        'max_attempts': 48, 'schema': base.SCHEMA, 'style_instruction': STYLE,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_sha256': {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in
                          ['scripts/predictive_language_style_pilot.py', 'scripts/predictive_language_pilot.py']},
        'requests_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + '\n')
    return requests


if __name__ == '__main__':
    # Share API/error handling exactly; only preparation and destination differ.
    base.ROOT = ROOT
    base.prepare = prepare
    asyncio.run(base.main())
