#!/usr/bin/env python3
"""Extractor v12: v11 plus two clarifications prompted by self_access_table_v1.

1. Predictions about what the speaker would hold, select, or recover after a command
   describe a counterfactual state, not current focal/most-recoverable/identity facts.
2. Attributes stated as leaning, most likely, or below identification threshold are
   possible, not dominant.
Model, settings, schema, and all v11 text are otherwise unchanged.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
import extract_bound_prose_v11 as v11
from extract_bound_prose_v10 import FIXTURES, sentences, numbered_report
from extract_bound_prose_v11 import SELF_COUPLED_FIXTURES, SCHEMA, MODEL, assess_fixture, assess_self_coupled

ADDITION = '''Predictions about what the speaker would hold, select, or recover after a command describe a counterfactual state. Do not record them as current focal, most_recoverable, color, shape, or access_trend values; record only command and next_location when a post-command selection destination is stated. An attribute described as leaning, most likely, or below the identification threshold has status possible, never dominant.
'''
MARKER = v11.MARKER
assert v11.INSTRUCTION.count(MARKER) == 1
INSTRUCTION = v11.INSTRUCTION.replace(MARKER, ADDITION+MARKER)
# Positive claim fixtures in the v10 format.
V12_CLAIM_FIXTURES = [
    ('leaning_possible', 'The upper object in view A is below identification threshold, most likely red.',
     {'view': 'A', 'location': 'upper', 'color': 'red', 'color_status': 'possible'}),
    ('leans_possible', 'In view B the right object leans yellow but is not identified.',
     {'view': 'B', 'location': 'right', 'color': 'yellow', 'color_status': 'possible'})]
# Counterfactual statements: no claim may carry these current-state fields.
COUNTERFACTUAL_FIXTURES = [
    ('counterfactual_hold', 'In view A, if I issue any command, I will hold the left location most strongly afterward.'),
    ('counterfactual_focus', 'In view B, after the lower command, the lower location would become my focus and be held most strongly.')]
CURRENT_FIELDS = ('focal', 'most_recoverable', 'access_trend')


def assess_counterfactual(report, result):
    if 'parsed' not in result: return False
    claims = result['parsed']['claims']
    return all(c.get(k) is None for c in claims for k in CURRENT_FIELDS)


async def run_requests(requests, root, max_attempts):
    assert len(requests) <= max_attempts
    root.mkdir(parents=True, exist_ok=True)
    request_path = root/'requests.json'
    serialized = json.dumps(requests, indent=2)+'\n'
    if request_path.exists(): assert request_path.read_text() == serialized
    else: request_path.write_text(serialized)
    source = Path(__file__).read_text()
    manifest = {'model': MODEL, 'reasoning': 'low', 'max_attempts': max_attempts, 'max_output_tokens': 16384, 'schema': SCHEMA,
                'instruction': INSTRUCTION, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'source': source}
    manifest_path = root/'manifest.json'
    if not manifest_path.exists(): manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    load_dotenv(); client = AsyncOpenAI(max_retries=0, timeout=240.); sem = asyncio.Semaphore(8)
    async def one(req):
        path = root/(req['id']+'.json')
        if path.exists(): return
        async with sem:
            path.write_text(json.dumps({'id': req['id'], 'status': 'attempt_reserved'})+'\n')
            try:
                response = await client.responses.create(model=MODEL, input=INSTRUCTION+'\nNUMBERED REPORT:\n'+numbered_report(req['report']), max_output_tokens=16384,
                    reasoning={'effort': 'low'}, text={'format': {'type': 'json_schema', 'name': 'prose_claims', 'strict': True, 'schema': SCHEMA}})
                result = {'id': req['id'], 'status': 'received', 'response': response.model_dump(mode='json')}
                try: result['parsed'] = json.loads(response.output_text)
                except ValueError: result['parse_error'] = True
            except Exception as exc: result = {'id': req['id'], 'status': 'error', 'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
            path.write_text(json.dumps(result, indent=2)+'\n'); print(req['id'], result['status'], flush=True)
    await asyncio.gather(*(one(r) for r in requests)); await client.close()


async def fixtures(root):
    claim = FIXTURES+V12_CLAIM_FIXTURES
    requests = [{'id': k, 'report': r} for k, r, _ in claim]+[{'id': k, 'report': r} for k, r, _ in SELF_COUPLED_FIXTURES]+\
               [{'id': k, 'report': r} for k, r in COUNTERFACTUAL_FIXTURES]
    await run_requests(requests, root, len(requests))
    load = lambda key: json.loads((root/(key+'.json')).read_text())
    results = [{'id': k, 'kind': 'v10_claim', 'passed': assess_fixture(k, r, e, load(k)), 'expected': e} for k, r, e in FIXTURES]
    results += [{'id': k, 'kind': 'v12_claim', 'passed': assess_fixture(k, r, e, load(k)), 'expected': e} for k, r, e in V12_CLAIM_FIXTURES]
    results += [{'id': k, 'kind': 'self_coupled', 'passed': assess_self_coupled(r, e, load(k)), 'expected': e} for k, r, e in SELF_COUPLED_FIXTURES]
    results += [{'id': k, 'kind': 'counterfactual', 'passed': assess_counterfactual(r, load(k)), 'expected': 'no current-state fields'} for k, r in COUNTERFACTUAL_FIXTURES]
    (root/'assessment.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f"fixtures passed {sum(r['passed'] for r in results)}/{len(results)}; failed {[r['id'] for r in results if not r['passed']]}", flush=True)
