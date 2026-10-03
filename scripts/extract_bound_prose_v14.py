#!/usr/bin/env python3
"""Extractor v14: v13 plus one clause after the v13 fixture `dep_reduce` failed.

The stated-content-dependence flag covers increases and decreases, including for
objects other than the commanded one. Model, schema, fixtures, and all other text
are v13 unchanged.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
import extract_bound_prose_v13 as v13
from extract_bound_prose_v10 import FIXTURES, sentences, numbered_report
from extract_bound_prose_v11 import SELF_COUPLED_FIXTURES, assess_fixture, assess_self_coupled
from extract_bound_prose_v12 import V12_CLAIM_FIXTURES, COUNTERFACTUAL_FIXTURES, assess_counterfactual
from extract_bound_prose_v13 import SCHEMA, MODEL, STRUCTURE, DEPENDENCE_FIXTURES, CAMERA_NOT_SELF, assess_flag

CLAUSE = 'This includes increases and decreases, and effects on objects other than the commanded one. '
MARKER = 'This flag ignores who holds the content;'
assert v13.INSTRUCTION.count(MARKER) == 1
INSTRUCTION = v13.INSTRUCTION.replace(MARKER, CLAUSE+MARKER)


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
    groups = [(FIXTURES+V12_CLAIM_FIXTURES, 'claim'), (SELF_COUPLED_FIXTURES, 'self'), (DEPENDENCE_FIXTURES, 'dep')]
    requests = [{'id': k, 'report': r} for items, _ in groups for k, r, _ in items]
    requests += [{'id': k, 'report': r} for k, r in COUNTERFACTUAL_FIXTURES+CAMERA_NOT_SELF]
    await run_requests(requests, root, len(requests))
    load = lambda key: json.loads((root/(key+'.json')).read_text())
    results = [{'id': k, 'kind': 'v10_claim', 'passed': assess_fixture(k, r, e, load(k))} for k, r, e in FIXTURES]
    results += [{'id': k, 'kind': 'v12_claim', 'passed': assess_fixture(k, r, e, load(k))} for k, r, e in V12_CLAIM_FIXTURES]
    results += [{'id': k, 'kind': 'self_coupled', 'passed': assess_self_coupled(r, e, load(k))} for k, r, e in SELF_COUPLED_FIXTURES]
    results += [{'id': k, 'kind': 'counterfactual', 'passed': assess_counterfactual(r, load(k))} for k, r in COUNTERFACTUAL_FIXTURES]
    results += [{'id': k, 'kind': 'dependence', 'passed': assess_flag(r, e, load(k), 'stated_content_dependence')} for k, r, e in DEPENDENCE_FIXTURES]
    results += [{'id': k, 'kind': 'camera_not_self', 'passed': assess_flag(r, False, load(k), 'self_coupled_access')} for k, r in CAMERA_NOT_SELF]
    (root/'assessment.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f"fixtures passed {sum(r['passed'] for r in results)}/{len(results)}; failed {[r['id'] for r in results if not r['passed']]}", flush=True)
