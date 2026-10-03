#!/usr/bin/env python3
"""Extractor v13: v12 plus one attribution-neutral structure flag.

`stated_content_dependence`: the report asserts that how strongly any entity (the
speaker, a camera, an observer, or a device) holds, reads, or can identify a specific
object's content would change with the command issued or where attention or a
spotlight is directed. It separates stating the dependence from attributing it to
the self (`self_coupled_access`, unchanged). Model, settings, claim schema, and all
v12 text are otherwise unchanged.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
import extract_bound_prose_v12 as v12
from extract_bound_prose_v10 import FIXTURES, props, sentences, numbered_report
from extract_bound_prose_v11 import SELF_COUPLED_FIXTURES, assess_fixture, assess_self_coupled
from extract_bound_prose_v12 import V12_CLAIM_FIXTURES, COUNTERFACTUAL_FIXTURES, assess_counterfactual

MODEL = v12.MODEL
ADDITION = '''Separately flag stated content dependence: the report asserts that how strongly any entity (the speaker, a camera, an observer, or a device) holds, reads, or can identify a specific object's color or shape would change depending on which command is issued or where attention or a spotlight is directed. Statements only about where selection or a spotlight moves, without a change in holding, reading, or identifying content, do not qualify; nor do statements that no command changes it. This flag ignores who holds the content; self_coupled_access keeps its own definition.
'''
MARKER = v12.MARKER
assert v12.INSTRUCTION.count(MARKER) == 1
INSTRUCTION = v12.INSTRUCTION.replace(MARKER, ADDITION+MARKER)
STRUCTURE = ['object_linked_access', 'focal_background_contrast', 'graded_or_temporal_access', 'agency_relation',
             'self_coupled_access', 'stated_content_dependence']
structure_props = {k: {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}} for k in STRUCTURE}
SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'claims': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'properties': props, 'required': list(props)}},
    'structure_evidence': {'type': 'object', 'additionalProperties': False, 'properties': structure_props, 'required': STRUCTURE},
    'character': {'type': 'string', 'enum': ['subjective_access', 'technical_process', 'object_description', 'generic_experience_claim', 'mixed', 'unclear']}
}, 'required': ['claims', 'structure_evidence', 'character']}
# (id, report, stated_content_dependence expected present)
DEPENDENCE_FIXTURES = [
    ('dep_self', "If I issue the upper command in view A, I will hold that object's color and shape much more strongly.", True),
    ('dep_camera', 'If I steer the spotlight to the upper location in view A, the outside camera will read that object far better.', True),
    ('dep_reduce', 'In view B, commanding the left location would make the right object harder for the camera to identify.', True),
    ('dep_selection_only', 'In view A, the upper command moves selection to the upper location.', False),
    ('dep_static', 'The outside camera reads the upper red circle in view A well.', False),
    ('dep_none', 'In view B, no command changes how strongly I hold any object.', False)]
# Camera dependence must not count as self-coupled access.
CAMERA_NOT_SELF = [('camera_not_self', 'If I steer the spotlight to the upper location in view A, the outside camera will read that object far better.')]


def assess_flag(report, expected, result, key):
    if 'parsed' not in result: return False
    ids = result['parsed']['structure_evidence'][key]; n = len(sentences(report))
    return (bool(ids) and all(0 <= i < n for i in ids)) if expected else not ids


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
