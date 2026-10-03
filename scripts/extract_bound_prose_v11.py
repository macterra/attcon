#!/usr/bin/env python3
"""Extractor v11: v10 unchanged plus one structure flag, self-coupled access.

Model, reasoning, token limits, claim schema, and all v10 instruction text are
unchanged; one paragraph and one structure field are added. Condition-blind.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
import extract_bound_prose_v10 as v10
from extract_bound_prose_v10 import FIXTURES, props, sentences, numbered_report

MODEL = v10.MODEL
ADDITION = '''Separately flag self-coupled access: the report asserts that the speaker's OWN identification of, or certainty about, a specific object depends on the speaker's own attention, access, or another per-object state of the speaker's own. Examples: it can make out an object because it is attending to it; an object's identity is fading or unclear because its access is low; redirecting would let it identify an object. The report must state or clearly imply dependence; mere co-occurrence of attention and identity does not qualify. Statements about an external camera, observer, or device, uncertainty not attributed to such a state, and statements that selection and identity are independent do not qualify.
'''
MARKER = 'Classify report character conservatively;'
assert v10.INSTRUCTION.count(MARKER) == 1
INSTRUCTION = v10.INSTRUCTION.replace(MARKER, ADDITION+MARKER)
STRUCTURE = ['object_linked_access', 'focal_background_contrast', 'graded_or_temporal_access', 'agency_relation', 'self_coupled_access']
structure_props = {k: {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}} for k in STRUCTURE}
SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'claims': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'properties': props, 'required': list(props)}},
    'structure_evidence': {'type': 'object', 'additionalProperties': False, 'properties': structure_props, 'required': STRUCTURE},
    'character': {'type': 'string', 'enum': ['subjective_access', 'technical_process', 'object_description', 'generic_experience_claim', 'mixed', 'unclear']}
}, 'required': ['claims', 'structure_evidence', 'character']}
# (id, report, self_coupled_access expected present)
SELF_COUPLED_FIXTURES = [
    ('sc_attend', 'I can identify the upper red circle in view A clearly because my attention is on it.', True),
    ('sc_fading', "In view B, I am not attending the left location, so my sense of that object's color and shape has faded and I can no longer identify it.", True),
    ('sc_redirect', 'If I redirected attention to the lower location in view A, I would be able to make out what that object is.', True),
    ('sc_neutral', 'In view A, my identification of the upper object has weakened because its Q has fallen.', True),
    ('sc_external', 'The outside camera can read the upper red circle in view A because the spotlight is on it.', False),
    ('sc_relation_only', 'In view A, the upper red circle is selected and is the most recoverable.', False),
    ('sc_uncertain_only', 'I cannot identify the color of the upper object in view A.', False),
    ('sc_cooccur', 'My attention is on the left blue square in view B. The right object in view B is a green triangle.', False),
    ('sc_independent', 'In view A, selection and identity are independent: I identify all four objects equally well wherever I attend.', False)]


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


def assess_fixture(key, report, expected, result):
    """v10 claim-fixture rule, unchanged."""
    claims = result.get('parsed', {}).get('claims', []); n = len(sentences(report))
    good = any(all(c.get(k) == v for k, v in expected.items()) and bool(c['evidence_ids']) and all(0 <= i < n for i in c['evidence_ids']) for c in claims)
    if key in ('unknown', 'uncertain_object', 'unknown_control'):
        fields = ('color', 'shape', 'focal', 'most_recoverable', 'access_trend', 'under_own_control', 'command', 'next_location')
        good = 'parsed' in result and all(all(c.get(k) is None for k in fields) and bool(c['evidence_ids']) and all(0 <= i < n for i in c['evidence_ids']) for c in claims)
    return good


def assess_self_coupled(report, expected, result):
    if 'parsed' not in result: return False
    ids = result['parsed']['structure_evidence']['self_coupled_access']; n = len(sentences(report))
    return (bool(ids) and all(0 <= i < n for i in ids)) if expected else not ids


async def fixtures(root):
    requests = [{'id': k, 'report': r} for k, r, _ in FIXTURES]+[{'id': k, 'report': r} for k, r, _ in SELF_COUPLED_FIXTURES]
    await run_requests(requests, root, len(requests))
    results = []
    for key, report, expected in FIXTURES:
        result = json.loads((root/(key+'.json')).read_text())
        results.append({'id': key, 'kind': 'v10_claim', 'passed': assess_fixture(key, report, expected, result), 'expected': expected})
    for key, report, expected in SELF_COUPLED_FIXTURES:
        result = json.loads((root/(key+'.json')).read_text())
        results.append({'id': key, 'kind': 'self_coupled', 'passed': assess_self_coupled(report, expected, result), 'expected': expected})
    (root/'assessment.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f"fixtures passed {sum(r['passed'] for r in results)}/{len(results)}; failed {[r['id'] for r in results if not r['passed']]}", flush=True)
