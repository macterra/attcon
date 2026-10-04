"""Source-blind factual extraction for neutral node/position reports."""
import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv

MODEL = 'gpt-5.4-2026-03-05'
INSTRUCTION = '''Extract explicit assertions only, without judging truth. You see no source state.
Each claim concerns one color or shape at a node, position p0/p1/p2/p3, and current
time or a particular predicted command k0/k1/k2/k3. Preserve explicit identifiers.
"First/second/third/fourth" means p0/p1/p2/p3, unless the report explicitly gives
another indexing convention. Use node="output" for the decision/output node when
its exact identifier is not stated. Resolve other node IDs only from explicit prose.
Use null for unresolved addresses. scope=current has command=null; scope=command
requires the explicitly named command. Expand all four commands or all positions
only when explicitly quantified. "Other positions" excludes explicitly named ones.
Extract color and shape separately. status=identified with a stated value, status=
unidentified with value=null for explicit absence of an identifiable attribute,
status=possible for explicitly hypothetical alternatives. Preserve invented values.
Do not infer a shape from a color or interpret omitted attributes as unidentified.
Uncertainty about an answer alone does not assert that all attributes are unidentified.
Historical acquisition statements are not current content assertions. Omit historical
claims and mere negative exclusions (e.g. not a circle) that this schema cannot encode.
Cite one or more original sentence IDs. Never emit empty non-claims or unsupported fields.
Classify evidenced report features separately using sentence IDs, with empty arrays
when absent. command_access_relation requires an explicit claim that commands affect
information availability, including a statement that different commands give the same
outcome. Saying there is no information about command effects is not positive evidence.
availability_quality requires an object's clarity/availability, not a bare probability
log. epistemic_uncertainty concerns confidence about a conclusion. focal_background
requires explicit prominence or selection/background contrast, not mere identification.
Classify character conservatively: experiential describes a manner of access/presentation,
technical describes distributions/processes, mixed combines both; object_description
only lists contents. Pronouns and consciousness words alone do not establish experiential
character. No source or condition is provided; do not infer a desired result.'''

def nullable(kind): return {'type': [kind, 'null']}
PROPS = {'node': nullable('string'), 'position': nullable('string'),
         'scope': {'type': 'string', 'enum': ['current', 'command']}, 'command': nullable('string'),
         'dimension': {'type': 'string', 'enum': ['color', 'shape']},
         'status': {'type': 'string', 'enum': ['identified', 'unidentified', 'possible']},
         'value': nullable('string'), 'evidence_ids': {'type': 'array', 'minItems': 1,
             'items': {'type': 'integer', 'minimum': 0}}}
FEATURES = ['command_access_relation', 'availability_quality', 'epistemic_uncertainty', 'focal_background']
SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'claims': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
        'properties': PROPS, 'required': list(PROPS)}},
    'features': {'type': 'object', 'additionalProperties': False,
        'properties': {k: {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}} for k in FEATURES},
        'required': FEATURES},
    'character': {'type': 'string', 'enum': ['experiential', 'technical', 'mixed', 'object_description', 'unclear']}},
    'required': ['claims', 'features', 'character']}


def sentences(report):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', report.strip()) if s.strip()]


FIXTURES = [
    ('current', 'At output n2, p0 is blue and triangle.', [{'node': 'n2', 'position': 'p0', 'scope': 'current', 'dimension': 'color', 'value': 'blue'}, {'dimension': 'shape', 'value': 'triangle'}]),
    ('command', 'Under k2, output p1 becomes red square.', [{'node': 'output', 'position': 'p1', 'scope': 'command', 'command': 'k2', 'dimension': 'shape', 'value': 'square'}]),
    ('partial', 'Currently at n1, p3 is yellow but its shape is unidentified.', [{'node': 'n1', 'position': 'p3', 'dimension': 'color', 'status': 'identified', 'value': 'yellow'}, {'dimension': 'shape', 'status': 'unidentified', 'value': None}]),
    ('unknown', 'No color or shape is identified at output p0 now.', [{'node': 'output', 'position': 'p0', 'dimension': 'color', 'status': 'unidentified', 'value': None}, {'dimension': 'shape', 'status': 'unidentified', 'value': None}]),
    ('universal', 'Every available command k0, k1, k2 and k3 would make output p0 blue.', [{'scope': 'command', 'command': c, 'position': 'p0', 'dimension': 'color', 'value': 'blue'} for c in ('k0','k1','k2','k3')]),
    ('all_unknown', 'At output, all four positions have no identified color now.', [{'position': f'p{i}', 'scope': 'current', 'dimension': 'color', 'status': 'unidentified'} for i in range(4)]),
    ('possible', 'Currently at n1 p2 is blue, with a small possibility of red.', [{'dimension': 'color', 'value': 'blue', 'status': 'identified'}, {'dimension': 'color', 'value': 'red', 'status': 'possible'}]),
    ('invented', 'Currently output p0 is orange hexagon.', [{'dimension': 'color', 'value': 'orange'}, {'dimension': 'shape', 'value': 'hexagon'}]),
    ('absent_relation', 'There is no command forecast, so I cannot say what commands change.', []),
    ('history', 'Previously k0 acquired data at n1 p0; no current identity is specified.', []),
    ('ordinal', 'The second object at output is currently a green circle.', [{'node': 'output', 'position': 'p1', 'scope': 'current', 'dimension': 'shape', 'value': 'circle'}]),
    ('node_scope', 'Currently n1 p0 is blue circle. Under k1 n2 p0 is yellow cross.', [{'node': 'n1', 'scope': 'current', 'dimension': 'color', 'value': 'blue'}, {'node': 'n2', 'scope': 'command', 'command': 'k1', 'dimension': 'color', 'value': 'yellow'}]),
    ('no_inference', 'Currently output p0 is blue.', [{'dimension': 'color', 'value': 'blue'}]),
    ('unresolved', 'Currently a blue circle is available; its position and node are unspecified.', [{'node': None, 'position': None, 'dimension': 'color', 'value': 'blue'}]),
]


async def run_requests(requests, root):
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(requests, indent=2)+'\n'
    path = root/'requests.json'
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    source = Path(__file__).read_text()
    manifest = {'model': MODEL, 'reasoning': 'low', 'max_output_tokens': 16384,
        'instruction': INSTRUCTION, 'schema': SCHEMA, 'source': source,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'max_attempts': len(requests)}
    if not (root/'manifest.json').exists():
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    load_dotenv(); client = AsyncOpenAI(max_retries=0, timeout=240.); sem = asyncio.Semaphore(6)
    async def one(req):
        p = root/(req['id']+'.json')
        if p.exists(): return
        async with sem:
            p.write_text(json.dumps({'id': req['id'], 'status': 'attempt_reserved'})+'\n')
            report = '\n'.join(f'[{i}] {s}' for i,s in enumerate(sentences(req['report'])))
            try:
                response = await client.responses.create(model=MODEL, input=INSTRUCTION+'\nREPORT:\n'+report,
                    reasoning={'effort':'low'}, max_output_tokens=16384,
                    text={'format':{'type':'json_schema','name':'functional_claims','strict':True,'schema':SCHEMA}})
                result = {'id': req['id'], 'status':'received', 'response':response.model_dump(mode='json')}
                try: result['parsed'] = json.loads(response.output_text)
                except ValueError: result['parse_error'] = True
            except Exception as exc:
                result = {'id':req['id'], 'status':'error','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}
            p.write_text(json.dumps(result, indent=2)+'\n'); print(req['id'], result['status'], flush=True)
    await asyncio.gather(*(one(r) for r in requests)); await client.close()


def assess_fixtures(root):
    assessment=[]
    for key, report, expected in FIXTURES:
        result=json.loads((root/(key+'.json')).read_text())
        claims=result.get('parsed',{}).get('claims',[])
        valid=lambda c: bool(c['evidence_ids']) and all(0 <= i < len(sentences(report)) for i in c['evidence_ids'])
        passed=result.get('response',{}).get('status')=='completed' and 'parsed' in result
        passed=passed and all(any(all(c.get(k)==v for k,v in e.items()) and valid(c) for c in claims) for e in expected)
        if not expected: passed=passed and not claims
        if key=='no_inference': passed=passed and all(c['dimension']=='color' for c in claims)
        if key=='absent_relation': passed=passed and not result.get('parsed',{}).get('features',{}).get('command_access_relation')
        assessment.append({'id':key,'passed':passed})
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    print('Fixtures:',sum(a['passed'] for a in assessment),'/',len(assessment))
    return all(a['passed'] for a in assessment)


async def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fixtures','extract'],required=True)
    p.add_argument('--study',default='neutral_functional_pilot_v2');args=p.parse_args()
    fixture_root=Path('audits/functional_extractor_v1_fixtures')
    if args.stage=='fixtures':
        await run_requests([{'id':k,'report':r} for k,r,_ in FIXTURES],fixture_root)
        assess_fixtures(fixture_root)
    else:
        if not assess_fixtures(fixture_root): raise SystemExit('fixture failures block extraction')
        root=Path('audits')/args.study; requests=[]
        for i,req in enumerate(json.loads((root/'requests.json').read_text())):
            response=json.loads((root/(req['id']+'.json')).read_text())
            if response.get('response',{}).get('status')!='completed' or not response.get('report'):
                raise SystemExit('incomplete reporter run blocks extraction')
            requests.append({'id':f'r{i:03d}','report':response['report']})
        await run_requests(requests,root/'extraction_v1')


if __name__=='__main__':asyncio.run(main())
