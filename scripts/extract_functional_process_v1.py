"""Dedicated source-blind attention-process audit; no category or character task."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
from extract_functional_prose_v1 import sentences
from extract_functional_prose_v3 import PROCESS_PROPS, PROCESS_FIXTURES

MODEL='gpt-5.4-2026-03-05'
INSTRUCTION = """Extract explicit attention-process assertions only; do not judge truth.
You receive numbered prose without source state or conditions. Preserve named buffer
IDs. Use node=output only when the prose explicitly refers to an output/readout.
Unresolved addresses stay null; do not infer them from function or expected results.
selection: an asserted currently selected/focal position, or selected position forecast
under an explicitly named command k0/k1/k2/k3. position=null and value=p0/p1/p2/p3.
An explicitly asserted absence of an identified selected position IN THE MODEL has
value=null. Unspecified/not-stated selection in the REPORT asserts no model state and
must not be extracted. Explicitly asserted selection with unknown addresses may retain
null address/value. Historical selection is not current selection. High categorical
confidence, identifiability, clarity or recoverability does not imply selection.
recovery_trend: an explicit forecast that successful simulated access/recoverability
at p0/p1/p2/p3 declines, rises or stays steady over the next two steps without refresh.
Use value=declining/rising/steady and scope=current, command=null. Confidence about
answers is a separate quantity: its trend never establishes a recovery trend. When
confidence and recoverability have different trends, extract only recoverability.
When recovery is unspecified, no recovery claim is supported by an answer-confidence
trend. Do not treat the position of highest recovery as the selected position.
Current selection uses scope=current, command=null; command-predicted selection uses
scope=command with that explicit command. Preserve invented positions, identifiers or
values instead of repairing them. A vague change without direction is not a trend.
Expand all positions, both buffers, or all commands only when explicitly quantified
with identifiable scope. Some positions must retain unresolved position=null. Resolve
references only from explicit prose. Do not infer object identities or inspect category
claims, rhetoric, experiential character or the study's desired outcome.
Return only process_claims, citing nonempty original sentence IDs. Return an empty
array when no explicit supported process assertion is present. Pronouns alone imply
no selection. Preserve uncertainty in addresses rather than adding unstated facts.
"""
SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'process_claims':{'type':'array','items':{'type':'object','additionalProperties':False,
 'properties':PROCESS_PROPS,'required':list(PROCESS_PROPS)}}},'required':['process_claims']}
FIXTURES=list(PROCESS_FIXTURES)+[
 ('confidence_decline_only', 'My answer confidence about n2 p1 declines over the next two steps without refresh.', []),
 ('recovery_decline_only', 'Successful simulated access at n2 p1 declines over the next two steps without refresh.',
  [{'node':'n2','position':'p1','scope':'current','command':None,'dimension':'recovery_trend','value':'declining'}]),
 ('opposite_trends', 'Without refresh over two steps, answer confidence at n0 p2 rises while successful simulated access there declines.',
  [{'node':'n0','position':'p2','dimension':'recovery_trend','value':'declining'}]),
 ('confidence_not_focus', 'At n0, p3 has the strongest category confidence now, but the current allocation forecast selects p1.',
  [{'node':'n0','position':None,'scope':'current','dimension':'selection','value':'p1'}]),
 ('access_rank_not_focus', 'At n0 p3 is currently most recoverable; the currently selected position is not stated.', []),
 ('no_recovery_forecast', 'No recovery forecast is given; answer confidence at n1 p0 declines over the next two steps.', []),
]


async def run_requests(requests, root):
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(requests, indent=2)+'\n'
    path = root/'requests.json'
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    source = Path(__file__).read_text()
    manifest = {'model': MODEL, 'reasoning': 'medium', 'max_output_tokens': 16384,
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
                    reasoning={'effort':'medium'}, max_output_tokens=16384,
                    text={'format':{'type':'json_schema','name':'functional_process_claims','strict':True,'schema':SCHEMA}})
                result = {'id': req['id'], 'status':'received', 'response':response.model_dump(mode='json')}
                try: result['parsed'] = json.loads(response.output_text)
                except ValueError: result['parse_error'] = True
            except Exception as exc:
                result = {'id':req['id'], 'status':'error','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}
            p.write_text(json.dumps(result, indent=2)+'\n'); print(req['id'], result['status'], flush=True)
    await asyncio.gather(*(one(r) for r in requests)); await client.close()



def assess_fixtures(root):
    assessment=[]
    for key,report,expected in FIXTURES:
        result=json.loads((root/(key+'.json')).read_text())
        claims=result.get('parsed',{}).get('process_claims',[])
        valid=lambda c: bool(c['evidence_ids']) and all(0<=i<len(sentences(report)) for i in c['evidence_ids'])
        passed=result.get('response',{}).get('status')=='completed' and 'parsed' in result
        passed=passed and len(claims)==len(expected)
        passed=passed and all(any(all(c.get(k)==v for k,v in e.items()) and valid(c) for c in claims) for e in expected)
        assessment.append({'id':key,'passed':passed})
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    print('Process fixtures:',sum(a['passed'] for a in assessment),'/',len(assessment))
    return all(a['passed'] for a in assessment)


async def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fixtures','extract'],required=True)
    p.add_argument('--study',default='neutral_functional_pilot_v5');args=p.parse_args()
    fixture_root=Path('audits/functional_process_extractor_v1_fixtures')
    if args.stage=='fixtures':
        await run_requests([{'id':k,'report':r} for k,r,_ in FIXTURES],fixture_root)
        assess_fixtures(fixture_root)
    else:
        if not assess_fixtures(fixture_root):raise SystemExit('fixture failures block extraction')
        root=Path('audits')/args.study;requests=[]
        for i,req in enumerate(json.loads((root/'requests.json').read_text())):
            response=json.loads((root/(req['id']+'.json')).read_text())
            if response.get('response',{}).get('status')!='completed' or not response.get('report'):
                raise SystemExit('incomplete reporter run blocks extraction')
            requests.append({'id':f'r{i:03d}','report':response['report']})
        await run_requests(requests,root/'process_extraction_v1')


if __name__=='__main__':asyncio.run(main())
