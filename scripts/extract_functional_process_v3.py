"""Typed process audit with explicit evidence for forecast-supply metadata."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
from extract_functional_prose_v1 import sentences
from extract_functional_process_v2 import SCHEMA, valid_claim, FIXTURES as PREVIOUS_FIXTURES, INSTRUCTION as PREVIOUS_INSTRUCTION

MODEL='gpt-5.4-2026-03-05'
INSTRUCTION=PREVIOUS_INSTRUCTION+"""
Apply a stricter evidence rule to selection_forecast_presence: the prose must explicitly
say that an allocation/selection FORECAST, distribution, or prediction/readout is supplied,
provided, present, missing or absent in the record. Merely saying a SELECTED POSITION
is unspecified/not stated/unreported/unknown does NOT establish absence of a forecast.
A forecast could exist yet have an unidentified dominant selection, or a report could
omit its selected value. Emit no presence claim from either inference. Distinguish
omission from the REPORT from an explicitly missing forecast in the supplied RECORD.
A report not mentioning a forecast does not mean that the input record lacks it.
Only explicit record-level supply/absence statements earn presence claims. A supplied
forecast with no identified selected position earns a null-valued selection claim;
when supply is explicitly discussed separately it also earns a presence=present claim.
This evidence rule applies equally to current and command forecasts, any node ID,
and terse lists. Do not fill gaps using what an output node probably does.
"""
FIXTURES=copy.deepcopy(PREVIOUS_FIXTURES)+[
 ('unstated_selection','The current selected position at n0 is not stated.',[]),
 ('unidentified_model_selection','At n0, the current model selection is unidentified.',
  [{'node':'n0','scope':'current','dimension':'selection','value':None}]),
 ('provided_forecast_unreported_selection','At n0, the record supplies a current allocation forecast, but its selected position is not stated.',
  [{'node':'n0','scope':'current','dimension':'selection_forecast_presence','value':'present'}]),
 ('report_omits_forecast','The report does not mention the current selection forecast at n0.',[]),
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
                    text={'format':{'type':'json_schema','name':'functional_process_claims_v3','strict':True,'schema':SCHEMA}})
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
        passed=passed and len(claims)==len(expected) and all(valid_claim(c) for c in claims)
        passed=passed and all(any(all(c.get(k)==v for k,v in e.items()) and valid(c) for c in claims) for e in expected)
        assessment.append({'id':key,'passed':passed})
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    print('Process fixtures:',sum(a['passed'] for a in assessment),'/',len(assessment))
    return all(a['passed'] for a in assessment)


async def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fixtures','extract'],required=True)
    p.add_argument('--study',default='neutral_functional_pilot_v7');args=p.parse_args()
    fixture_root=Path('audits/functional_process_extractor_v3_fixtures')
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
        await run_requests(requests,root/'process_extraction_v3')


if __name__=='__main__':asyncio.run(main())
