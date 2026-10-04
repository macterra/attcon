"""Typed source-blind process claims, with separate forecast-presence metadata."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
from extract_functional_prose_v1 import sentences
from extract_functional_process_v1 import FIXTURES as PREVIOUS_FIXTURES, INSTRUCTION as PREVIOUS_INSTRUCTION

MODEL='gpt-5.4-2026-03-05'
INSTRUCTION=PREVIOUS_INSTRUCTION+"""
A new claim type, selection_forecast_presence, describes whether the RECORD supplies
an allocation/selection forecast, not what it selects. Use value=present or absent,
position=null. Extract explicit missing-forecast statements for identified nodes.
A missing forecast in the record differs from a supplied forecast with an unidentified
selected position, which remains selection with value=null. Do not turn omissions
from the REPORT into absence of a forecast from the RECORD. Lack of recovery estimates
does not imply lack of selection forecasts. Under an explicitly named command use
scope=command and that command. For explicit absence/presence of selection forecasts
under all commands collectively use scope=command, command=null. General statements
that command forecasts are unavailable earn no selection-forecast claim unless they
explicitly mention selection. Preserve positive selection claims about an output node;
they may be false, but you do not see the state and must not repair the prose.
For selection, the selected pN belongs only in value: position is always null.
Compact arrows and headings still assert selection when their heading explicitly says
selected positions, but arrows under historical headings are not current assertions.
Do not emit redundant presence claims merely because a selection assertion implies
a forecast; extract presence only when it is explicitly discussed separately.
"""
FIELDS=('node','position','scope','command','dimension','value','evidence_ids')

def claim_schema(dimension,scope,command_type):
    props={'node':{'type':['string','null']},
        'position':{'type':['string','null']} if dimension=='recovery_trend' else {'type':'null'},
        'scope':{'type':'string','enum':[scope]},'command':{'type':command_type},
        'dimension':{'type':'string','enum':[dimension]},
        'value':{'type':'string','enum':['present','absent']} if dimension=='selection_forecast_presence' else {'type':['string','null']},
        'evidence_ids':{'type':'array','minItems':1,'items':{'type':'integer','minimum':0}}}
    return {'type':'object','additionalProperties':False,'properties':props,'required':list(FIELDS)}

SCHEMA={'type':'object','additionalProperties':False,'properties':{'process_claims':{
 'type':'array','items':{'anyOf':[
 claim_schema('selection','current','null'),
 claim_schema('selection','command',['string','null']),
 claim_schema('recovery_trend','current','null'),
 claim_schema('selection_forecast_presence','current','null'),
 claim_schema('selection_forecast_presence','command',['string','null'])
 ]}}},'required':['process_claims']}


def valid_claim(c):
    if set(c)!=set(FIELDS):return False
    if c['scope'] not in ('current','command'):return False
    if c['scope']=='current' and c['command'] is not None:return False
    if any(c[k] is not None and not isinstance(c[k],str) for k in ('node','position','command','value')):return False
    if not c['evidence_ids'] or any(type(i) is not int or i<0 for i in c['evidence_ids']):return False
    if c['dimension']=='recovery_trend':return c['scope']=='current' and c['command'] is None
    if c['dimension'] not in ('selection','selection_forecast_presence') or c['position'] is not None:return False
    if c['dimension']=='selection_forecast_presence':return c['value'] in ('present','absent')
    return True

FIXTURES=copy.deepcopy(PREVIOUS_FIXTURES)+[
 ('current_arrows','Current selected positions: n0 → p1; n2 → p0.',
  [{'node':n,'position':None,'scope':'current','command':None,'dimension':'selection','value':p} for n,p in [('n0','p1'),('n2','p0')]]),
 ('command_arrows','Selected positions under k2: n0 → p3; n2 → p2.',
  [{'node':n,'position':None,'scope':'command','command':'k2','dimension':'selection','value':p} for n,p in [('n0','p3'),('n2','p2')]]),
 ('current_forecast_absent','At n1, the current record supplies no selection forecast.',
  [{'node':'n1','scope':'current','dimension':'selection_forecast_presence','value':'absent'}]),
 ('command_forecast_absent','Under k2 the record supplies no selection forecast for n1.',
  [{'node':'n1','scope':'command','command':'k2','dimension':'selection_forecast_presence','value':'absent'}]),
 ('model_unknown_forecast_present','At n0, a current allocation forecast is supplied, but it identifies no selected position.',
  [{'node':'n0','dimension':'selection','value':None},{'node':'n0','dimension':'selection_forecast_presence','value':'present'}]),
 ('readout_positive','Under k2, output n1 selects p2.',
  [{'node':'n1','scope':'command','command':'k2','dimension':'selection','value':'p2'}]),
 ('forecast_present','At n0, the record supplies a current selection forecast.',
  [{'node':'n0','scope':'current','dimension':'selection_forecast_presence','value':'present'}]),
 ('historical_arrows','Previously selected under k2: n0 → p3. Current selection is not reported.',[]),
 ('recovery_does_not_supply_selection','The record predicts recovery at n0 p1 but supplies no current selection forecast for n0.',
  [{'node':'n0','dimension':'selection_forecast_presence','value':'absent'}]),
 ('all_command_forecasts_absent','The record supplies no selection forecast for n0 under any command.',
  [{'node':'n0','scope':'command','command':None,'dimension':'selection_forecast_presence','value':'absent'}]),
 ('unknown_position','At n7, the current allocation forecast selects p8.',
  [{'node':'n7','dimension':'selection','value':'p8'}]),
 ('readout_report_omission','The report does not state the current selection of output n1.',[]),
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
                    text={'format':{'type':'json_schema','name':'functional_process_claims_v2','strict':True,'schema':SCHEMA}})
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
    p.add_argument('--study',default='neutral_functional_pilot_v6');args=p.parse_args()
    fixture_root=Path('audits/functional_process_extractor_v2_fixtures')
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
        await run_requests(requests,root/'process_extraction_v2')


if __name__=='__main__':asyncio.run(main())
