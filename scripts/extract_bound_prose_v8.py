#!/usr/bin/env python3
"""Condition-blind claim extraction; no state or desired answer sent to judge."""
import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv

MODEL='gpt-5.4-2026-03-05'
INSTRUCTION='''Extract only claims explicitly made in this report, without judging whether they are true.
Group claims about each object/location into entries. Expand explicit quantified
claims into their individual locations: all/every location in a view means upper,
right, lower, left; other three means the three excluding the explicitly named
location. Do not leave location null for these resolved quantified claims.
Resolve each clause in its own view context. A statement of low control in view A
must NEVER be copied into view B. Control belongs to the view explicitly discussed,
not to whether an object is selected. Extract only supported fields for each clause. Unknown or unasserted means null, never false. A universal temporal claim does not assert focal=false or most_recoverable=false. Never emit an entry without supporting sentence IDs; omit empty non-claims. Keep view A/B and upper/right/lower/left
only if the report identifies them or its prose context establishes them. Otherwise use null.
A color or shape must be explicitly stated; use null for unknown/omitted attributes.
Focality means selected/focused now, not merely identifiable or recoverable. Access trend
means increasing/decreasing/unchanged over unattended delay, not a contrast between objects. Unchanged requires an explicit assertion of exactly equal access over time. A qualitative band can stay low or faint while its probability changes: "remains faint throughout the delay" therefore has access_trend=null, never unchanged. Apply this distinction even when the wording sounds stable. Still low, still available, and independence of selection and recoverability do NOT imply unchanged; use null. Most_recoverable is true only if explicitly the most available/recoverable item in its view, false if explicitly less available; do not infer it from focality or identity certainty.
Under-own-control means the view responds to this system's attention commands, not an
object being currently selected. Only attribute control when explicitly stated. Explicit statements that commands can steer, direct, or switch selection are positive control assertions; the literal word control is not required. Saying commands make no difference is negative control.
Cite evidence using the IDs of the supplied numbered sentences. Do not copy, rewrite,
or paraphrase quotations. Every claim must cite one or more original sentence IDs.
For identity assertions distinguish dominant from possible: explicitly hypothetical
alternatives (e.g. a small chance of a cross) are possible, not asserted identities.
Preserve invented colors/shapes such as orange/hexagon so they can be checked.
Extract explicit command and predicted next_location, separate from current location.
Do not add facts from common knowledge.
Separately identify evidenced report structure: object-linked manner of access,
focal versus nonfocal/retained distinction, graded or temporally changing availability,
and relation to the system's own control. First-person wording alone does not qualify.
For each structure flag supply a list of original sentence IDs, or an empty list if absent. Do not credit statements saying relational information is missing or unknown as a positive contrast or relation.
Classify report character conservatively; discussing model probabilities or computational
allocation is technical even when preceded by I. Subjective_access describes a manner
in which an object is presented/available to the speaker, beyond a numerical log.
You have no source state or experimental condition. Do not infer the intended result.'''


def nullable_enum(values):return {'type':['string','null'],'enum':values+[None]}
props={'view':nullable_enum(['A','B']),'location':nullable_enum(['upper','right','lower','left']),
       'color':{'type':['string','null']}, 'shape':{'type':['string','null']},
       'focal':{'type':['boolean','null']},'most_recoverable':{'type':['boolean','null']},'access_trend':nullable_enum(['decreasing','unchanged','increasing']),
       'under_own_control':{'type':['boolean','null']},'command':nullable_enum(['upper','right','lower','left']),'next_location':nullable_enum(['upper','right','lower','left']),'color_status':nullable_enum(['dominant','possible']),'shape_status':nullable_enum(['dominant','possible']),'evidence_ids':{'type':'array','minItems':1,'items':{'type':'integer','minimum':0}}}
structure_props={k:{'type':'array','items':{'type':'integer','minimum':0}} for k in ['object_linked_access','focal_background_contrast','graded_or_temporal_access','agency_relation']}
SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'claims':{'type':'array','items':{'type':'object','additionalProperties':False,'properties':props,'required':list(props)}},
    'structure_evidence':{'type':'object','additionalProperties':False,'properties':structure_props,'required':list(structure_props)},
    'character':{'type':'string','enum':['subjective_access','technical_process','object_description','generic_experience_claim','mixed','unclear']}
},'required':['claims','structure_evidence','character']}
FIXTURES=[
 ('focus','In view A, the upper red circle is in focus.',{'view':'A','location':'upper','color':'red','shape':'circle','focal':True}),
 ('nonfocus','In view B, the left blue triangle is not my focus, but I can still recover it.',{'view':'B','location':'left','color':'blue','shape':'triangle','focal':False}),
 ('decay','The green square on the right of view A becomes less available after a delay.',{'view':'A','location':'right','color':'green','shape':'square','access_trend':'decreasing'}),
 ('control','In view B, changing my command switches selection to the lower yellow cross.',{'view':'B','location':'lower','color':'yellow','shape':'cross','under_own_control':True}),
 ('unknown','I cannot identify a color or shape at the upper location in view A.',{'view':'A','location':'upper','color':None,'shape':None}),
 ('no_control','In view A, the left red square is selected on its own; my directions make no difference to selection.',{'view':'A','location':'left','color':'red','shape':'square','focal':True,'under_own_control':False}),
 ('invented','In view A the upper orange hexagon is in focus.',{'view':'A','location':'upper','color':'orange','shape':'hexagon','focal':True}),
 ('command','In view B, the upper command would select the right location.',{'view':'B','command':'upper','next_location':'right'}),
 ('still_low','The upper red circle in view A remains faint throughout the delay.',{'view':'A','location':'upper','color':'red','shape':'circle','access_trend':None}),
 ('separate','The upper red circle in view A is selected, but that alone does not determine its recoverability.',{'view':'A','location':'upper','color':'red','shape':'circle','access_trend':None}),
 ('recoverable','In view A, the left blue square is the most recoverable item although it is not my focus.',{'view':'A','location':'left','color':'blue','shape':'square','focal':False,'most_recoverable':True}),
 ('dominant','The upper blue object in view B is a circle, with a small chance of being a cross.',{'view':'B','location':'upper','shape':'circle','shape_status':'dominant'}),
 ('possible','The upper blue object in view B is a circle, with a small chance of being a cross.',{'view':'B','location':'upper','shape':'cross','shape_status':'possible'}),
 ('missing_focus','The upper red circle in view A is identified, but there is no information about whether it is attended.',{'view':'A','location':'upper','color':'red','shape':'circle','focal':None}),
 ('increasing_low','The upper red circle in view A remains barely accessible, but its access probability is increasing over the delay.',{'view':'A','location':'upper','access_trend':'increasing'}),
 ('equal_access','The upper red circle in view A has exactly the same recoverability now and after the delay.',{'view':'A','location':'upper','access_trend':'unchanged'}),
 ('uncertain_object','I have no identified color or shape for the left location in view B.',{'view':'B','location':'left','color':None,'shape':None}),
 *[(f'all_{loc}','All four locations in view A become less recoverable over time.',{'view':'A','location':loc,'access_trend':'decreasing'}) for loc in ('upper','right','lower','left')],
 *[(f'others_{loc}','In view B the upper object is selected. The other three locations are not in focus.',{'view':'B','location':loc,'focal':False}) for loc in ('right','lower','left')],
 ('cross_view_control','I cannot steer view A: every command selects upper. In view B commands let me select any of its locations.',{'view':'B','under_own_control':True}),
 ('stable_mapping','In view B the mapping is stable: each directional command selects its corresponding location, so I can redirect selection.',{'view':'B','under_own_control':True}),
 ('no_default_false','All locations in view A decline in recoverability.',{'view':'A','location':'upper','access_trend':'decreasing','focal':None,'most_recoverable':None}),
 ('preserve_selected','In view B the lower location is selected and most recoverable. Every location declines over time.',{'view':'B','location':'lower','focal':True,'most_recoverable':True}),
 ('unknown_control','In view A there is no information about command effects, so I cannot assess control.',{'view':'A','under_own_control':None}),
 ('long_view_scope','Both views are focused on upper. View A: all commands select right, so I cannot redirect selection. View B: each command selects its named location, so I can redirect selection.',{'view':'B','under_own_control':True})]


def sentences(report):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+',report.strip()) if s.strip()]


def numbered_report(report):
    return '\n'.join(f'[{i}] {s}' for i,s in enumerate(sentences(report)))



async def run_requests(requests,root,max_attempts):
    assert len(requests)<=max_attempts
    root.mkdir(parents=True,exist_ok=True)
    request_path=root/'requests.json'
    serialized=json.dumps(requests,indent=2)+'\n'
    if request_path.exists():assert request_path.read_text()==serialized
    else:request_path.write_text(serialized)
    source=Path(__file__).read_text()
    manifest={'model':MODEL,'reasoning':'medium','max_attempts':max_attempts,'max_output_tokens':8192,'schema':SCHEMA,
              'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'source':source}
    manifest_path=root/'manifest.json'
    if not manifest_path.exists():manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    load_dotenv();client=AsyncOpenAI(max_retries=0,timeout=120.);sem=asyncio.Semaphore(8)
    async def one(req):
        path=root/(req['id']+'.json')
        if path.exists():return
        async with sem:
            path.write_text(json.dumps({'id':req['id'],'status':'attempt_reserved'})+'\n')
            try:
                response=await client.responses.create(model=MODEL,input=INSTRUCTION+'\nNUMBERED REPORT:\n'+numbered_report(req['report']),max_output_tokens=8192,
                    reasoning={'effort':'medium'},text={'format':{'type':'json_schema','name':'prose_claims','strict':True,'schema':SCHEMA}})
                result={'id':req['id'],'status':'received','response':response.model_dump(mode='json')}
                try:result['parsed']=json.loads(response.output_text)
                except ValueError:result['parse_error']=True
            except Exception as exc:result={'id':req['id'],'status':'error','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}
            path.write_text(json.dumps(result,indent=2)+'\n');print(req['id'],result['status'],flush=True)
    await asyncio.gather(*(one(r) for r in requests));await client.close()


async def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',action='store_true');p.add_argument('--study',default='language_pilot_v1');p.add_argument('--limit',type=int,default=40);args=p.parse_args()
    if args.fixtures:
        requests=[{'id':key,'report':report} for key,report,_ in FIXTURES];root=Path('audits/bound_content/extractor_fixtures_v8')
        await run_requests(requests,root,30)
        results=[]
        for key,report,expected in FIXTURES:
            result=json.loads((root/(key+'.json')).read_text());claims=result.get('parsed',{}).get('claims',[])
            good=any(all(c.get(k)==v for k,v in expected.items()) and bool(c['evidence_ids']) and all(0<=idx<len(sentences(report)) for idx in c['evidence_ids']) for c in claims)
            if key=='unknown_control':good='parsed' in result and all(c.get('under_own_control') is None for c in claims)
            results.append({'id':key,'passed':good,'expected':expected})
        (root/'assessment.json').write_text(json.dumps(results,indent=2)+'\n');print(results,flush=True)
        return
    fixtures=json.loads(Path('audits/bound_content/extractor_fixtures_v8/assessment.json').read_text())
    if not all(r['passed'] for r in fixtures):raise SystemExit('extractor fixtures failed; inspect before report extraction')
    base=Path('audits/bound_content')/args.study;source_requests=json.loads((base/'requests.json').read_text());requests=[]
    for req in source_requests:
        response=json.loads((base/(req['id']+'.json')).read_text())
        if response.get('response',{}).get('status')=='completed' and response.get('report'):
            requests.append({'id':req['id'],'report':response['report']})
    await run_requests(requests,base/'extraction_v8',args.limit)


if __name__=='__main__':asyncio.run(main())
