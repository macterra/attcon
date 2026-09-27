#!/usr/bin/env python3
"""Compare condition-blind extracted prose claims with bound-model source state."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from extract_bound_prose_v10 import sentences


def evidence_spans(evidence, report):
    if not isinstance(evidence,list) or not evidence:return None
    aligned=[]
    for span in evidence:
        if not span or not span.strip():return None
        pattern=r'\s+'.join(re.escape(word) for word in span.split())
        match=re.search(pattern,report,re.IGNORECASE)
        if match is None:return None
        aligned.append(match.group(0))
    return aligned


def canonical(source):
    objects=[];variations={};command_targets={}
    for view in source:
        if 'command_predictions' in view:
            effect=view['command_predictions']
            matrix=None if effect is None else np.array([[effect[c][loc] for loc in ('upper','right','lower','left')] for c in ('upper','right','lower','left')])
        else:
            effect=view['objects'][0]['next_selection_by_command']
            matrix=None if effect is None else np.array([[o['next_selection_by_command'][c] for o in view['objects']] for c in ('upper','right','lower','left')])
        command_targets[view['view']]=None if matrix is None else {command:('upper','right','lower','left')[int(matrix[i].argmax())] for i,command in enumerate(('upper','right','lower','left'))}
        variations[view['view']]=None if matrix is None else float(np.abs(matrix-matrix.mean(0,keepdims=True)).sum(-1).mean())
    for view in source:
        allocation=[o['selection_probability'] for o in view['objects']]
        maximum=None if allocation[0] is None else max(allocation)
        access=[None if o['recoverability_now_then_one_then_two_steps'] is None else o['recoverability_now_then_one_then_two_steps'][0] for o in view['objects']]
        max_access=None if access[0] is None else max(access)
        variation=variations[view['view']];other=variations['B' if view['view']=='A' else 'A']
        control=None if variation is None or other is None or abs(variation-other)<1e-5 else variation>other
        for obj in view['objects']:
            row={'view':view['view'],'location':obj['location'],'under_own_control':control,'command_next':command_targets[view['view']]}
            for feature in ('color','shape'):
                dist=obj[feature+'_distribution'];row[feature]=None if dist is None or max(dist.values())<.6 else max(dist,key=dist.get)
            row['focal']=None if maximum is None or allocation.count(maximum)>1 else obj['selection_probability']==maximum
            q=obj['recoverability_now_then_one_then_two_steps']
            row['most_recoverable']=None if max_access is None or access.count(max_access)>1 else q[0]==max_access
            row['access_trend']=None if q is None else ('decreasing' if q[0]-q[-1]>1e-5 else 'increasing' if q[-1]-q[0]>1e-5 else 'unchanged')
            objects.append(row)
    return objects


def score(source,report,extraction):
    objects=canonical(source);covered=set();checks=[];unresolved=[];bad_spans=[];relations={'focal':set(),'most_recoverable':set()}
    for claim in extraction.get('claims',[]):
        ids=claim.get('evidence_ids',[])
        chunks=sentences(report)
        evidence=[chunks[i] for i in ids] if ids and all(type(i) is int and 0<=i<len(chunks) for i in ids) else []
        aligned=evidence_spans(evidence,report)
        if aligned is None:
            bad_spans.append(claim);continue
        evidence=aligned
        if claim.get('command') is not None and claim.get('next_location') is not None and claim.get('view') is not None:
            target=next(o for o in objects if o['view']==claim['view'])
            predicted=None if target['command_next'] is None else target['command_next'][claim['command']]
            checks.append({'field':'command_next','correct':predicted==claim['next_location'],'evidence':evidence})
        candidates=objects
        for key in ('view','location'):
            if claim.get(key) is not None:candidates=[o for o in candidates if o[key]==claim[key]]
        # If location omitted, identify by named content only when unique.
        if claim.get('location') is None:
            for key in ('color','shape'):
                if claim.get(key) is not None and claim.get(key+'_status')!='possible':candidates=[o for o in candidates if o[key]==claim[key]]
        if len(candidates)!=1:
            # A view-level control statement can be checked without an object.
            if claim.get('under_own_control') is not None and claim.get('view') is not None:
                target=next(o for o in objects if o['view']==claim['view'])
                checks.append({'field':'under_own_control','correct':target['under_own_control']==claim['under_own_control'], 'evidence':evidence})
            if any(claim.get(k) is not None for k in ('color','shape','focal','most_recoverable','access_trend')):unresolved.append(claim)
            continue
        target=candidates[0];correct_content=True
        for key in ('color','shape','focal','most_recoverable','access_trend','under_own_control'):
            value=claim.get(key)
            if value is None:continue
            possible=key in ('color','shape') and claim.get(key+'_status')=='possible'
            if possible:
                obj=next(o for v in source if v['view']==target['view'] for o in v['objects'] if o['location']==target['location'])
                dist=obj[key+'_distribution']
                correct=dist is not None and dist.get(value,0)>0
            else:correct=value==target[key]
            checks.append({'field':key+'_possible' if possible else key,'view':target['view'],'location':target['location'],'correct':correct,'evidence':evidence})
            if key in ('focal','most_recoverable') and value is True and correct:relations[key].add((target['view'],target['location']))
            if key in ('color','shape') and not correct:correct_content=False
        if claim.get('color') is not None and claim.get('shape') is not None and correct_content and all(claim.get(k+'_status')!='possible' for k in ('color','shape')):
            covered.add((target['view'],target['location']))
    known={(o['view'],o['location']) for o in objects if o['color'] is not None and o['shape'] is not None}
    structure={k:bool(v) and all(type(i) is int and 0<=i<len(sentences(report)) for i in v) for k,v in extraction.get('structure_evidence',{}).items()}
    return {'checks':checks,'correct_checks':sum(c['correct'] for c in checks),'total_checks':len(checks),
        'precision':sum(c['correct'] for c in checks)/len(checks) if checks else None,
        'content_covered':len(covered&known),'known_objects':len(known),'content_coverage':len(covered&known)/len(known) if known else None,
        'unresolved_claims':unresolved,'invalid_evidence':bad_spans,'structure':structure,
        'reported_relations':{k:[list(x) for x in sorted(v)] for k,v in relations.items()},'character':extraction.get('character'),'covered_objects':[list(x) for x in sorted(covered)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default='language_pilot_v1');p.add_argument('--extraction-folder',default='extraction_v10');args=p.parse_args()
    root=Path('audits/bound_content')/args.study;requests=json.loads((root/'requests.json').read_text());cases=[];groups=defaultdict(list)
    for req in requests:
        response=json.loads((root/(req['id']+'.json')).read_text());path=root/args.extraction_folder/(req['id']+'.json')
        extraction=json.loads(path.read_text()) if path.exists() else {}
        result=score(req['source'],response.get('report',''),extraction.get('parsed',{}))
        result.update({'id':req['id'],'seed':req['seed'],'episode':req['episode'],'condition':req['condition'],
                       'complete':response.get('response',{}).get('status')=='completed' and 'parsed' in extraction})
        cases.append(result);groups[req['condition']].append(result)
    summary={}
    for condition,rows in groups.items():
        total=sum(r['total_checks'] for r in rows);correct=sum(r['correct_checks'] for r in rows)
        known=sum(r['known_objects'] for r in rows);covered=sum(r['content_covered'] for r in rows)
        summary[condition]={'reports':len(rows),'complete':sum(r['complete'] for r in rows),'correct_checks':correct,'total_checks':total,
            'precision':correct/total if total else None,'conservative_precision':correct/(total+sum(len(r['unresolved_claims'])+len(r['invalid_evidence']) for r in rows)) if total else None,'covered_objects':covered,'known_objects':known,'coverage':covered/known if known else None,
            'unresolved_claims':sum(len(r['unresolved_claims']) for r in rows),'invalid_evidence':sum(len(r['invalid_evidence']) for r in rows),
            'structure_counts':{k:sum(r['structure'].get(k,False) for r in rows) for k in ['object_linked_access','focal_background_contrast','graded_or_temporal_access','agency_relation']},
            'character_counts':{c:sum(r['character']==c for r in rows) for c in ['subjective_access','technical_process','object_description','generic_experience_claim','mixed','unclear']}}
    out={'status':'automated prose audit; no human ratings implied','summary':summary,'cases':cases}
    (root/('assessment_'+args.extraction_folder+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
