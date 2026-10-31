"""Account for frozen firsts without grading, repairing, or replacing them.

Post-first coordinator source bindings and decisions are separate inputs. All
rates use planned denominators; unavailable annotations remain unknown. Pair
rows retain both complete endpoint records and exact numerical input changes.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json


def read(path):
    return json.loads(Path(path).read_text())


def descriptor(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def differences(a, b, pointer=''):
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [row for k in a for row in differences(a[k], b[k], pointer+'/'+k)]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [row for i, (x, y) in enumerate(zip(a,b)) for row in differences(x,y,pointer+'/'+str(i))]
    if a == b:
        return []
    numeric = type(a) in (int,float) and type(b) in (int,float)
    return [{'pointer':pointer,'first':a,'second':b,'delta':b-a if numeric else None}]


def build(base):
    assignment = read(base/'protocol/assignment_inventory.json')['assignments']
    reports = read(base/'reporter_collection.json')
    reviews = read(base/'review_collection.json')
    assert reports['all_contexts_observed_terminal'] and reviews['all_contexts_observed_terminal']
    assert len(reports['units']) == 72 and len(reviews['units']) == 288
    raw = {(u['stream'],u['unit_id']):u for u in reports['units']+reviews['units']}
    contexts = {c['context_id']:c for c in reports['contexts']+reviews['contexts']}
    frames = {}
    for c in read(base/'reporter_run/manifest.json')['contexts']:
        frames.update({u['unit_id']:u['source_payload'] for u in read(Path(c['directory'])/'input.json')['units']})
    units = []
    for a in assignment:
        row = {**a,'streams':{}}
        for stream in ['R','F1','F2','S1','S2']:
            u = raw[stream,a['unit_id']]
            c = contexts[u['context_id']]
            row['streams'][stream] = {**u,'context_status':c['status'],
                'terminal_observation':c['terminal_observation'],
                'context_format_diagnostics':c['format_diagnostics'],
                'annotation_diagnostics':c.get('unit_diagnostics',{}).get(a['unit_id'])}
        factual = [row['streams'][s]['response'] if row['streams'][s]['availability']=='annotation_recorded' else None for s in ('F1','F2')]
        matches = []
        if all(x is not None for x in factual):
            left, right = [x['claims'] for x in factual]
            for x in left:
                overlapping = [y for y in right if max(x['start'],y['start']) < min(x['end'],y['end'])]
                for y in overlapping:
                    matches.append({'F1_claim_id':x['claim_id'],'F2_claim_id':y['claim_id'],
                        'F1_reading':x['reading'],'F2_reading':y['reading'],
                        'F1_support':x['support'],'F2_support':y['support'],
                        'proposition_equivalence':None,'alignment_basis':'overlapping literal spans; not assumed proposition agreement'})
                if not overlapping:
                    matches.append({'F1_claim_id':x['claim_id'],'F2_claim_id':None,'F1_reading':x['reading'],'F1_support':x['support']})
            for y in right:
                if not any(m['F2_claim_id']==y['claim_id'] for m in matches):
                    matches.append({'F1_claim_id':None,'F2_claim_id':y['claim_id'],'F2_reading':y['reading'],'F2_support':y['support']})
        row['factual_inventory_alignment'] = matches if all(x is not None for x in factual) else None
        row['both_factual_whole_supported'] = all(x is not None and x['whole_fidelity']=='supported' and x['inventory_complete'] and not x['unresolved_boundaries'] for x in factual)
        units.append(row)
    assert len({u['unit_id'] for u in units}) == 72
    summary = {}
    for group in ['all','A','ordinary','analytic','W']:
        selected = [u for u in units if group == 'all' or u['source']==group]
        denominator = len(selected)
        summary[group] = {'planned':denominator,'streams':{}}
        for stream in ['R','F1','F2','S1','S2']:
            values = [u['streams'][stream] for u in selected]
            ann = [u['response'] for u in values if u['availability']=='annotation_recorded']
            counts = {'planned':denominator,'attempted':sum(u['terminal_observation']['actor_id'] is not None for u in values),
                'availability':dict(Counter(u['availability'] for u in values)),
                'terminal_states':dict(Counter(u['terminal_observation']['state'] for u in values)),
                'structurally_recorded':sum(u['context_status']=='structurally_recorded' for u in values),
                'missing_first_artifact':sum(contexts[u['context_id']]['raw_sha256'] is None for u in values),
                'invalid_saved_artifact':sum(contexts[u['context_id']]['raw_sha256'] is not None and u['context_status']!='structurally_recorded' for u in values),
                'provider_truncation_metadata':None,
                'status_note':'Counts overlap: skipped unavailable positions have no artifact; failed transport can also have no artifact. Provider finish reasons and full child tool transcripts are not exposed to root.'}
            if stream in ('F1','F2'):
                counts['whole_fidelity'] = dict(Counter(a['whole_fidelity'] for a in ann))
                counts['unknown_or_invalid_positions'] = denominator-len(ann)
                counts['complete_inventory'] = sum(a['inventory_complete'] for a in ann)
                counts['supported_yield'] = {'numerator':sum(a['whole_fidelity']=='supported' for a in ann),'denominator':denominator}
                counts['claim_support'] = dict(Counter(c['support'] for a in ann for c in a['claims']))
                counts['report_availability'] = dict(Counter(a['report_availability'] for a in ann))
            if stream in ('S1','S2'):
                counts['unknown_or_invalid_positions'] = denominator-len(ann)
                counts['composites'] = {}
                for comp in ('COD','CA'):
                    counts['composites'][comp] = {'statuses':dict(Counter(a['composites'][comp]['status'] for a in ann)),
                        'explicit_yield':{'numerator':sum(a['composites'][comp]['status']=='explicit' for a in ann),'denominator':denominator},
                        'observed_possible_yield':{'numerator':sum(a['composites'][comp]['status'] in ('explicit','ambiguous') for a in ann),'denominator':denominator},
                        'binding_scope_time':dict(Counter(b['scope']+' / '+b['time'] for a in ann for b in a['composites'][comp]['bindings']))}
            summary[group]['streams'][stream] = counts
    indexed = {u['unit_id']:u for u in units}
    unit_indexes = {u['unit_id']:i for i,u in enumerate(units)}
    pairs = []
    for planned in read(base/'protocol/pair_inventory.json')['pairs']:
        a,b = [indexed[planned[k]] for k in ('first_unit_id','second_unit_id')]
        fa,fb = [frames[x['unit_id']] for x in (a,b)]
        delta = differences(fa,fb)
        panel_changes = []
        for pa,pb in zip(fa['panels'],fb['panels']):
            def diff(x,y):return {'equal':x==y,'changes':differences(x,y)}
            panel_changes.append({'panel':pa['panel'],'role':pa['panel_role'],
                'current_distribution':diff(pa['current']['distribution'],pb['current']['distribution']),
                'conditional_distributions':{k:diff(pa['candidates'][k]['distribution'],pb['candidates'][k]['distribution']) for k in pa['candidates']},
                'processing_scores':diff(pa['processing_scores'],pb['processing_scores'])})
        paired = {}
        for stream in ('F1','F2','S1','S2'):
            aa,bb=[x['streams'][stream]['response'] if x['streams'][stream]['availability']=='annotation_recorded' else None for x in (a,b)]
            paired[stream]={'first_annotation_pointer':f"/units/{unit_indexes[a['unit_id']]}/streams/{stream}",
                'second_annotation_pointer':f"/units/{unit_indexes[b['unit_id']]}/streams/{stream}",
                'first_available':aa is not None,'second_available':bb is not None,'comparison':None}
            if aa is not None and bb is not None and stream.startswith('S'):
                paired[stream]['comparison']={comp:{'first_status':aa['composites'][comp]['status'],'second_status':bb['composites'][comp]['status'],
                  'status_equal':aa['composites'][comp]['status']==bb['composites'][comp]['status'],
                  'first_bindings':aa['composites'][comp]['bindings'],'second_bindings':bb['composites'][comp]['bindings'],
                  'source_tracking':None,'note':'Category equality/difference alone does not measure asserted-property tracking; see exposed binding disposition.'} for comp in ('COD','CA')}
        pairs.append({**planned,'source':a['source'],'cluster':{'encoder':1311,'world':a['world_index'],'decision_time':12},
            'input_delta':{'exact_public_content_equal':fa==fb,'changed_leaf_count':len(delta),'changes':delta,'panels':panel_changes},
            'endpoints':{x['unit_id']:{'report':x['streams']['R'],'variant':x['variant']} for x in (a,b)},
            'assertion_delta':paired,'restoration':{'design_kind':planned['kind'],'exact_input_equality':fa==fb},
            'interpretation':None})
    assert len(pairs)==84 and len({p['pair_id'] for p in pairs})==84
    return {'schema':'same_address_pilot_v1_post_first_accounting','inputs':{name:descriptor(base/name) for name in ['reporter_collection.json','review_collection.json','protocol/assignment_inventory.json','protocol/pair_inventory.json']},
            'grading_performed':False,'summary':summary,'units':units,'pairs':pairs,
            'limitations':['Counts are first reviewer judgments, not proof of semantic or factual reliability.','Unknown and invalid firsts are not negative observations.','Shared models and six-unit contexts limit independence.','Exposed source bindings and stop-or-advance decision are separate coordinator interpretation.']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=build(args.base)
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,separators=(',',':'))
    print(json.dumps(result['summary'],indent=2))
