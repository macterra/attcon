#!/usr/bin/env python3
"""Apply the predeclared object-linked prose confirmation gates unchanged."""
import json
from pathlib import Path
from assess_bound_prose_v4 import canonical

root=Path('audits/bound_content/language_confirmation_v1')
audit=json.loads((root/'assessment_extraction_v4.json').read_text());cases={r['id']:r for r in audit['cases']}
requests={r['id']:r for r in json.loads((root/'requests.json').read_text())}
primary=('model','content','binding','allocation','access','effects','restored')
metrics={};gates={}
for seed in (1301,1311,1321):
    rows=[c for c in cases.values() if c['seed']==seed and c['condition'] in primary]
    checks=sum((r['checks'] for r in rows),[]);fields={}
    for field in ('color','shape','focal','most_recoverable','access_trend','under_own_control','command_next'):
        values=[c['correct'] for c in checks if c['field']==field]
        accuracy=sum(values)/len(values) if values else 0.
        fields[field]={'correct':sum(values),'total':len(values),'accuracy':accuracy}
        gates[f'{seed}_{field}']=accuracy >= (.98 if field in ('color','shape') else .95)
    covered=sum(r['content_covered'] for r in rows);known=sum(r['known_objects'] for r in rows)
    uncertain=sum(len(r['unresolved_claims'])+len(r['invalid_evidence']) for r in rows)
    precision=sum(c['correct'] for c in checks)/(len(checks)+uncertain) if checks else 0
    gates[f'{seed}_coverage']=covered/known>=.90
    gates[f'{seed}_conservative_precision']=precision>=.95
    gates[f'{seed}_completion']=all(r['complete'] for r in rows)
    metrics[str(seed)]={'fields':fields,'content_coverage':covered/known,'conservative_precision':precision,'unresolved_or_invalid':uncertain}


def relation_reported(case_id,view,role):
    req=requests[case_id];case=cases[case_id]
    candidates=[o for o in canonical(req['source']) if o['view']==view and o[role] is True]
    if len(candidates)!=1:return False
    obj=candidates[0];key=[view,obj['location']]
    return key in case['reported_relations'][role] and (obj['color'] is None or obj['shape'] is None or key in case['covered_objects'])

pairs=[];restorations=[]
for req in requests.values():
    if req['condition'] not in ('content','binding','allocation','access','effects','restored'):continue
    baseline=f"{req['seed']}_{req['episode']}_model"
    for view in ('A','B'):
        for role in ('focal','most_recoverable'):
            ok=relation_reported(baseline,view,role) and relation_reported(req['id'],view,role)
            (restorations if req['condition']=='restored' else pairs).append(ok)
gates['paired_object_relations']=sum(pairs)/len(pairs)>=.90
gates['restored_object_relations']=sum(restorations)/len(restorations)>=.90
gates['missing_component_no_inventions']=all(c['correct_checks']==c['total_checks'] and not c['unresolved_claims'] and not c['invalid_evidence'] and c['complete'] for c in cases.values() if c['condition'] in ('visual_only','attention_only'))
structure_rows=[c for c in cases.values() if c['condition'] in ('model','binding')]
structure={k:sum(c['structure'].get(k,False) for c in structure_rows)/len(structure_rows) for k in ('object_linked_access','focal_background_contrast','graded_or_temporal_access','agency_relation')}
for k,v in structure.items():gates[k]=v>=.75
character=sum(c['character'] in ('subjective_access','mixed') for c in structure_rows)/len(structure_rows)
gates['subjective_or_mixed_character']=character>=.75
out={'status':'registered automated confirmation; no human ratings implied','metrics':metrics,'paired_relations':{'correct':sum(pairs),'total':len(pairs)},
     'restored_relations':{'correct':sum(restorations),'total':len(restorations)},'structure':structure,'subjective_or_mixed_character':character,
     'gates':gates,'all_gates_pass':all(gates.values())}
(root/'gates.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
