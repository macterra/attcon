import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
BASE=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_11')
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_11.json';SHA='861514d2ea78d2073085b44a0042c46c17bc60a0205e9640bb087b5067d1d6a2'
c=w.bound_chunk(CHUNK,SHA)
def record(i,component_decisions,evidence_decisions,binding_decisions,concerns,notes):
 r=c['records'][i]; units={p['unit_id']:p for p in r['property_table']}
 result={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
 result.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=concerns,notes=notes)
 for comp,(assessment,reason) in zip(r['components'],component_decisions,strict=True):
  result['component_reviews'].append(dict(component_path=comp['path'],original_status=comp['recorded_status'],assessment=assessment,notes=reason))
 for e,(grounding,reason,extras) in zip(r['evidence'],evidence_decisions,strict=True):
  props=[];ids=[v['unit_id'] for v in e['geometry']['intersections']]+extras
  for uid in dict.fromkeys(ids):
   p=units[uid]
   if p['kind']=='separator':
    props.append(dict(unit_id=uid,value_path=[],value=p['value'],semantic_role='context_only',reason='Literal separator has no source-content meaning.'));continue
   if p['kind']=='origin':
    def leaves(value,path):
     if isinstance(value,dict):
      for k,v in value.items():yield from leaves(v,path+[k])
     else: yield path,value
    for path,value in leaves(p['value'],[]):
     role='processing_relation_operand' if path[-1]=='kind' else 'context_only'
     props.append(dict(unit_id=uid,value_path=path,value=value,semantic_role=role,reason=f"Exact {'/'.join(path)} for {p['context']['panel']}/{p['context']['view']}/{p['context']['command']}/{p['context']['node']}; identifies provenance or represented/record/feedback time, not an asserted access quality. "+('Same node prefix in the full report binds this operand outside the quote to the identified content.' if uid in extras else 'Field is explicitly present in the quoted origin.')))
   else:
    props.append(dict(unit_id=uid,value_path=[],value=p['value'],semantic_role='content_operand' if p['kind']=='role_peaks' else 'context_only',reason=('Complete '+str(p['context'])+' candidate-maximum set; coordinates identify represented content without an organization or access-status predicate. '+('The matching node prefix in the full text binds this content outside the quoted origin.' if uid in extras else 'Exact quoted peak-set operand.') if p['kind']=='role_peaks' else 'Renderer coordinate convention; only the orientation clause is quoted, so the candidate-preface clause supplies context only.')))
  sources={units[uid]['source_resolution']['origin_class']['origin_source_key'] for uid in ids if units[uid]['source_resolution'] is not None}
  meaning='mixed_sources' if len(sources)>1 else 'source_specific' if len(sources)==1 else 'global_convention'
  ev=e['recorded_evidence'];prefix=f"Recorded referent={ev['referent']}, attribute={ev['attribute']}, scope={ev['scope']}, time={ev['time']}. "
  result['evidence_reviews'].append(dict(evidence_path=e['path'],grounding=grounding,source_meaning=meaning,properties=props,notes=prefix+reason))
 for b,(assessment,witnesses,reason) in zip(r['bindings'],binding_decisions,strict=True):
  own=[e['path'] for e in r['evidence'] if e['binding_id']==b['binding_id']]
  result['binding_reviews'].append(dict(binding_path=b['path'],feature_witnesses=[dict(feature=f,evidence_paths=paths) for f,paths in witnesses.items()],binding_evidence_paths=own,assessment=assessment,notes=reason))
 raw=(json.dumps(result,indent=2,ensure_ascii=True)+'\n').encode();draft=BASE/'work'/f'working_{i:02d}.json'
 with draft.open('xb') as f:f.write(raw)
 print(w.save_first_record(c,r['id'],BASE/'work',raw))
def unestablished(i):
 r=c['records'][i];assert not r['reference_established']
 result={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
 result.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics retained without semantic salvage: '+json.dumps(r['reference_measurement']['derivation'],sort_keys=True,separators=(',',':')))
 raw=(json.dumps(result,indent=2)+'\n').encode();draft=BASE/'work'/f'working_{i:02d}.json'
 with draft.open('xb') as f:f.write(raw)
 print(w.save_first_record(c,r['id'],BASE/'work',raw))
