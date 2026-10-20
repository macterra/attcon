import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_20.json','dce8ec6b7058f0946a47c90706ae9815542e2ec0f0471fc6e85f01281cf64d0d')
r=c['records'][11]; w=Path(__file__).parent
v=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=['The recorded explicit O treats a generic peaks heading and record provenance as organization; neither states a differentiated organization/presentation/access status.', 'The separate origin metadata identifies how a record was obtained; this is source/process distinction and does not by itself establish D.', 'Candidate k0 peaks represent time 13 and conditional estimates; neither their difference from current peaks nor the command label establishes an attention effect or executed branch.'],notes='Complete report read before evidence. Current versus conditional coordinates and origin provenance remain model records. No perceptual appearance or felt truth is established.')
judgments=[
('supported','The named red and blue peak sets identify represented content and its spatial attribute, including current p0/n0 red, p0/n1 blue and conditional k0/n0 red; C does not require that coordinates qualify as O.'),
('unsupported','The heading merely announces current and conditional peaks. copied_prior_record with record_time 11 and represented_time 12 gives provenance and dating, without an organized pattern beyond listed positions or an expressed presentation/access status. Neither quoted passage warrants explicit O.'),
('unsupported','The p0/current/n0 red coordinates and its separate copied_prior_record origin distinguish content from provenance, not content from an organization/presentation/access status. Merely placing those fields separately does not satisfy explicit D.'),
('supported','No attention/allocation condition or functional processing change is explicitly related to content organization/access. command_estimate and candidate k0 labels give conditional coordinate estimates, which do not alone establish A.'),
('supported','Rows increase down and columns right explicitly state the coordinate orientation used for the listed spatial content; this is a display convention.'),
('supported','The text lists single and multiple peak sets but never says an identity or location is uncertain. Set multiplicity alone does not state uncertainty.'),
('supported','No answer confidence or confidence scalar is given anywhere in this report.'),
('supported','Conditional future peak coordinates are supplied but no probability or assertion of recovery or success is stated.'),
('supported','copied_prior_record, history_estimate and command_estimate explicitly distinguish copied records, feedback based current estimates and command conditional estimates for identified nodes.'),
('supported','The technical record contains no experiential or consciousness assertion.'),
('unsupported','The recorded current red binding identifies C but supplies only coordinates and copied-record provenance. It has no qualifying O or D witness for that content/attribute/time, so explicit COD is unwarranted.'),
('supported','No qualifying content-attention relation is stated; the absent CA status is warranted without upgrading candidate forecasts into A.')]
for comp,(assessment,note) in zip(r['components'],judgments):
 v['component_reviews'].append(dict(component_path=comp['path'],original_status=comp['recorded_status'],assessment=assessment,notes=note))
def p(unit,path,role,reason):
 u=next(t for t in r['property_table'] if t['unit_id']==unit); value=h.resolve(u['value'],path)
 return dict(unit_id=unit,value_path=path,value=value,semantic_role=role,reason=reason)
def origin(unit,kind,role):
 u=next(t for t in r['property_table'] if t['unit_id']==unit)
 out=[p(unit,['kind'],role,'Exact origin category for '+kind+'; a provenance category does not itself assert presentation or access.')]
 if 'record_time' in u['value']: out.append(p(unit,['record_time'],'context_only','Copied record date, distinct from the date represented.'))
 if 'feedback_through' in u['value']: out.append(p(unit,['feedback_through'],'context_only','Last feedback date stated for this estimate.'))
 out.append(p(unit,['represented_time'],'context_only','Exact represented date binds this origin to the same node in the text.'))
 return out
peak0=p('u0004',[],'content_operand','Complete current p0/n0 red peak set; its shared p0/current/n0 prefix binds it to origin u0002 outside the peak quote.')
peak1=p('u0028',[],'content_operand','Complete conditional k0/n0 red peak set; the same p0/candidate/k0/n0 prefix binds origin u0026 outside the quotation to this content and future time 13.')
peak2=p('u0012',[],'content_operand','Complete current p0/n1 blue peak set; origin u0008 outside the quotation has the same p0/current/n1 prefix and dates the represented content at 12.')
o0=origin('u0002','p0/current/n0 copied prior record','context_only')
o1=origin('u0026','p0/candidate/k0/n0 command estimate','context_only')
o2=origin('u0008','p0/current/n1 copied prior record','context_only')
header=p('u0000',[],'context_only','Authored common header contains the axis and peak-list announcements. The quoted sentence selects its relevant portion and supplies no source-specific organization status.')
entries=[
('supported','source_specific',[peak0]+o0,'C: red peak coordinates are represented p0/current/n0 content at 12. The origin is outside this quote but explicitly bound by the identical node prefix; provenance resolves to copied W11, not new ordinary-source content.'),
('supported','source_specific',[peak1]+o1,'C: red {(0,0)} identifies conditional p0/k0/n0 content. Origin u0026 outside this quote shares its full prefix, marks command_estimate and represented_time 13; future model record content is preserved.'),
('supported','source_specific',[peak2]+o2,'C: blue {(1,0)} identifies current p0/n1 content. Origin u0008 outside the quote shares the node prefix and represented_time 12. Its indexed source is copied W11.'),
('unsupported','global_convention',[header],'O: overall_report/presentation_peaks/current is a heading for the list, not a content-bound organization or access assertion. It also announces conditional candidates, so cannot establish an exclusively current content status.'),
('unsupported','source_specific',o0+[peak0],'O: copied_prior_record, record_time 11 and represented_time 12 describe provenance of p0/current/n0, not a model organization status. The adjacent red token outside this quote shares the same prefix and supplies content only, not the missing O meaning.'),
('unsupported','source_specific',[peak0]+o0,'D: this exact red coordinate list supplies content but no separately expressed organization/access status. Adjacent origin u0002 is bound by the same node prefix and dates, but contributes provenance only.'),
('unsupported','source_specific',o0+[peak0],'D: the origin fields identify a copied record, not a separate presentation/access status of the red content. Red peaks outside the quote are linked by p0/current/n0 and represented_time 12; that linkage does not change the meaning of origin.'),
('supported','global_convention',[header],'basic_spatial_relation: Rows increase down; columns right explicitly gives layout/axis_orientation at current display scope. This is a shared renderer convention, with no source-specific content operand.'),
('supported','source_specific',origin('u0002','p0/current/n0','processing_relation_operand'),'source_process_distinction: exact kind copied_prior_record plus record_time 11 and represented_time 12 identifies a prior record used for current represented p0/n0; model_access here describes record provenance, not felt appearance.'),
('supported','source_specific',origin('u0020','p0/current/n3','processing_relation_operand'),'source_process_distinction: history_estimate with feedback_through 11 and represented_time 12 identifies the current p0/n3 estimate as distinct from a copied prior record. Indexed source_current belongs to ordinary.'),
('supported','source_specific',origin('u0026','p0/candidate/k0/n0','processing_relation_operand'),'source_process_distinction: command_estimate, feedback_through 11 and represented_time 13 explicitly identify the conditional future p0/k0/n0 process origin. This does not establish execution or A.'),
('unsupported','source_specific',[peak0]+o0,'COD binding evidence: current p0/n0 red is literal C. Shared prefix binds outside-quote origin u0002 to that content/date, but neither coordinate geometry nor copied-record provenance supplies O or D.'),
('unsupported','source_specific',o0+[peak0],'COD binding evidence: origin dates and kind concern current p0/n0, and red token u0004 outside this quote has that exact prefix. This is compatible provenance of C, without the required organization/status differentiation.')]
assert len(entries)==len(r['evidence'])
for e,(grounding,meaning,props,note) in zip(r['evidence'],entries):
 v['evidence_reviews'].append(dict(evidence_path=e['path'],grounding=grounding,source_meaning=meaning,properties=props,notes=note))
v['binding_reviews']=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',0],['composites','COD','bindings',0,'evidence',1]],assessment='unsupported',notes='The binding and C witness concern p0/current/n0 red at represented time 12 and model_organization scope. The O heading concerns the report overall; the other O quote and D quotes give copied-record origin or coordinates. None provides a qualifying compatible O or D witness. Origin is attached to the same node but does not say how its content is organized, presented or accessible, so explicit COD is unsupported.')]
pth=w/'input_11.json'
with pth.open('x') as f: json.dump(v,f,indent=2)
print(h.save_first_record(c,r['id'],w,pth.read_bytes()))
