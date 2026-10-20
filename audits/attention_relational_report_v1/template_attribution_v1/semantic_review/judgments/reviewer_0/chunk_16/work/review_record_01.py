import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_16.json','1ba8f3a46e99d796822029fae67f6b8a1bfad4501de7131fd67364f20113f37f'); i=c['ordered_template_ids'][1]
def v(p):return h.source_view(c,i,p,0,10000)['value']
def prop(n,path,role,reason):
 u=v(['property_table',n]);val=u['value']
 for p in path:val=val[p]
 return dict(unit_id=u['unit_id'],value_path=path,value=val,semantic_role=role,reason=reason)
content=prop(4,[],'content_operand','The complete red peak set identifies p0/current/n0 red at (0,3), current represented time 12. This is copied W11 content according to indexed provenance.')
orig=[prop(2,[k],'context_only','The same p0/current/n0 prefix explicitly binds this provenance field to the red peak set, including when the field is outside the quoted overlap. It describes copying a time-11 record into a time-12 representation, without asserting an organization, emergence or accessibility status.') for k in ['kind','record_time','represented_time']]
history=[prop(20,[k],'context_only','The p0/current/n3 origin explicitly identifies the history-estimate source and its feedback/represented times. Compared with p0/current/n0 copied-prior-record metadata in the full report, this distinguishes estimate provenance from copied record provenance; neither is an appearance claim.') for k in ['kind','feedback_through','represented_time']]
notes={
('features','C'):('supported','p0/current/n0 red={(0,3)} identifies represented red content and its location attribute. The coordinate set suffices for C only.'),
('features','O'):('unsupported','copied_prior_record with record_time 11 and represented_time 12 states provenance and bookkeeping times. It does not state that the red content is accessible, emerges, or has a differentiated presentation/constituent organization. The original rationale converts provenance into access status.'),
('features','D'):('unsupported','The separate red coordinate and origin fields distinguish content from provenance, but no separate organization/presentation/access status is expressed. Physical location plus copying metadata does not establish reference D.'),
('features','A'):('supported','The report lists conditional k0/k1/k2 origins and peaks but relates no allocation/processing condition to organization, emergence, presentation or access of red or blue content.'),
('auxiliary','basic_spatial_relation'):('supported','Rows increase down and columns right explicitly specifies two directional spatial relations for the display grid.'),
('auxiliary','identity_or_location_uncertainty'):('supported','Multiple maxima are listed for some nodes, but the report expresses no uncertainty claim about identity or location. Tied coordinate inventories alone do not explicitly state uncertainty.'),
('auxiliary','answer_confidence'):('supported','No answer-confidence value or confidence statement appears in the report.'),
('auxiliary','recovery_or_success_forecast'):('supported','Conditional candidate peak sets do not assert recovery or successful outcomes; no success forecast appears.'),
('auxiliary','source_process_distinction'):('supported','history_estimate with feedback_through 11 and represented_time 12 is explicitly separated from copied_prior_record origins and completed_record origins. This is a provenance/process distinction at model scope.'),
('auxiliary','experiential_assertion'):('supported','The renderer asserts origin fields and peaks, without any assertion of experience or felt appearance.'),
('composites','COD'):('unsupported','C is explicit, but the original O and D witnesses contain provenance only. Shared p0/current/n0 and current-time labels cannot supply the missing organization/access meaning.'),
('composites','CA'):('supported','No allocation/content-processing relation to organization/presentation/access is stated, so absent A supports absent CA.')}
cr=[]
for comp in v(['components']):
 a,n=notes[tuple(comp['path'])];cr.append(dict(component_path=comp['path'],original_status=comp['recorded_status'],assessment=a,notes=n))
er=[]
for e in v(['evidence']):
 p=e['path']; key=tuple(p[:2]);
 if key==('features','C'):
  g='supported';sm='source_specific';ps=[content];n='The exact peak-set proposition identifies the recorded red content; copied-W11 attribution follows u0004 source_resolution, not the ordinary panel label. It states no accessibility predicate.'
 elif key==('auxiliary','basic_spatial_relation'):
  g='supported';sm='global_convention';ps=[prop(0,[],'context_only','Only the row/column orientation sentence is quoted from this header; the full value is the authored coordinate and candidate convention, not an actual content operand.')];n='The prefix directly states grid direction; the unquoted candidate sentence contributes no source-specific spatial content.'
 elif key==('auxiliary','source_process_distinction'):
  g='supported';sm='mixed_sources';ps=history+orig;n='The quoted source-current history-estimate metadata and the explicitly matching origin fields for copied-W11 nodes in the full report distinguish estimation from copied record provenance. Both sources are retained; this does not prove accessible organization.'
 else:
  g='unsupported';sm='source_specific';ps=[content]+orig;n='Both exact propositions concern p0/current/n0 at represented time 12; their shared prefix explicitly binds the coordinate content and provenance, even when one operand is outside this evidence quote. The copying kind and times do not assert an organization/presentation/access status, so this is not a grounded '+('/'.join(map(str,p[:2])))+' witness. The indexed source is copied W11.'
 er.append(dict(evidence_path=p,grounding=g,source_meaning=sm,properties=ps,notes=n))
br=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature=f,evidence_paths=[e['path'] for e in v(['evidence']) if e['path'][:2]==['features',f]]) for f in ['C','O','D']],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',j] for j in [0,1]],assessment='unsupported',notes='The binding red coordinate and origin share p0/current/n0, current representation time 12 and model_access labels. C identifies red at (0,3), while O and D reuse copied-record provenance without an accessibility/presentation/organization predicate. Exact referent/time agreement does not establish the missing feature meanings.')]
r=dict(id=i,unit_id=v(['unit_id']),text_sha256=v(['text_sha256']),reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['Origin bookkeeping is converted to organization/access in the original O/D/COD rationale.','A full-grid tied peak set is not a perceptual appearance or answer-confidence measurement.','The p0/current/n0 witnesses are copied W11; the panel source label ordinary does not make them current ordinary-source content.','All conditional candidate statements remain future conditions, including native records with executed_branch_claim false; no executed or currently accessible outcome is established.'],notes='The complete 9612-character report was inspected before the exact annotation, evidence and property operands. No original status is changed or repaired; unsupported judgments preserve those statuses.')
print(h.save_first_record(c,i,'audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_16/work',json.dumps(r,ensure_ascii=True,indent=2).encode()+b'\n'))
