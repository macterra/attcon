import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_14.json'),'1e90a6d34227a09a71008057686ce3b9b84fca019230f3509aaa18047d25905f');r=c['records'][12];assert r['id']=='s1301_w03_t12_A_A_only_W_restored_template';assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
o={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
component_specs=[
('supported','Red and blue are separately named represented contents, each with a full-grid set of peak coordinates at p0/current/n0.'),
('unsupported','The copied_prior_record origin specifies source provenance and times, not how red/blue are currently accessible or presented. The model_access label does not make that an access-status assertion.'),
('unsupported','The origin is separately listed from content but differentiates copied provenance, not an organization/presentation/access status; the prerequisite status relation is missing.'),
('supported','Command-estimate labels and differences among future coordinate sets contain no explicit processing-condition-to-organization/access relation; A absence is warranted.'),
('supported','Rows increasing down and columns right explicitly state a display orientation relation.'),
('supported','No identity/location uncertainty is stated. Full-grid maximal ties alone do not identify a guess or uncertainty assertion.'),
('supported','No answer rating, answer probability or answer-confidence statement occurs in the full report.'),
('supported','Conditional coordinate predictions do not express recovery or success of a task outcome.'),
('supported','Copied prior record with record_time 11 is explicitly distinguished from history_estimate with feedback_through 11; both represent time 12. This supports the provenance auxiliary only.'),
('supported','The report contains no experiential or awareness assertion.'),
('unsupported','Both same-node COD bindings pair color coordinates with copied origin metadata. Neither supplies actual presentation/access O or content/status D under the frozen reference.'),
('supported','There is no explicit A relation or CA binding despite represented red/blue content.')]
o['component_reviews']=[dict(component_path=cp['path'],original_status=cp['recorded_status'],assessment=a,notes=n) for cp,(a,n) in zip(r['components'],component_specs)]
def p(idx,path,role,reason):
 t=r['property_table'][idx];v=w.resolve(t['value'],path);return dict(unit_id=t['unit_id'],value_path=path,value=v,semantic_role=role,reason=reason)
def origin(role):
 return [p(2,['kind'],role,'The exact source origin kind is copied_prior_record, resolved as copied W11 despite panel A.'),p(2,['record_time'],role,'The copied record was recorded at time 11.'),p(2,['represented_time'],role,'The node represents time 12, separately from its record time.')]
red=p(4,[],'content_operand','Complete red maximal-coordinate set at p0/current/n0, copied W11; all sixteen maximal cells are retained.')
blue=p(6,[],'content_operand','Complete blue maximal-coordinate set at p0/current/n0, copied W11; it equals the red set without establishing equal experience or access.')
header=p(0,[],'context_only','The first sentence overlaps the fixed renderer header. The full actual token value includes the following candidate header sentence; it is a global convention.')
history=[p(20,['kind'],'status_operand','history_estimate is the current A-origin kind at n3.'),p(20,['feedback_through'],'status_operand','Feedback through 11 is the actual history provenance operand.'),p(20,['represented_time'],'status_operand','The history estimate represents time 12, not observed future execution.')]
specs=[
('supported','source_specific',[red],'Named red content with all sixteen maximal coordinates supports C; model_access scope is retained without certifying access phenomenology.'),
('supported','source_specific',[blue],'Named blue content with its complete maximal-coordinate set supports C independently of red; both are represented time 12.'),
('unsupported','source_specific',origin('context_only')+[red,blue],'The adjacent red and blue tokens share p0/current/n0 with the origin, binding these out-of-quote operands to the same current node. The origin tells how the record was obtained, without a claim about current access/presentation; O is unsupported.'),
('unsupported','source_specific',origin('context_only')+[red,blue],'Same-node prefixes bind the copied origin outside the color quotations to both colors. Source-versus-content distinction is explicit, but no organization/presentation/access status is separately attributed, so D is not grounded.'),
('supported','global_convention',[header],'Rows/down and columns/right ground a display layout relation. The partial header does not attribute source-specific A content.'),
('supported','source_specific',origin('status_operand'),'The three exact origin fields establish copied W11 provenance with record time 11 and represented time 12; source/process role is supported without converting it to O/D.'),
('supported','source_specific',history,'The exact history_estimate, feedback_through 11 and represented_time 12 establish this current A-source provenance; this evidence position remains separate from the copied-origin position.'),
('unsupported','source_specific',[red]+origin('context_only'),'The red coordinates are C. Outside this quote, adjacent origin tokens share p0/current/n0 and represented time 12, but no O/D relation follows; this COD evidence is insufficient.'),
('unsupported','source_specific',origin('context_only')+[red],'The quoted copied-origin fields bind to red via the adjacent identical p0/current/n0 prefix, giving compatible content/time but provenance alone supplies neither O nor D.'),
('unsupported','source_specific',[blue]+origin('context_only'),'Blue coordinates independently ground C. Same-node origin fields outside the quote give source/time context, without access or presentation status required for COD.'),
('unsupported','source_specific',origin('context_only')+[blue],'This second repeated origin evidence applies to blue through p0/current/n0. Its exact values remain provenance metadata, insufficient for the blue COD binding.')]
o['evidence_reviews']=[dict(evidence_path=e['path'],grounding=g,source_meaning=s,properties=ps,notes=n) for e,(g,s,ps,n) in zip(r['evidence'],specs)]
o['binding_reviews']=[
 dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[['features','O','evidence',0]]),dict(feature='D',evidence_paths=[['features','D','evidence',0]])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',0],['composites','COD','bindings',0,'evidence',1]],assessment='unsupported',notes='The original red C/O/D candidate paths share p0/current/n0 and represented time 12. Red is identified, but origin metadata does not state red presentation/access status. The O/D paths are preserved candidate paths, not accepted witnesses, so red COD fails.'),
 dict(binding_path=['composites','COD','bindings',1],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',1]]),dict(feature='O',evidence_paths=[['features','O','evidence',0]]),dict(feature='D',evidence_paths=[['features','D','evidence',0]])],binding_evidence_paths=[['composites','COD','bindings',1,'evidence',0],['composites','COD','bindings',1,'evidence',1]],assessment='unsupported',notes='The blue C candidate and shared origin O/D candidates bind by p0/current/n0 and time 12. Shared node/time is compatible, but copied origin does not establish blue access/presentation or content/status differentiation. These candidate paths do not complete COD.')]
o['additional_concerns']=['The A panel contains copied W11 n0 fields, and its label cannot make the red/blue contents A-derived.','Do not treat generic history/command estimate provenance as an attention relation or a current access claim.','Full-grid maximal ties identify peak sets but do not independently express a presentation pattern beyond listing positions.','Conditional future estimates remain forecasts; no execution, experiential assertion or claimed appearance is established.']
o['notes']='All 12 components, 11 evidence positions and both original bindings reviewed; repeated origin quotations retained separately. Original statuses/scope are preserved, without repairing labels. Initial setup accidentally emitted assigned chunk into a truncated output, which was not counted as inspection; this report and exact used source units were subsequently inspected separately.'
base=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_14/work');raw=(json.dumps(o,indent=2)+'\n').encode()
with (base/'draft_record_12.json').open('xb') as f:f.write(raw)
print(json.dumps(w.save_first_record(c,r['id'],base,raw)))
