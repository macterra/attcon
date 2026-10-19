import sys,json
from pathlib import Path
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_05.json','9821af4a85f4f4f5c5b4fd00d5516a46fcb0e0142af0b0629c8a5fda898b2bc9');r=c['records'][16]
def p(n,path,role,why):
 t=r['property_table'][n];return dict(unit_id=t['unit_id'],value_path=path,value=w.resolve(t['value'],path),semantic_role=role,reason=why)
def origin(n):return [p(n,[k],'status_operand','Exact individual origin field '+k+' of the specified node/view.') for k in r['property_table'][n]['value']]
x=p(22,[],'content_operand','A-source current n3 red content at (1,2), established from indexed source_current provenance.')
y=p(118,[],'content_operand','Native-W p1 current n3 red complete sixteen-cell peak set.')
h=p(0,[],'context_only','Authored global axis/conditional-record convention.')
def comp(path,status,assessment,notes):return dict(component_path=path,original_status=status,assessment=assessment,notes=notes)
cr=[
comp(['features','C'],'explicit','supported','Red current n3 singleton in p0 and sixteen-cell set in p1 identify represented red content and its locations. Content does not require an appearance claim.'),
comp(['features','O'],'explicit','supported','p0 n3 history_estimate with feedback_through 11 and represented_time 12 supplies a differentiated estimate representation status for the same red content. The coordinates alone would fail O; the explicit origin carries model-scope representation status.'),
comp(['features','D'],'explicit','supported','Red (1,2) and its separately specified history-estimate status are attributed to the same p0 current n3 at represented time 12. This distinguishes content/location from representation status, not answer confidence.'),
comp(['features','A'],'absent','supported','No explicit attention/allocation or processing condition is linked to a red presentation/access effect. A source identity is not itself feature A.'),
comp(['auxiliary','basic_spatial_relation'],'explicit','supported','The prefix explicitly states row-down/column-right orientation.'),
comp(['auxiliary','identity_or_location_uncertainty'],'absent','supported','Peak alternatives and history estimates are given without identity/location uncertainty wording or confidence assertions.'),
comp(['auxiliary','answer_confidence'],'absent','supported','No confidence in an answer appears.'),
comp(['auxiliary','recovery_or_success_forecast'],'absent','supported','Candidate records contain no prediction of recovery or task success.'),
comp(['auxiliary','source_process_distinction'],'explicit','supported','Current n0 copied_prior_record with record_time 11 is explicit provenance, contrasted in the full report with n3 history_estimate and p1 completed records.'),
comp(['auxiliary','experiential_assertion'],'absent','supported','The text does not assert private experience, awareness or felt appearance.'),
comp(['composites','COD'],'explicit','supported','The binding combines p0 current n3 red (1,2) with its separate history-estimate representation status, at represented time 12. This supports the structural model-scope content/status conjunction; no sensory quality is implied.'),
comp(['composites','CA'],'absent','supported','The report contains no qualifying attention-content relation, matching absent CA.')]
def ev(path,g,s,props,n):return dict(evidence_path=path,grounding=g,source_meaning=s,properties=props,notes=n)
er=[
ev(['features','C','evidence',0],'supported','source_specific',[x],'The quote directly identifies p0 current n3 red at (1,2), indexed A-source estimate content.'),
ev(['features','C','evidence',1],'supported','source_specific',[y],'The separate p1 current n3 red sixteen-peak set is native-W content, not A content by association.'),
ev(['features','O','evidence',0],'supported','source_specific',origin(20)+[x],'history_estimate, feedback horizon 11 and represented time 12 explicitly give the record representation status of red n3. The outside-quote red operand is immediately linked by p0/current/n3 in the full text. This is model organization/status, not appearance or present task performance.'),
ev(['features','O','evidence',1],'supported','source_specific',[x]+origin(20),'Red coordinates alone are not O. The same-node history_estimate origin outside the quote supplies differentiated representation status, linked by the exact current n3 prefix and represented time 12. No comparison to p1 coordinates is needed.'),
ev(['features','D','evidence',0],'supported','source_specific',origin(20)+[x],'This origin quote gives separate representation status for the same n3 red content, whose (1,2) operand is outside the quotation in the following red entry.'),
ev(['features','D','evidence',1],'supported','source_specific',[x]+origin(20),'The content quote obtains differentiated status only from the adjacent exact n3 origin fields outside the quote, not from the singleton itself. The full report binds these to represented time 12.'),
ev(['auxiliary','basic_spatial_relation','evidence',0],'supported','global_convention',[h],'Global coordinate-axis directions explicitly supply basic spatial orientation; they establish no source-specific effect.'),
ev(['auxiliary','source_process_distinction','evidence',0],'supported','source_specific',origin(2),'Exact copied_prior_record kind and record_time 11 versus represented_time 12 mark copied-W11 provenance, even though p0 has A panel provenance.'),
ev(['composites','COD','bindings',0,'evidence',0],'supported','source_specific',origin(20)+[x],'Binding origin position supplies the history-estimate status, and the following same-prefix red content operand outside the quote supplies its referent at time 12. Both are indexed A-source current operands.'),
ev(['composites','COD','bindings',0,'evidence',1],'supported','source_specific',[x]+origin(20),'The binding red position identifies (1,2), while the preceding same-node origin outside the quotation explicitly supplies differentiated representation status. No generic header is used to establish this conjunction.')]
br=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[['features','O','evidence',0],['features','O','evidence',1]]),dict(feature='D',evidence_paths=[['features','D','evidence',0],['features','D','evidence',1]])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',0],['composites','COD','bindings',0,'evidence',1]],assessment='supported',notes='The selected C/O/D witnesses all concern p0 current n3 red (1,2), represented time 12 and history-estimate status with feedback through 11. The separate origin fields supply representation status; coordinates supply identified content. Model_organization is retained as the recorded structural domain, without promotion to appearance. Indexed operands are A-source current, with no copied-W11 substitution and no inference of an attention effect from that provenance.')]
j={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']};j.update(component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['The singleton is identified content, not intrinsically clarity or a perceptual organization measure. O relies on the exact estimate-status relation.','Source-specific A-origin content here does not establish feature A, causality, spontaneous self-report or private experience.','p1 native-W alternatives are a separate source and are excluded from the p0 red binding.'],notes='All 10518 report characters read in two pages before all 12 components, 10 evidence positions and one original binding. Exact field-level origin values and peak operands preserved. No source/model-world substitution or annotation upgrade.')
pth=Path(__file__).parent/'author_16.json'
with pth.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2)
print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,pth.read_bytes())))
