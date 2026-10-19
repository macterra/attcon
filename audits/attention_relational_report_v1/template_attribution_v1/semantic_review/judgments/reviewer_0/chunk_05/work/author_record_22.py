import sys,json
from pathlib import Path
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_05.json','9821af4a85f4f4f5c5b4fd00d5516a46fcb0e0142af0b0629c8a5fda898b2bc9');r=c['records'][22]
def p(n,path,role,why):
 t=r['property_table'][n];return dict(unit_id=t['unit_id'],value_path=path,value=w.resolve(t['value'],path),semantic_role=role,reason=why)
def origin(n):return [p(n,[k],'status_operand','Individual exact origin field '+k+' of this node/view.') for k in r['property_table'][n]['value']]
x=p(100,[],'content_operand','Exact native-W p1 current n0 red singleton (0,1).')
h=p(0,[],'context_only','Authored global renderer header; no source-specific content or processing relation.')
def comp(path,status,assessment,notes):return dict(component_path=path,original_status=status,assessment=assessment,notes=notes)
cr=[
comp(['features','C'],'explicit','supported','The sole C witness directly names p1 current n0 red at (0,1), identifying represented color and location content.'),
comp(['features','O'],'explicit','supported','The generic peaks announcement is insufficient. The other witness explicitly attributes completed_record status, record_time 12 and represented_time 12 to that same red representation, differentiated from estimate/copy statuses elsewhere. This is model_access O only.'),
comp(['features','D'],'explicit','supported','The red coordinate and separately stated same-node completed_record origin distinguish identified content from its representation status at time 12.'),
comp(['features','A'],'absent','supported','No explicit attention/allocation condition or processing relation is linked to red presentation/access. The A panel label and command estimates do not supply feature A.'),
comp(['auxiliary','basic_spatial_relation'],'explicit','supported','The row-down/column-right rule literally states coordinate orientation.'),
comp(['auxiliary','identity_or_location_uncertainty'],'absent','supported','The report gives exact peak sets without expressing identity doubt or location uncertainty.'),
comp(['auxiliary','answer_confidence'],'absent','supported','No answer confidence or confidence level is asserted.'),
comp(['auxiliary','recovery_or_success_forecast'],'absent','supported','Conditional candidate records do not assert recovery or successful performance.'),
comp(['auxiliary','source_process_distinction'],'explicit','supported','Current n0 copied_prior_record and candidate k0 n2 command_estimate are literal distinct provenance kinds with their own time/horizon fields.'),
comp(['auxiliary','experiential_assertion'],'absent','supported','The report contains no experience or awareness assertion.'),
comp(['composites','COD'],'explicit','supported','The one binding uses p1 current n0 red and its completed-record representation status, compatible model_access/current at represented time 12. O is supplied by that exact status witness rather than a generic header.'),
comp(['composites','CA'],'absent','supported','A is absent and no attention-content binding is available.')]
def ev(path,g,s,props,n):return dict(evidence_path=path,grounding=g,source_meaning=s,properties=props,notes=n)
er=[
ev(['features','C','evidence',0],'supported','source_specific',[x],'Red at (0,1) identifies native-W represented current content, not actual sensory appearance.'),
ev(['features','O','evidence',0],'unsupported','global_convention',[h],'The header simply announces current and conditional candidate peaks; it states no differentiated access/presentation of p1 n0 red.'),
ev(['features','O','evidence',1],'supported','source_specific',origin(98)+[x],'The exact p1/current/n0 completed-record status distinguishes this representation from estimates/copies. The red operand lies outside the origin quote but directly follows under the same prefix and represented time 12, providing the content to which status is attributed.'),
ev(['features','D','evidence',0],'supported','source_specific',[x]+origin(98),'This red quotation contributes identified content. Outside the quote the adjacent p1/current/n0 origin supplies separate completed_record status, record_time 12 and represented_time 12; coordinates alone would not establish D.'),
ev(['features','D','evidence',1],'supported','source_specific',origin(98)+[x],'The origin quotation supplies representation status independently of the same-node red content outside the quote. Common prefix and time explicitly bind them in the full text.'),
ev(['auxiliary','basic_spatial_relation','evidence',0],'supported','global_convention',[h],'Grid directions state a global spatial mapping, without proving content organization or attention effects.'),
ev(['auxiliary','source_process_distinction','evidence',0],'supported','source_specific',origin(2),'Current n0 is explicitly copied_prior_record with record_time 11 and represented_time 12, indexed copied-W11 despite A panel membership.'),
ev(['auxiliary','source_process_distinction','evidence',1],'supported','source_specific',origin(38),'Candidate k0 n2 is explicitly command_estimate, feedback_through 11 and represented_time 13, indexed A-source conditional provenance. Future remains conditional, not executed.'),
ev(['composites','COD','bindings',0,'evidence',0],'supported','source_specific',[x]+origin(98),'This binding position supplies red content. The full text gives its completed-record status in adjacent p1/current/n0 origin outside this quote; both operands share represented time 12 and native-W source.'),
ev(['composites','COD','bindings',0,'evidence',1],'supported','source_specific',origin(98)+[x],'This separate duplicate-origin binding position supplies compatible representation status, with the following red singleton outside the quotation linked by same current n0 prefix. It does not depend on A panel content or generic header language.')]
br=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[['features','O','evidence',1]]),dict(feature='D',evidence_paths=[['features','D','evidence',0],['features','D','evidence',1]])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',0],['composites','COD','bindings',0,'evidence',1]],assessment='supported',notes='Compatible witnesses concern the same p1/current/n0 red singleton, completed_record status and record/represented time 12 in model_access/current scope. The origin is a status of this representation, while red coordinates identify its content. Header O evidence is excluded. Both binding operands have native-W provenance; source-specific A origins elsewhere are not substituted for them.')]
j={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']};j.update(component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['The generic O announcement is unsupported even though separate source-specific completed-record status supports O.','Native-W red/status binding cannot establish an A-content endpoint from joined panel membership.','Completed-record model status and represented locations do not establish perceptual appearance or felt truth.'],notes='Full 6774-character report read before complete canonical annotation, all 12 components, all 10 evidence positions and sole binding. Exact origin field leaves and full peak singleton named, with required outside-quote binding explanations. No repair or upgrade.')
pth=Path(__file__).parent/'author_22.json'
with pth.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2)
print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,pth.read_bytes())))
