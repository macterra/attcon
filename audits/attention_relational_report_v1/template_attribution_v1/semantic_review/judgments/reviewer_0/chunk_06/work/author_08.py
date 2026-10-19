import sys,json
from pathlib import Path
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_06.json','b8ee8f672053efcae8359b3688fbeb9d231d88f772241cf897ae50be7ec15708');r=c['records'][8]
def prop(u,p,v,role,reason):return dict(unit_id=u,value_path=p,value=v,semantic_role=role,reason=reason)
def cp(p,s,a,n):return dict(component_path=p,original_status=s,assessment=a,notes=n)
def ev(p,g,m,ps,n):return dict(evidence_path=p,grounding=g,source_meaning=m,properties=ps,notes=n)
red0=prop('u0004',[],[[0,3]],'content_operand','Names the red peak at p0/current/n0; full text binds its immediately preceding n0 origin to this content.')
red3=prop('u0022',[],[[0,1]],'content_operand','Names p0/current/n3 red content at the history-estimated current node.')
redw=prop('u0100',[],[[0,3]],'content_operand','Names the red content in p1/current/n0, a native W current record.')
kind0=prop('u0002',['kind'],'copied_prior_record','context_only','Identifies copying provenance at the same p0/current/n0 address, not an asserted presentation or access quality.')
rt0=prop('u0002',['record_time'],11,'context_only','The source record was made at 11; distinguish this from its represented current time.')
t0=prop('u0002',['represented_time'],12,'context_only','Binds the n0 content to represented time 12 through the full matching node address.')
k3=prop('u0020',['kind'],'history_estimate','context_only','The n3 origin is an estimate label; it supplies provenance but no explicit uncertainty assertion.')
f3=prop('u0020',['feedback_through'],11,'context_only','The estimator uses feedback through 11 for the red n3 content; this does not quantify location uncertainty.')
t3=prop('u0020',['represented_time'],12,'context_only','The immediately preceding same-address origin dates p0/current/n3 red to 12.')
kw=prop('u0098',['kind'],'completed_record','context_only','The full text assigns a completed-record origin to p1/current/n0 red.')
rw=prop('u0098',['record_time'],12,'context_only','Dates the native W source record for p1/current/n0.')
tw=prop('u0098',['represented_time'],12,'context_only','The same p1/current/n0 address explicitly dates represented red content to 12.')
header=prop('u0000',[],'Rows increase down; columns right. Current and conditional candidate peaks follow.','context_only','The quoted first sentence is the renderer axis convention; it does not describe any content-specific presentation or access.')
ck=prop('u0044',['kind'],'command_estimate','processing_relation_operand','Specifies a conditional command-based estimate at p0/candidate/k0/n3; no executed branch is asserted.')
cf=prop('u0044',['feedback_through'],11,'processing_relation_operand','Dates the feedback available to this conditional estimate, rather than claiming a successful action.')
ct=prop('u0044',['represented_time'],13,'context_only','The candidate origin represents future time 13; preserve its forecast scope.')
cre=prop('u0046',[],[[0,1]],'content_operand','Same full candidate/k0/n3 address binds this red peak to the command-estimate origin outside the quoted overlap.')
a=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,
component_reviews=[
cp(['features','C'],'explicit','supported','Red peaks are identified at p0/current/n0=(0,3), p0/current/n3=(0,1), and p1/current/n0=(0,3). These are represented content attributes, with no appearance claim.'),
cp(['features','O'],'explicit','unsupported','The copied-prior-record origin states provenance and dates; the other witness states only axis directions. Neither explicitly describes current access, presentation, or a content-specific organized pattern beyond coordinate lists.'),
cp(['features','D'],'explicit','unsupported','The cited n0 origin separates provenance metadata from red coordinates, but does not separately attribute an organization/presentation/access status as required for D.'),
cp(['features','A'],'absent','supported','The full text lists current and conditional peaks with command_estimate and native_conditional_record origins. It does not explicitly relate an allocation or processing condition to organization, emergence or access of identified content.'),
cp(['auxiliary','basic_spatial_relation'],'explicit','supported','The grid convention explicitly maps increasing rows down and increasing columns right. This supports the recorded grid_layout relation only.'),
cp(['auxiliary','identity_or_location_uncertainty'],'explicit','unsupported','history_estimate at p0/current/n3 labels provenance. Its red peak is a singleton; neither the quoted origin nor the report explicitly expresses identity/location uncertainty for that referent.'),
cp(['auxiliary','answer_confidence'],'absent','supported','No answer confidence judgment or confidence scalar is stated in the report.'),
cp(['auxiliary','recovery_or_success_forecast'],'absent','supported','Conditional future peak records are supplied, but there is no forecast of recovery or success.'),
cp(['auxiliary','source_process_distinction'],'explicit','supported','The report distinguishes a copied prior record at current n0 from a command estimate at candidate k0/n3, including different represented times and feedback provenance.'),
cp(['auxiliary','experiential_assertion'],'absent','supported','The report states coordinate and provenance records without asserting experience or felt appearance.'),
cp(['composites','COD'],'explicit','unsupported','The original n0/red binding has identified content, but its duplicated origin witnesses provide provenance rather than O and D statuses. Coordinate and date compatibility does not supply the missing semantic relation.'),
cp(['composites','CA'],'absent','supported','No compatible attention/content-organization relation is stated; command labels alone cannot establish A.')],
evidence_reviews=[
ev(['features','C','evidence',0],'supported','source_specific',[red0,kind0,rt0,t0],'The p0 n0 red peak identifies represented content at time 12. Indexed provenance is copied W11 even though the panel is A; scope remains model representation, with no appearance inference.'),
ev(['features','C','evidence',1],'supported','source_specific',[red3,k3,f3,t3],'The n3 red singleton identifies source-current A content at represented time 12. The matching full address binds its history-estimate origin outside this quotation.'),
ev(['features','C','evidence',2],'supported','source_specific',[redw,kw,rw,tw],'The p1 n0 red singleton identifies native W content at represented time 12. Its same-address full-text origin supplies time and provenance outside the peak quotation.'),
ev(['features','O','evidence',0],'unsupported','source_specific',[kind0,rt0,t0,red0],'The full matching node address links this copied W11 origin to current n0 red, but copied provenance and dates do not assert an O presentation/access status.'),
ev(['features','O','evidence',1],'unsupported','global_convention',[header],'The quote is the axis convention portion of u0000. Direction conventions provide no O status of any identified content; they cannot bind this display statement to n0 model access.'),
ev(['features','D','evidence',0],'unsupported','source_specific',[kind0,rt0,t0,red0],'The n0 provenance is separately written beside n0 red. It does not distinguish content from organization/presentation/access status, so the recorded D role is unsupported.'),
ev(['auxiliary','basic_spatial_relation','evidence',0],'supported','global_convention',[header],'The first sentence explicitly supplies grid directions, matching the recorded grid_layout/axis_mapping display relation. It is global renderer convention, not source-content evidence.'),
ev(['auxiliary','identity_or_location_uncertainty','evidence',0],'unsupported','source_specific',[k3,f3,t3,red3],'A history-estimate label with feedback through 11 and represented time 12 is not explicit location uncertainty. The linked n3 red singleton identifies the estimated content without expressing doubt.'),
ev(['auxiliary','source_process_distinction','evidence',0],'supported','source_specific',[kind0,rt0,t0,red0],'The quoted current n0 origin explicitly identifies a copied record from 11 representing 12. Full-text n0 red supplies its content operand; this is source provenance, without upgrading O.'),
ev(['auxiliary','source_process_distinction','evidence',1],'supported','source_specific',[ck,cf,ct,cre],'The k0/n3 origin explicitly labels an A command estimate using feedback through 11 for 13. The same full address binds candidate red outside the quote; it distinguishes estimation from copying without asserting execution or success.'),
ev(['composites','COD','bindings',0,'evidence',0],'supported','source_specific',[red0,kind0,rt0,t0],'This binding entry supports the C red content operand at current n0. Same-address provenance dates it but supplies no additional O/D witness.'),
ev(['composites','COD','bindings',0,'evidence',1],'unsupported','source_specific',[kind0,rt0,t0,red0],'First origin entry of the COD binding: copied W11 provenance is tied to n0 red at 12, but it is not an organization/access description required by O.'),
ev(['composites','COD','bindings',0,'evidence',2],'unsupported','source_specific',[kind0,rt0,t0,red0],'The second identical origin entry is preserved at its distinct evidence position. Repetition does not turn the same provenance fields into a separate D organization/access status.')],
binding_reviews=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',0],['composites','COD','bindings',0,'evidence',1],['composites','COD','bindings',0,'evidence',2]],assessment='unsupported',notes='C evidence 0 and all three own entries share p0/current/n0; its red attribute is bound by that address to copied-record origin, record_time=11 and represented_time=12. However origin is provenance, not the required presentation/access status. O evidence 1 instead concerns the global display convention and cannot provide compatible model-access evidence. D evidence 0 repeats provenance without an O status. Therefore there are no semantically compatible O/D feature witnesses despite exact referent/time overlap.')],
additional_concerns=['Panel A contains copied W11 current n0 evidence. It cannot be treated as an A-content processing witness solely from the panel label.','Conditional candidate time 13 must not be converted into an executed branch or a current appearance claim.','The report contains multi-peak coordinate lists elsewhere; those lists alone do not establish O/D or uncertainty for the cited n3 singleton.'],notes='Complete text, all 12 components, all 13 evidence positions including the duplicated binding origin, and the original COD binding were inspected. Provenance and coordinate labels were not promoted into presentation/access statuses.')
w=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_06/work');p=w/'authored_08.json'
with p.open('x') as f:f.write(json.dumps(a,indent=2)+'\n')
print(h.save_first_record(c,r['id'],w,p.read_bytes()))
