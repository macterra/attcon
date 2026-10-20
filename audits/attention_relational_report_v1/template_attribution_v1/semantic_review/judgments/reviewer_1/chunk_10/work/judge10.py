import runpy,sys,json
sys.argv=['local_review.py','10','load']
x=runpy.run_path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_10/work/local_review.py')
r,c,h,W=x['r'],x['c'],x['h'],x['W']
def prop(i,p,role,reason):
 u=r['property_table'][i];return dict(unit_id=u['unit_id'],value_path=p,value=h.resolve(u['value'],p),semantic_role=role,reason=reason)
def peaks(i,reason):return prop(i,[],'content_operand',reason)
def fields(i,paths,reason,role='status_operand'):return [prop(i,p,role,reason) for p in paths]
current=fields(8,[['kind'],['record_time'],['represented_time']],'The p0/current/n1 prefix binds these provenance/time fields to its red peak token u0010 outside this origin quote; completed_record is record provenance, not an access or presentation assertion.')
conditional=fields(38,[['kind'],['executed_branch_claim'],['represented_time']],'The p0/candidate/k0/n2 prefix binds this native conditional provenance to the red content u0040 outside the origin quote; it is represented for time 13 and explicitly does not claim an executed branch.')
inherited=fields(50,[['kind'],['represented_time'],['inherited_origin','kind'],['inherited_origin','record_time'],['inherited_origin','represented_time']],'The p0/candidate/k1/n0 prefix binds inherited_current at 13 to the nested completed_record from time 12; these are individually cited provenance fields, not an attention or appearance relation.','processing_relation_operand')
notes=[
('supported','Named red content is identified by a singleton current n0 peak, all sixteen tied n1 peaks, and a conditional n2 singleton; coordinates suffice for C without establishing access.'),
('unsupported','completed_record and native_conditional_record classify the origin of records. They do not state how red is organized/presented or currently accessible; the list of maxima supplies no additional organization/access statement.'),
('unsupported','Red maxima and an adjacent origin field are separately printed for n1, but provenance/completion is not a separately attributed content organization/presentation/access status. Their co-occurrence does not meet D.'),
('supported','No allocation or content-processing condition is explicitly related to content organization/access. Candidate labels and conditional provenance alone do not establish A.'),
('supported','The n0 red singleton at row 0 column 1 explicitly gives basic location under the stated coordinate convention; this supports the auxiliary status, although its evidence display/sensor scope is unsupported.'),
('supported','No explicit uncertainty statement is made; tied maxima list positions without verbally attributing uncertainty to identity or location.'),
('supported','No answer-selection confidence or response-certainty claim occurs in this complete report.'),
('supported','Conditional maxima at time 13 do not forecast recovery or success; there is no success outcome claim.'),
('supported','Native conditional nonexecution is explicitly distinguished from an inherited current record with nested completed-record provenance; this is source/process metadata rather than perceptual status.'),
('supported','No felt experience or awareness assertion is made by these origin fields and peak coordinates.'),
('unsupported','The current n1 red referent and time are shared, but the putative O and D witnesses give provenance only; an identified coordinate set plus completed_record cannot establish COD.'),
('supported','A lacks a content-processing-to-organization/access relation, so no compatible C/A binding establishes CA.')]
out={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
out['component_reviews']=[dict(component_path=comp['path'],original_status=comp['recorded_status'],assessment=a,notes=n) for comp,(a,n) in zip(r['components'],notes)]
P=peaks
reviews=[
('supported',[P(4,'Complete current n0 red peak-set operand identifies red location (0,1).')],'The quoted n0 red singleton identifies represented content at current time, without independently asserting access.'),
('supported',[P(10,'Complete current n1 red peak set includes every grid cell; no unique maximum is implied.')],'The full sixteen-position set identifies red represented locations at current n1; tied coordinates do not add access or clarity.'),
('supported',[P(40,'Complete red candidate n2 peak operand is (0,1).')]+conditional,'The candidate k0/n2 prefix and its adjacent origin bind this red coordinate to a conditional time-13 representation, not present or executed content.'),
('unsupported',current+[P(10,'Outside the origin quote, the identical p0/current/n1 prefix explicitly binds the red peak set to the quoted origin.')],'The origin applies to the same n1 record at 12, but completion describes provenance rather than model access or content presentation.'),
('unsupported',conditional+[P(40,'Outside this origin quote, p0/candidate/k0/n2/red supplies the same-record content operand.')],'Conditional origin and nonexecution identify provenance and time, not an organization/access status of red.'),
('unsupported',[P(10,'All sixteen current n1 red peak coordinates are the content operand, without a differentiated presentation status.')]+current,'The companion origin is bound by identical n1 prefix, but record completion and coordinates do not provide the distinct organization/access status required for D.'),
('unsupported',current+[P(10,'The adjacent red token outside the origin overlap is explicitly bound by p0/current/n1.')],'There is a content/provenance distinction in syntax, not a content-versus-presentation/access distinction under the reference.'),
('unsupported',[P(4,'Red n0 location is the actual spatial operand, resolved to native W provenance.')],'The location is spatial, but its recorded display_or_sensor_property scope is not supported by the W source; this is a native record coordinate.'),
('supported',conditional,'kind=native_conditional_record, executed_branch_claim=false and represented_time=13 explicitly characterize the candidate source/process status; this is not a current execution claim.'),
('supported',inherited,'The outer inherited_current kind and time 13, with individually cited nested completed_record kind and both time-12 fields, explicitly expose inherited versus native provenance.'),
('unsupported',[P(10,'Current n1 red set supplies the C operand for this COD binding.')]+current,'This binds red and its record provenance at 12 but provides no warranted O/D presentation/access status for the recorded COD role.'),
('unsupported',current+[P(10,'Red content outside the quoted origin shares its exact current/n1 prefix and time.')],'The exact prefix links origin and content, but that relation is provenance, not the required organization/access and differentiation conjunction.')]
out['evidence_reviews']=[dict(evidence_path=e['path'],grounding=g,source_meaning='source_specific',properties=p,notes=n) for e,(g,p,n) in zip(r['evidence'],reviews)]
out['binding_reviews']=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',1]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',0],['composites','COD','bindings',0,'evidence',1]],assessment='unsupported',notes='The red n1 set and completed-record origin share p0/current/n1 and represented time 12. C has an exact witness; no compatible O or D witness states content organization/presentation/access. Completion metadata does not fill those missing roles.')]
out['additional_concerns']=['Original model_access scopes on provenance/coordinate claims are not access assertions. The basic-spatial evidence is native W, not a display/sensor observation.','All tied n1 maxima must remain the complete sixteen-cell set; a coordinate list does not imply uncertainty, organization, appearance, or experiential truth.','The future candidate explicitly disclaims execution; no current processing or perceptual scope is inferred.']
out['notes']='Established original retained with all twelve original statuses, twelve distinct evidence positions including duplicates, and its one original binding; no status or scope is repaired.'
print(json.dumps(h.save_first_record(c,r['id'],W,(json.dumps(out,indent=2)+'\n').encode())))
