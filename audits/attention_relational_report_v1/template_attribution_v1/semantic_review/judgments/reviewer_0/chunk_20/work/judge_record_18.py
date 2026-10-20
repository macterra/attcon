import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_20.json'),'dce8ec6b7058f0946a47c90706ae9815542e2ec0f0471fc6e85f01281cf64d0d');r=c['records'][18]
assert r['id']=='s1311_w01_t12_ordinary_joined_baseline_template'
def p(i,path,role,reason):
 u=w.source_view(c,r['id'],['property_table',i],0,20)['value'];return dict(unit_id=u['unit_id'],value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
red=p(4,[],'content_operand','The complete red current n0 singleton identifies location content (0,2). Indexed source is copied_W11 despite the ordinary panel label.')
ori=[p(2,[k],'status_operand' if k=='kind' else 'context_only','Exact current n0 origin field '+k+' describes copied_prior_record provenance from record time 11 representing time 12; no presentation/access quality is asserted.') for k in ['kind','record_time','represented_time']]
head=p(0,[],'context_only','Only the quoted first clause specifies grid orientation: rows down and columns right. The remainder announces the inventory; the token is a global renderer convention.')
cs=[
 ('supported','Red current n0 at (0,2) is an explicitly identified represented content/location attribute. This suffices for C.'),
 ('unsupported','The sole O quote is copied_prior_record provenance with record and represented times. It reports where a record came from rather than spatial organization, presentation or accessibility of red beyond coordinates.'),
 ('unsupported','The red coordinate and copied origin are separate content/provenance fields. They do not separately attribute an organization/presentation/access status to the content as required for D.'),
 ('supported','The full report lists current and conditional peaks and command estimates but no explicit allocation/processing relation to qualifying content presentation/access. Arbitrary commands and changed coordinates are insufficient for A.'),
 ('supported','The row-down/column-right statement explicitly gives grid spatial orientation in display scope.'),
 ('supported','Neither the four-cell ties at n1/n2 nor the singleton estimates are explicitly described as uncertain identities/locations. No separate uncertainty assertion is made.'),
 ('supported','No answer confidence value or verbal confidence assessment appears in the complete report.'),
 ('supported','Candidate location tables make no success or recovery claim; no outcome-success forecast is expressed.'),
 ('supported','The copied_prior_record kind, record time 11 and represented time 12 explicitly distinguish stored source provenance. The auxiliary distinction is supported without treating origin as access quality.'),
 ('supported','The coordinate/origin inventory asserts no private experience, consciousness or felt perceptual appearance.'),
 ('unsupported','The COD binding has compatible current red n0 content and copied-record provenance, but no qualifying O or D status. Same prefix and time cannot replace those missing meanings.'),
 ('supported','With C present but no qualifying A relation, absent CA remains warranted.')]
cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=n) for x,(a,n) in zip(r['components'],cs)]
es=[
 ('supported','source_specific',[red],'The red singleton grounds C at current n0/represented time 12. No organization or appearance status is implied by the location.'),
 ('unsupported','unrelated_property',ori,'Copied origin and record/represented-time metadata have literal source meaning but are unrelated to the claimed O organization/access role.'),
 ('unsupported','unrelated_property',[red],'The red location is identified content only; it expresses no separately attributed presentation/access status for D.'),
 ('unsupported','unrelated_property',ori,'The origin is separate from content yet describes record history. It does not distinguish identification from a qualifying organization/access status.'),
 ('supported','global_convention',[head],'The first header clause explicitly supplies display grid orientation at report time; it is a convention shared by all entries, not source-specific model organization.'),
 ('supported','source_specific',ori,'The quoted current n0 origin explicitly marks copied-record provenance and its times, supporting source/process distinction. Indexed meaning is copied_W11, while model_access annotation alone does not establish access quality.'),
 ('unsupported','unrelated_property',[red],'This binding quote supplies the C/red content operand at current time only, not the O/D operands needed for COD.'),
 ('unsupported','unrelated_property',ori+[red],'The outside red peak operand is linked in the full report by the identical current/n0 prefix to the quoted origin at represented time 12. That establishes content/provenance correspondence only; copied source history does not ground complete COD.')]
er=[dict(evidence_path=x['path'],grounding=g,source_meaning=s,properties=ps,notes=n) for x,(g,s,ps,n) in zip(r['evidence'],es)]
br=[dict(binding_path=r['bindings'][0]['path'],feature_witnesses=[dict(feature='C',evidence_paths=[r['evidence'][0]['path']]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[r['evidence'][6]['path'],r['evidence'][7]['path']],assessment='unsupported',notes='The own red singleton and copied origin both concern current n0 at represented time 12, and compatible C is present. The only proposed O token is provenance rather than organization/access; the proposed D pair repeats that provenance beside coordinates. No qualifying O or D witness is available for the same red attribute. Current p1 completed records and future candidates are not substituted into the current p0 binding.')]
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['The original O/D/COD rationale treats origin metadata as presentation/access status without textual support.','The indexed current red n0 content is copied_W11 rather than ordinary-source current output; panel naming does not establish source meaning.','The displayed current and conditional records preserve distinct temporal/provenance meanings; no completed future branch or perceptual appearance follows from their coordinates.'],notes='Complete 6918-character report read first. All 12 original statuses, all 8 evidence positions and the single binding are reviewed and preserved. Exact property operands name full peak sets and individual origin fields without semantic repair.')
print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())))
