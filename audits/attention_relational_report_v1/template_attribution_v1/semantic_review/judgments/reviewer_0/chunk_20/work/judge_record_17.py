import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_20.json'),'dce8ec6b7058f0946a47c90706ae9815542e2ec0f0471fc6e85f01281cf64d0d');r=c['records'][17]
assert r['id']=='s1311_w00_t12_analytic_A_only_current_restored_template'
def p(i,path,role,reason):
 u=w.source_view(c,r['id'],['property_table',i],0,20)['value'];return dict(unit_id=u['unit_id'],value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def origin(i,reason):
 return [p(i,[k],'status_operand' if k=='kind' else 'context_only',reason+' Exact '+k+' is provenance/temporal context and asserts no presentation/access quality.') for k in r['property_table'][i]['value']]
cs=[
 ('supported','The report identifies red current n1 and its location (0,2), supporting explicit C. This is represented coordinate content without a perceptual-appearance claim.'),
 ('unsupported','The current n1 copied origin and candidate k1/n0 command estimate specify provenance and different represented times. They do not state a constituent pattern beyond coordinates or a presentation/access status for identified content.'),
 ('unsupported','The red n1 coordinate and its copied origin are separate fields, but origin is source history rather than a qualifying organization/presentation/access status. Their separation alone does not establish D.'),
 ('supported','The complete report contains command estimates and conditional peak sets, but no explicit processing/allocation condition related to qualifying organization/access of the same content. A remains absent under the coordinate/command exclusion.'),
 ('supported','Rows increase down and columns right explicitly specify display-axis orientation. Retaining unclear time is reasonable for a general convention.'),
 ('supported','No explicit uncertainty about identity or location is asserted. Equal peak alternatives and generic estimate names alone do not express such uncertainty.'),
 ('supported','No answer confidence value or verbal confidence judgment is given.'),
 ('supported','The future coordinate tables give no recovery or success expectation.'),
 ('supported','History_estimate with feedback through 11 and represented time 12 explicitly identifies an estimation process for current n3. This is provenance, not an organization-quality judgment.'),
 ('supported','No assertion of experience, feeling, consciousness or perceptual appearance is present.'),
 ('unsupported','The COD binding shares current n1 and red with the coordinate and copied-origin entries, but no O/D status is expressed. Same referent/time and separate metadata are insufficient.'),
 ('supported','There is C but no qualifying A relation, so absent CA is supported.')]
cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=n) for x,(a,n) in zip(r['components'],cs)]
red=p(10,[],'content_operand','The complete red singleton is current n1 location content. Indexed origin is copied_W11 despite the analytic panel label.')
n1=origin(8,'Current n1: copied_prior_record from record_time 11 representing 12.')
k1=origin(50,'Candidate k1/n0: command_estimate using feedback through 11 for represented time 13.')
n3=origin(20,'Current n3: history_estimate using feedback through 11 for represented time 12.')
head=p(0,[],'context_only','Only the quoted row-down/column-right first clause states spatial orientation; the containing second clause announces inventory. This is a global renderer convention.')
es=[
 ('supported','source_specific',[red],'The exact singleton explicitly names red current n1 at (0,2), directly grounding C. It gives represented content, not organization quality.'),
 ('unsupported','unrelated_property',n1,'The current n1 copied origin and temporal fields describe provenance. They do not state how the red content is presented, organized or accessible in the recorded O role.'),
 ('unsupported','unrelated_property',k1,'The candidate k1/n0 command estimate describes a conditional source process at future time 13. It is neither current n1 nor a qualifying organization/presentation/access status even for its own candidate node.'),
 ('unsupported','unrelated_property',[red],'Red (0,2) is content identification and location only; no separate qualifying status is attributed for D.'),
 ('unsupported','unrelated_property',n1,'The origin supplies record history and represented time but no organization/access status. Keeping it separate from content does not make it a D witness.'),
 ('supported','global_convention',[head],'The first header clause literally states grid-axis directions in display scope. Its generic time is retained as unclear, and no source-specific content property is inferred.'),
 ('supported','source_specific',n3,'The origin names an analytic history-estimation process and its feedback horizon at current represented time 12. It supports auxiliary source/process distinction while the model_organization scope label supplies no organization-quality evidence.'),
 ('unsupported','unrelated_property',[red],'The binding quote has compatible red current n1 content, but no O or D status, so it cannot ground complete COD.'),
 ('unsupported','unrelated_property',n1+[red],'The full text binds the outside red peak operand to the identical current/n1 prefix at represented time 12. That supplies content/provenance correspondence only; copied origin remains unrelated to qualifying O/D status for COD.')]
er=[dict(evidence_path=x['path'],grounding=g,source_meaning=s,properties=ps,notes=n) for x,(g,s,ps,n) in zip(r['evidence'],es)]
br=[dict(binding_path=r['bindings'][0]['path'],feature_witnesses=[dict(feature='C',evidence_paths=[r['evidence'][0]['path']]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[r['evidence'][7]['path'],r['evidence'][8]['path']],assessment='unsupported',notes='C/red current n1 is compatible with the own coordinate and origin quotations at represented time 12. The current origin is provenance, not an O/D status. The other O candidate is k1/n0 at future time 13, differs in node/time, and still only describes command-estimate provenance. Neither can supply a compatible qualifying O witness; the D candidates likewise fail. The original explicit COD therefore remains unsupported without relabeling it.')]
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['Current copied origin and future candidate-estimate origin differ in node and represented time; they cannot jointly certify current red n1 organization.','Source/process provenance is substituted for presentation/access status in the original O/D/COD rationale.','The current n1 red content is copied_W11 by indexed origin; analytic panel labeling is insufficient to claim analytic-source content.','The global orientation convention has unclear time and is not promoted to model appearance or an attention effect.'],notes='Full 5046-character report read before source interpretation. All 12 original statuses, all 9 evidence positions and the single COD binding are retained and reviewed with exact property operands. No source repair or semantic upgrade is performed.')
print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())))
