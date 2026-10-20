import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_20.json'),'dce8ec6b7058f0946a47c90706ae9815542e2ec0f0471fc6e85f01281cf64d0d');r=c['records'][12]
assert r['id']=='s1311_w01_t12_A_A_only_current_donor_template'
def p(i,path,role,reason):
 u=w.source_view(c,r['id'],['property_table',i],0,20)['value']
 return dict(unit_id=u['unit_id'],value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def origin(i,prefix):
 return [p(i,[k],'status_operand' if k=='kind' else 'context_only',prefix+' Exact '+k+' is provenance/temporal context; it makes no claim about appearance or access quality.') for k in r['property_table'][i]['value']]
spec=[
 ('supported','The current n1 and n2 red four-position sets and candidate k0/n1 red singleton explicitly identify represented red content and positions. C is warranted; the model_access scope annotation does not itself establish an access-status report.'),
 ('unsupported','Rows-down/columns-right describes a global coordinate convention. The n1 copied_prior_record origin describes provenance, not how red is organized, presented or accessible. Neither witness supplies O beyond coordinate inventory.'),
 ('unsupported','The red n1 positions and its copied origin are distinct content/provenance fields. They do not separately attribute an organization/presentation/access status to red, so the original explicit D rationale substitutes origin for status.'),
 ('supported','Candidate command estimates are listed, but no processing/allocation condition is explicitly related to organization, emergence, presentation or access of the same content. Coordinate changes and command labels alone leave A absent.'),
 ('supported','The row-down/column-right clause explicitly gives basic display spatial orientation.'),
 ('unsupported','The cited history_estimate token states an estimation source and feedback horizon, but does not explicitly say n3 identity or location is uncertain. Estimation process alone is not an uncertainty assertion.'),
 ('supported','No answer confidence value or verbal confidence report is present in the complete coordinate/origin inventory.'),
 ('supported','The conditional future peak entries make no recovery or success claim. Candidate coordinates are not outcome-success forecasts.'),
 ('supported','Current copied-record provenance and candidate command-estimate provenance are explicitly named, with distinct temporal fields. This supports the auxiliary source/process distinction without claiming model access quality.'),
 ('supported','The text gives no experiential or consciousness assertion; represented red/blue labels do not assert felt perception.'),
 ('unsupported','Current n1 red coordinates and copied origin have a common prefix/time, but qualifying O and D are missing. Grid orientation cannot be rebound from display convention into red model_access status.'),
 ('supported','There is identified content but no qualifying A relation, so absent CA is warranted.')]
cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=n) for x,(a,n) in zip(r['components'],spec)]
red1=p(10,[],'content_operand','The complete four-position red current n1 set identifies location content. Indexed origin resolves to copied_W11 despite the A panel label.')
red2=p(16,[],'content_operand','The complete four-position red current n2 set identifies different location content; indexed origin is copied_W11, not an A output inferred from the panel label.')
future=p(34,[],'content_operand','The red candidate k0/n1 singleton identifies conditional future location content. Indexed provenance is A source_conditional.')
header=p(0,[],'context_only','Only the row-down/column-right first clause is the quoted orientation witness. The containing header is a global renderer convention; its remaining clause is inventory context.')
n1=origin(8,'Current n1: copied_prior_record from record_time 11, represented_time 12.')
n0=origin(2,'Current n0: copied_prior_record from record_time 11, represented_time 12.')
n3=origin(20,'Current n3: history_estimate, feedback_through 11, represented_time 12.')
k0=origin(32,'Candidate k0/n1: command_estimate, feedback_through 11, represented_time 13.')
es=[
 ('supported','source_specific',[red1],'The quotation names red positions for current n1 at represented time 12. C is directly grounded, although a model_access presentation status does not follow from these coordinates.'),
 ('supported','source_specific',[red2],'The quotation names red positions for current n2 at represented time 12. These are content attributes, not appearance or accessibility qualities.'),
 ('supported','source_specific',[future]+k0,'The same candidate k0/n1 prefix in the full report binds the adjacent origin fields to the quoted red singleton; those origin operands lie outside this quote and explicitly establish represented_time 13. The singleton remains a conditional future C witness, not a current result.'),
 ('unsupported','global_convention',[header],'Grid orientation is explicit in display scope but gives no content-specific organization, presentation or access status. Its exact text cannot establish the stronger O role.'),
 ('unsupported','unrelated_property',n1,'The copied prior record kind and record/represented times specify provenance of current n1. They are unrelated to the recorded model_access O role: no accessible/inaccessible or differentiated presentation status is stated.'),
 ('unsupported','unrelated_property',[red1],'The red current n1 four-position set supplies C only; coordinate alternatives alone do not separately attribute an organization/access status for D.'),
 ('unsupported','unrelated_property',n1,'Origin is distinct from coordinates, but copied-record provenance is not the separately stated organization/access status required for D. The common current/n1 prefix does not change its meaning.'),
 ('supported','global_convention',[header],'The row/column orientation clause is a literal basic spatial convention in display scope. The second header clause adds no source-specific property.'),
 ('unsupported','unrelated_property',n3,'History_estimate identifies the process producing an n3 estimate and its feedback/time context. It does not explicitly quantify or assert identity/location uncertainty; no uncertainty is inferred from a generic estimate label.'),
 ('supported','source_specific',n0,'The origin kind and two times explicitly mark current n0 as a copied earlier record. Indexed provenance is copied_W11. This supports source/process distinction only, not accessible content quality.'),
 ('supported','source_specific',k0,'Command_estimate with feedback through 11 and represented time 13 explicitly identifies the conditional A estimation process at k0/n1. Its model_access scope label cannot certify access status or execution.'),
 ('unsupported','unrelated_property',[red1],'The binding quote identifies current red n1 positions, but offers no O or D status. A coordinate set alone cannot complete COD.'),
 ('unsupported','unrelated_property',n1+[red1],'The full report binds the red operand outside this origin quotation to the same current/n1 prefix at represented time 12. That connection establishes content/provenance identity only; it does not supply model access organization or differentiation for COD.')]
er=[dict(evidence_path=x['path'],grounding=g,source_meaning=s,properties=ps,notes=n) for x,(g,s,ps,n) in zip(r['evidence'],es)]
br=[dict(binding_path=r['bindings'][0]['path'],feature_witnesses=[dict(feature='C',evidence_paths=[r['evidence'][0]['path']]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[r['evidence'][11]['path'],r['evidence'][12]['path']],assessment='unsupported',notes='The two own quotes share current/n1 and represented time 12, and the red-positions C evidence is compatible. The available O orientation evidence is about the grid in display scope; the remaining O and D proposals concern copied record provenance. Neither expresses a model_access status for red n1. There are therefore no compatible qualifying O or D witnesses. The n2 red and k0/n1 future red entries are not substituted for this current n1 binding.')]
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['The original O/D/COD rationale treats provenance as organization/access status; the report supplies no separate qualifying access status.','The original uncertainty witness is a history_estimate label, which gives source process rather than explicit location/identity uncertainty.','Current red n1/n2 peak sets are indexed copied_W11 content even in the A panel. Candidate k0/n1 is an A conditional estimate; its existence alone does not establish an A attention relation.','Future candidate content is preserved as future; no execution, enhancement or perceptual-appearance claim is inferred.'],notes='Full 3390-character report read first. All 12 components, all 13 evidence positions and the single COD binding retain original paths/statuses. Property values are exact bound token operands; no source-model meaning is assigned from panel label alone.')
print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())))
