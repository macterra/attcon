import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_20.json'),'dce8ec6b7058f0946a47c90706ae9815542e2ec0f0471fc6e85f01281cf64d0d')
r=c['records'][11]
assert r['id']=='s1311_w00_t12_ordinary_joined_prospective_donor_template'
def p(i,path,role,reason):
 u=w.source_view(c,r['id'],['property_table',i],0,20)['value']
 return dict(unit_id=u['unit_id'],value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def origin(i,reason):
 u=r['property_table'][i]['value']
 return [p(i,[k],'status_operand' if k=='kind' else 'context_only',reason+' This field records provenance or its temporal context, not appearance/access quality.') for k in u]
components=[
 ('supported','The red current n0 full peak set, red conditional k0/n0 singleton, and blue current n1 singleton name represented contents and location attributes. This warrants C without granting presentation quality.'),
 ('unsupported','The overall current/conditional-peaks heading labels an inventory; copied_prior_record identifies record provenance. Neither states a spatial/constituent pattern beyond coordinates or how red n0 is presented or accessible. A separate origin field is not itself the reference O status.'),
 ('unsupported','Red n0 has a coordinate peak set and a copied-record origin. This separates content from provenance, not identified content from organization/presentation/access status. No qualifying O status is separately attributed to red n0.'),
 ('supported','The report lists conditional command estimates and inherited current records, but no allocation/content-processing condition is explicitly related to a qualifying organization, emergence, presentation or access status. Command and peak changes alone do not establish A.'),
 ('supported','Rows increasing down and columns right explicitly specify spatial orientation for the display coordinate layout. This is a renderer convention, not source-specific model organization.'),
 ('supported','Multiple coordinate peaks are listed, but there is no separate assertion of uncertain identity or location, guessing, or confidence. The tied coordinate inventories are not labeled uncertainty.'),
 ('supported','There is no numerical answer confidence or verbal confidence statement in the complete report.'),
 ('supported','Conditional candidate peaks are represented future coordinates; no recovery probability, success expectation, or outcome forecast is stated.'),
 ('supported','The report explicitly distinguishes copied_prior_record, history_estimate through feedback 11, and command_estimate at represented time 13. This is a provenance/process distinction, not evidence that model appearance differs.'),
 ('supported','The complete report asserts no experience, feeling, consciousness, or perceptual appearance.'),
 ('unsupported','The current red n0 coordinate set and its copied origin share a referent, but neither header nor provenance supplies O and D under the frozen reference. Compatible referent/time alone cannot establish COD.'),
 ('supported','C is present but no qualifying attention/content-processing-to-organization/access relation is expressed; absent CA remains warranted.')
]
cr=[dict(component_path=v['path'],original_status=v['recorded_status'],assessment=a,notes=n) for v,(a,n) in zip(r['components'],components)]
full='The complete set of sixteen red peak positions is the represented location content for p0/current/n0 at time 12; indexed provenance resolves it to copied_W11, despite the ordinary panel label.'
red=p(4,[],'content_operand',full)
cand=p(28,[],'content_operand','The red peak singleton is bound to candidate k0/n0 by its full prefix; it represents a conditional forecast, not an executed or current result.')
blue=p(12,[],'content_operand','The blue singleton identifies p0/current/n1 location content at current represented time 12; its indexed origin is copied_W11.')
head=p(0,[],'context_only','Only the quoted header clause is relevant: it announces current and conditional candidate peak lists. The containing header is a global renderer convention; its axis clause does not become content organization.')
axis=p(0,[],'context_only','The first clause fixes row-down/column-right orientation. The rest of this header only announces the inventory; no source-specific content operand is supplied by the convention.')
n0=origin(2,'Exact fields of the quoted p0/current/n0 origin; kind=copied_prior_record and record_time=11 distinguish origin from represented_time=12.')
n3=origin(20,'Exact fields of the quoted p0/current/n3 history_estimate with feedback through 11 and represented_time=12.')
k0=origin(26,'Exact fields of the quoted p0/candidate/k0/n0 command_estimate with feedback through 11 and represented_time=13.')
specs=[
 ('supported','source_specific',[red],'Red, n0, current, and the entire sixteen-coordinate set are literal represented content. Current scope is an inventory at represented time 12, not a report of felt red appearance.'),
 ('supported','source_specific',[cand]+k0,'The candidate prefix binds the red singleton to the adjacent origin for the same k0/n0. The origin operand lies outside this quotation but the complete report places both under the identical candidate prefix, explicitly making represented_time=13 a forecast. This supplies C only.'),
 ('supported','source_specific',[blue],'The quoted blue current n1 singleton identifies content and location. The source meaning is the indexed copied W11 record, not ordinary-model output inferred from the panel name.'),
 ('unsupported','global_convention',[head],'The header at overall_report announces both current and conditional entries, so even its time range spans current and forecast. It gives no differentiated presentation or access status and is not a source-specific O witness.'),
 ('unsupported','unrelated_property',n0,'The quote gives provenance at p0/current/n0: copied record 11 representing 12. Those fields concern origin and time; no organization/presentation/access status is expressed for red. Indexed copied_W11 provenance does not cure the wrong semantic role.'),
 ('unsupported','unrelated_property',[red],'The complete red n0 coordinate set identifies alternatives/maximal positions. It does not independently express or distinguish any organization/presentation/access status. Coordinate content cannot be a D witness alone.'),
 ('unsupported','unrelated_property',n0,'A separate provenance token is present, but copied_prior_record and its two times distinguish record history from represented time, not content from a qualifying organization/access status. The recorded D role is unwarranted.'),
 ('supported','global_convention',[axis],'The row-down and column-right clause explicitly states a basic layout orientation in display scope at report time. The other header clause remains context only.'),
 ('supported','source_specific',n0,'The copied origin kind and record/represented times explicitly state source provenance. This supports the auxiliary source/process distinction; it does not support model_access quality or upgrade provenance into O.'),
 ('supported','source_specific',n3,'History estimate and feedback_through=11 identify the estimation process for current n3 representing 12. The quote supports provenance/process wording only; model_access is not independently certified by the origin label.'),
 ('supported','source_specific',k0,'Command estimate, feedback through 11, and represented time 13 explicitly mark conditional estimation at k0/n0. This is a future estimate/process distinction; no actual execution or experienced access follows.'),
 ('unsupported','unrelated_property',[red],'This binding quotation names the current red n0 content, but supplies neither O nor D. It is correctly located in the text and indexed to copied_W11, yet is insufficient as a complete COD witness.'),
 ('unsupported','unrelated_property',n0+[red],'The origin shares p0/current/n0 with the red peak set. The red operand is outside this quote and linked by the identical current/n0 prefix in the full report. Provenance and coordinate content remain distinct fields without a qualifying organization/access status, so COD is unsupported.')
]
er=[dict(evidence_path=e['path'],grounding=g,source_meaning=s,properties=ps,notes=n) for e,(g,s,ps,n) in zip(r['evidence'],specs)]
br=[dict(binding_path=r['bindings'][0]['path'],feature_witnesses=[dict(feature='C',evidence_paths=[r['evidence'][0]['path']]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[r['evidence'][11]['path'],r['evidence'][12]['path']],assessment='unsupported',notes='Both own quotations refer to p0/current/n0 and represented time 12, and C/red is compatible. The proposed O header refers to the overall mixed-time inventory; the other O/D proposal identifies copied record provenance. Neither expresses organization/presentation/access status for red n0. Thus no compatible O or D feature witness exists, despite shared prefix and exact quote grounding. No conditional red k0/n0 forecast or blue n1 coordinate is substituted into this current-red binding.')]
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['Origin provenance was substituted for organization/presentation/access status in the original O, D and COD rationale.','The generic header spans current and conditional future entries; its overall-report/current annotation does not make future forecasts current.','Source classes follow indexed origin: ordinary-panel copied records are W11 content; history and command estimates are ordinary-source current and conditional properties. No A-source content witness or perceptual-appearance claim is established.','Peak ties are preserved as listed coordinate sets. No uncertainty, felt clarity, or attentional enhancement is inferred from set size.'],notes='Complete report read before annotation and property inspection. All 12 original statuses, 13 evidence-list positions and the single COD binding are reviewed without changing the original annotation. This is a semantic first judgment on bound packet values; source geometry alone is not treated as semantic success.')
raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode()
print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,raw)))
