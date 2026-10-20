import sys,json,pathlib
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
C=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_18.json');SHA='00f7c30a0463a31224f1d86edb5794dd76a4b87538d05fbddbb568b7c13d2ac1'
c=h.bound_chunk(C,SHA);r=c['records'][0];work=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_18/work')
units={p['unit_id']:p for p in r['property_table']}
def prop(uid,path,role,reason):
 v=units[uid]['value']
 for k in path:v=v[k]
 return dict(unit_id=uid,value_path=path,value=v,semantic_role=role,reason=reason)
def red():return prop('u0094',[],'content_operand','Complete red candidate k2/n3 peak set identifies represented red content; same full-text p0/candidate/k2/n3 key binds it to the separately printed origin, whose represented_time is 13.')
def blue():return prop('u0096',[],'content_operand','Complete blue candidate k2/n3 peak set identifies represented blue content. Its matching key binds it to the same origin for time 13; this is another content location list, not red presentation status.')
def origin():return [prop('u0092',['kind'],'status_operand','The shared p0/candidate/k2/n3 key binds command_estimate provenance to the peak set outside this token; this generic estimator label does not express presentation or access.'),prop('u0092',['feedback_through'],'context_only','The estimator uses feedback through 11; feedback cutoff is not an attentional processing effect.'),prop('u0092',['represented_time'],'context_only','The shared key explicitly assigns the conditional peak record to represented time 13, contrary to the annotation current time at decision 12.')]
reviews=[
('supported','The red and blue k2/n3 peak sets identify represented contents and locations. C is explicit; the annotation current time is inaccurate because the matching origin represents 13, a conditional future.'),
('unsupported','The origin says command_estimate with feedback through 11 and represented_time 13. Together with a coordinate list this is provenance and a forecast, without an expressed organized pattern or differentiated access/presentation status.'),
('unsupported','A separate origin record is metadata, and red/blue peak locations are content attributes. Neither supplies a separately expressed organization/presentation/access status; generic estimate labels cannot establish D.'),
('unsupported','Candidate k2 and command_estimate name a conditional estimator record. The text states no allocation/content-processing condition related to red organization, emergence or access; no actual A-content relation is identified.'),
('supported','The opening sentence explicitly defines row-down and column-right grid orientation.'),
('unsupported','The full report includes many complete tied peak sets (for example p0 current n1 and n2) and multi-peak k2/n3 sets. Under the candidate-peaks header these represent unresolved location alternatives; a categorical absent location-uncertainty annotation misses that content.'),
('supported','The report prints origins and peak coordinates without an answer-confidence assertion or numerical answer confidence.'),
('supported','Conditional future peaks are not recovery or success predictions; no recovery/success criterion or likelihood is stated.'),
('supported','The text distinguishes copied_prior_record provenance with record_time 11 from command_estimate based on feedback through 11. This supports provenance/process distinction without supporting O/D/A. The candidate evidence time should be future_forecast.'),
('supported','No first-person experiential or felt-appearance assertion occurs in the complete report.'),
('unsupported','The k2/n3 red content is identified, but provenance and blue coordinates do not supply O and D; the binding current time also conflicts with represented_time 13.'),
('unsupported','The shared key links red coordinates to estimator provenance, without a content-processing condition related to organization/access. C alone and a command label do not establish CA.')]
components=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=n) for x,(a,n) in zip(r['components'],reviews)]
# Explicit per-position decisions, preserving duplicate quotations as separate reviews.
spec=[
('unsupported','source_specific',[red()]+origin(),'The quote identifies red content in the analytic conditional record, but its recorded current time conflicts with the same-key origin time 13; it is not a current-state witness.'),
('unsupported','source_specific',[blue()]+origin(),'Blue content is identified in the analytic conditional record; the recorded current time is incompatible with the same-key origin represented_time 13.'),
('unsupported','source_specific',origin()+[red()],'The command_estimate origin concerns the same red candidate via the full-text key. Generic provenance and forecast timing do not express red organization or present access.'),
('unsupported','source_specific',[red()]+origin(),'The four red peak coordinates enumerate alternatives; they do not describe a constituent pattern or access status. The origin outside this overlap binds them to a forecast for 13.'),
('unsupported','source_specific',origin()+[red()],'Same-key provenance is separately printed but is not a separately attributed organization/presentation/access status for red.'),
('unsupported','source_specific',[red()]+origin(),'Red peak locations are identified content attributes; printing them apart from generic estimator provenance does not establish status differentiation.'),
('unsupported','source_specific',origin()+[red()],'The source is analytic command_estimate, not an explicit allocation or processing effect on red organization/access. The full-text shared key establishes record identity only.'),
('unsupported','source_specific',[red()]+origin(),'The coordinate set has no processing-relation predicate. Its same-key origin is generic conditional estimator metadata, not an actual A-content property.'),
('supported','global_convention',[prop('u0000',[],'context_only','The quoted initial sentence is the orientation prefix of this fixed renderer header; the remaining header marks conditional candidate semantics.')],'The row/column convention is explicit in the partial header overlap. It supplies basic orientation, without source-specific attention content.'),
('supported','source_specific',[prop('u0002',['kind'],'status_operand','copied_prior_record is explicit provenance for p0/current/n0.'),prop('u0002',['record_time'],'context_only','The copied record originates at time 11.'),prop('u0002',['represented_time'],'context_only','The copied record is represented for current time 12; record time and represented time differ.')],'The indexed origin class is copied_W11. The record explicitly reports a copied prior origin; past_observed applies to record origin, not fresh observation at represented time 12.'),
('unsupported','source_specific',origin(),'command_estimate with feedback through 11 supports process provenance, but this evidence is annotated current while its represented_time is 13, a conditional forecast.'),
('unsupported','source_specific',[red()]+origin(),'As COD content evidence this identifies red, but the binding current time is wrong; neither this list nor its linked generic origin supplies O/D.'),
('unsupported','source_specific',origin()+[red()],'The linked command_estimate provenance is not organization status. Its time 13 also conflicts with the current COD binding.'),
('unsupported','source_specific',[blue(),red()]+origin(),'The full text places blue and red in the same conditional k2/n3 record, but blue coordinates are another content attribute, not red status differentiation; all concern forecast time 13.'),
('unsupported','source_specific',[red()]+origin(),'The red peak record is a forecast for 13 under its same-key origin; it contains no processing-condition-to-organization/access relation for CA.'),
('unsupported','source_specific',origin()+[red()],'This is analytic conditional provenance. It establishes no allocation effect or relation to red organization/access and cannot serve as the CA processing witness.')]
evidence=[dict(evidence_path=e['path'],grounding=g,source_meaning=s,properties=p,notes=n) for e,(g,s,p,n) in zip(r['evidence'],spec)]
bindings=[]
for b,n in zip(r['bindings'],['Same p0/candidate/k2/n3 key establishes content identity, but O and D have no qualifying feature evidence: origin is generic provenance and blue is a coordinate list. The recorded current time contradicts represented_time 13.','C and A candidate quotes share p0/candidate/k2/n3, but no compatible A witness expresses a processing/allocation relation to red organization/access. The origin instead denotes a conditional estimate for 13, not the recorded current state.']):
 features=[]
 for f in b['required_features']:
  paths=[e['path'] for e in r['evidence'] if e['component_id']=='component:/features/'+f] if f=='C' else []
  features.append(dict(feature=f,evidence_paths=paths))
 bindings.append(dict(binding_path=b['path'],feature_witnesses=features,binding_evidence_paths=[e['path'] for e in r['evidence'] if e['binding_id']==b['binding_id']],assessment='unsupported',notes=n))
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=components,evidence_reviews=evidence,binding_reviews=bindings,additional_concerns=['Candidate records represent time 13 conditional forecasts, not current time 12 or executed branches.','Bare maximal-cell sets and command_estimate metadata are overinterpreted as O/D/A; multi-peak alternatives may be missed location uncertainty.','No perceptual-appearance or felt-truth endpoint is established.'],notes='Complete 9828-character report inspected before evidence; every original component, evidence position and binding reviewed. Exact source properties distinguish analytic conditional records from the copied W11 origin. No labels repaired.')
raw=(json.dumps(out,ensure_ascii=True,indent=2)+'\n').encode();print(h.save_first_record(c,r['id'],work,raw))
