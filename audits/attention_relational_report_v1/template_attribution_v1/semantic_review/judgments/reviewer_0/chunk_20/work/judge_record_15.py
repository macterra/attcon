import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_20.json'),'dce8ec6b7058f0946a47c90706ae9815542e2ec0f0471fc6e85f01281cf64d0d');r=c['records'][15]
assert r['id']=='s1311_w00_t12_ordinary_A_only_current_donor_template'
def p(i,path,role,reason):
 u=w.source_view(c,r['id'],['property_table',i],0,20)['value']
 return dict(unit_id=u['unit_id'],value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def origin(i,reason):
 return [p(i,[k],'status_operand' if k=='kind' else 'context_only',reason+' Exact '+k+' specifies origin or time, not appearance/access quality.') for k in r['property_table'][i]['value']]
cs=[
 ('supported','The red current n0 token explicitly names represented red content and its complete sixteen-position location set. C is warranted independently of any organization claim.'),
 ('unsupported','The only O witness, rows-down/columns-right, is a global axis convention. Assigning its referent to red n0 does not make it a content-specific spatial pattern, differentiated representation, presentation or access status beyond coordinates.'),
 ('unsupported','Current n0 origin is copied_prior_record from time 11 representing 12, alongside red coordinate content. This distinguishes provenance from coordinates, not content identification from a separately expressed organization/presentation/access status.'),
 ('supported','The complete report lists current and conditional coordinate/origin tokens but does not explicitly relate processing/allocation to qualifying presentation, emergence, organization or access. A command estimate label and changed coordinate set alone do not warrant A.'),
 ('supported','The header explicitly fixes row and column orientation in display scope, supporting basic spatial relation.'),
 ('supported','No separate uncertainty assertion about content identity/location is stated. Multiple peak positions and estimate labels alone do not assert uncertainty.'),
 ('supported','The report contains no answer confidence rating, probability of an answer, or verbal confidence judgment.'),
 ('supported','No success or recovery expectation is stated; conditional peak coordinates do not give an outcome-success forecast.'),
 ('supported','The command_estimate origin for conditional k0/n0, with feedback through 11 and represented time 13, explicitly identifies a source processing provenance rather than a completed record.'),
 ('supported','There is no assertion of feeling, experience, consciousness or perceptual appearance in the complete report.'),
 ('unsupported','COD has red current n0 content, a global axis convention and copied provenance. The latter two do not supply O and D, so their co-occurrence and shared annotation labels are insufficient.'),
 ('supported','C is explicit but no qualifying A relation is stated, supporting the original absent CA status.')]
cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=n) for x,(a,n) in zip(r['components'],cs)]
red=p(4,[],'content_operand','The complete sixteen-position peak set identifies red current n0 location content. Indexed origin resolves to copied_W11 rather than the ordinary source indicated by the panel label.')
head=p(0,[],'context_only','The first clause supplies only the quoted row-down/column-right convention. The remaining header clause announces current/conditional inventory; no source-specific O content is assigned to the whole token.')
n0=origin(2,'The quoted current n0 copied origin has record_time 11 and represented_time 12.')
k0=origin(26,'The quoted candidate k0/n0 command estimate has feedback_through 11 and represented_time 13.')
es=[
 ('supported','source_specific',[red],'The exact current red n0 location set grounds C. Current represented time is preserved; the coordinate inventory makes no felt-appearance claim.'),
 ('unsupported','global_convention',[head],'The quote gives a grid convention. It does not name red n0, and the annotation cannot supply that missing relation or turn display orientation into model organization.'),
 ('unsupported','unrelated_property',n0,'Copied prior origin and the record/represented-time distinction concern source history, not organization, presentation or access status for current n0. The recorded D role is unsupported.'),
 ('unsupported','unrelated_property',[red],'The red sixteen-position set identifies content. No separately attributed presentation/access status occurs in this quotation; coordinates alone are not D.'),
 ('supported','global_convention',[head],'The first header clause explicitly states spatial orientation for the grid in display scope. It remains a global convention, not an A-specific effect.'),
 ('supported','source_specific',k0,'The origin kind explicitly identifies a command estimate at candidate k0/n0, with feedback and represented-time fields marking the conditional future. This supports auxiliary process provenance, not experienced access or actual execution.'),
 ('unsupported','unrelated_property',[red],'The binding has a literal C content operand at current red n0 but no O or D status in this coordinate quotation.'),
 ('unsupported','global_convention',[head,red],'The full report places the red n0 coordinate list under the global axis convention; the red operand is outside the quoted header and linked only as one entry using those coordinates. That establishes coordinate interpretation, not content-specific model organization or a COD status. The annotated red referent is not literal in the header.'),
 ('unsupported','unrelated_property',n0+[red],'The complete report binds the red peak operand outside this quote to the same current/n0 prefix. This shares content and represented time 12 with the origin, but origin remains copied-record provenance rather than the status needed for COD.')]
er=[dict(evidence_path=x['path'],grounding=g,source_meaning=s,properties=ps,notes=n) for x,(g,s,ps,n) in zip(r['evidence'],es)]
br=[dict(binding_path=r['bindings'][0]['path'],feature_witnesses=[dict(feature='C',evidence_paths=[r['evidence'][0]['path']]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[r['evidence'][6]['path'],r['evidence'][7]['path'],r['evidence'][8]['path']],assessment='unsupported',notes='Current red n0 location evidence supplies compatible C at represented time 12, and the node-wide origin shares that prefix. O proposes only the generic axis convention: it has no literal red referent and gives coordinate interpretation, not a qualifying model-organization status. D proposes copied provenance plus coordinates, which do not distinguish identification from organization/access. No compatible O/D witnesses therefore exist. Candidate k0/n0 command-estimate provenance is future 13 and is not substituted into this current COD binding.')]
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['The O and COD annotations assign a red n0 referent and model_organization scope to a generic axis convention that describes only grid coordinates.','The D and COD rationale substitutes a provenance status for organization/presentation/access status.','Current red n0 is copied_W11 according to indexed origin; the ordinary panel name does not make it ordinary-source content.','Future conditional command estimates are not executed branches or a report of attention-related appearance.'],notes='Complete 4110-character report read first. All 12 original component statuses, 9 evidence positions including duplicate quotations, and the single COD binding are preserved and reviewed. No semantic upgrade, source edit or annotation repair is performed.')
print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())))
