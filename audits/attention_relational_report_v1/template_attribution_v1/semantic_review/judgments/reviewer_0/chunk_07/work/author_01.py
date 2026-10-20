import json,pathlib,subprocess
chunk='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_07.json'
r=json.load(open(chunk))['records'][1];w=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_07/work')
def prop(i,path,role,reason):
 t=r['property_table'][i];v=t['value']
 for k in path:v=v[k]
 return dict(unit_id=t['unit_id'],value_path=path,value=v,semantic_role=role,reason=reason)
def peaks(i,reason):return prop(i,[],'content_operand',reason)
def origin(i):
 return [prop(i,[k],'context_only','Exact provenance field for this node; source derivation and represented time alone do not state current access or organized presentation.') for k in r['property_table'][i]['value']]
cnotes=[
('supported','Red peak sets identify represented red-location content at n0 and n2; the copied prior provenance does not remove content identification.'),
('unsupported','The quoted copied_prior_record and history_estimate fields distinguish provenance, record time and estimate origin. They do not explicitly state current accessibility or organized presentation. The coordinate sets are insufficient by themselves for O.'),
('unsupported','Listing red coordinates separately from origin metadata distinguishes content from derivation provenance, but no separately expressed presentation/access status is established for red at n0.'),
('supported','The full report has command_estimate and inherited_current origins but no functional allocation/processing relation to organization, emergence or access of the red/blue contents. Arbitrary k0/k1/k2 labels cannot supply A.'),
('supported','The opening sentence explicitly gives row-down and column-right spatial mapping.'),
('unsupported','The header says conditional candidate peaks, which states branch conditionality rather than identity or location uncertainty. Multiple peak coordinates alone do not warrant treating that header as an uncertainty assertion.'),
('supported','No answer confidence or confidence number is stated anywhere in the full report.'),
('supported','Candidate represented times and command estimates give no claim of recovery or success likelihood.'),
('supported','Nested inherited_origin explicitly distinguishes a copied prior record at time 11 from its inherited-current use represented at time 13. This supports a provenance/process distinction, with the evidence time caveat below.'),
('supported','No first-person experience or consciousness assertion appears in the full report.'),
('unsupported','The n0 red location identifies content, but provenance does not supply the required model-access O and D witnesses; the third binding quote also names blue while its recorded attribute says red.'),
('supported','No compatible content-attention relation is stated, so the recorded absent CA is warranted.')]
components=[dict(component_path=c['path'],original_status=c['recorded_status'],assessment=a,notes=n) for c,(a,n) in zip(r['components'],cnotes)]
P0=peaks(4,'The full text shares p0/current/n0 between the red peak and adjacent origin token, binding this content to represented time 12 without turning provenance into access.')
P2=peaks(16,'The complete peak set is the represented red-location content for p0/current/n2, not an assertion of presentation quality.')
P3=peaks(22,'The full text shares p0/current/n3 between its history-estimate origin and this red-location peak; this explicit node identifier binds the otherwise unquoted content operand.')
E=[]
def ev(i,g,m,p,n):E.append(dict(evidence_path=r['evidence'][i]['path'],grounding=g,source_meaning=m,properties=p,notes=n))
ev(0,'supported','source_specific',[P0],'At p0/current/n0 this quote literally identifies red-location content at (0,0). Its indexed source is copied W11 despite the A panel label; no current access quality is inferred.')
ev(1,'supported','source_specific',[P2],'At p0/current/n2 the red peak set covers the complete grid. That remains identified represented location content; it supplies no organization or uncertainty claim by itself. Indexed provenance is copied W11.')
ev(2,'unsupported','source_specific',origin(2)+[P0],'The copied_prior_record kind, record_time 11 and represented_time 12 are source-specific provenance. Same-node adjacency binds them to the n0 red content outside this quote, but none states a model-access status.')
ev(3,'unsupported','source_specific',origin(20)+[P3],'history_estimate with feedback_through 11 and represented_time 12 names the source-specific estimator history. Same-node n3 text binds it to red content outside the overlap; derivation history alone does not establish current access.')
ev(4,'unsupported','source_specific',[P0]+origin(2),'The n0 red peak is content rather than a status differentiation. Its companion origin fields are bound by the repeated p0/current/n0 identifier in the full text, and still describe provenance rather than a separate access/presentation status.')
ev(5,'unsupported','source_specific',origin(2)+[P0],'The n0 origin quote separately gives copied-prior provenance and times. Although the n0 red token explicitly binds content outside the overlap, this does not distinguish an organization/access status from identified red content.')
ev(6,'supported','global_convention',[prop(0,[],'context_only','Authored header includes the quoted row/column directions as a fixed renderer convention.')],'The partial header overlap explicitly states row-down/column-right mapping. It is a global rendering convention, not a source-content relation or actual A operand.')
ev(7,'unsupported','global_convention',[prop(0,[],'context_only','Complete header retained; only the conditional-candidate sentence is quoted at this position.')],'The other partial overlap in the same header says conditional candidate peaks. This generic convention supplies neither location nor identity uncertainty, and no source-content operand.')
ev(8,'unsupported','source_specific',[
prop(32,['kind'],'context_only','The prospective node is inherited_current.'),
prop(32,['represented_time'],'context_only','Outer represented time is 13, so the annotation current time is not warranted.'),
prop(32,['inherited_origin','kind'],'context_only','Nested copied_prior_record names inherited provenance explicitly.'),
prop(32,['inherited_origin','record_time'],'context_only','The inherited record was made at time 11.'),
prop(32,['inherited_origin','represented_time'],'context_only','The inherited origin represents time 12, distinct from outer time 13.')],
'This quote supports an inheritance/provenance distinction from copied W11 to a prospective represented node. Its recorded current time misdescribes the outer represented time 13; model_access is not explicitly asserted by origin metadata.')
ev(9,'supported','source_specific',[P0],'The binding red peak is a literal C witness for n0 red at represented time 12, with copied-W11 provenance. It supplies no O or D witness.')
ev(10,'unsupported','source_specific',origin(2)+[P0],'The binding repeats the n0 copied-prior origin. The same n0 red content is explicitly bound outside this overlap, but provenance/times do not supply the claimed access status or COD.')
ev(11,'unsupported','source_specific',[peaks(6,'This is the complete blue peak set at n0, whose role differs from the binding red attribute.'),P0], 'The quote is p0/current/n0/blue at (0,2), while the recorded evidence attribute and COD binding say red. Shared node/time does not erase that attribute mismatch; neither peak set states organization/access status.')
b=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[e['path'] for e in r['evidence'][9:12]],assessment='unsupported',notes='C evidence 0 identifies n0 red, and the shared p0/current/n0 prefix binds its copied-prior origin to represented time 12. C evidence 1 concerns n2 and cannot serve this n0 binding. Both O origin quotes describe provenance rather than explicit access status; D has no independent such status. The last own quote describes blue although attribute red is recorded. No compatible model-access COD witness set exists.')]
y=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=components,evidence_reviews=E,binding_reviews=b,additional_concerns=['A panel ownership cannot be substituted for actual source: n0/n2 and candidate n1 inherited fields resolve to copied W11; the n3 history estimate is actual source-current A.','The source-process quote has outer represented_time 13 while its annotation records current; origin metadata does not establish claimed perceptual appearance.'],notes='Full 3930-character report, all 12 components, all 12 evidence positions and the original COD binding inspected. Source-specific classification identifies actual provenance operands even where the recorded semantic role is unsupported.')
p=w/'authored_01.json';p.write_text(json.dumps(y,indent=2)+'\n')
subprocess.run(['python3','scripts/attention_relational_template_review_workspace_v1.py','--chunk',chunk,'--chunk-sha256','75ae0b350529484b0168488f74a4d9270567afe75d8ec1c47cd93598dcf8527e','save-first-record','--id',r['id'],'--input',str(p),'--workspace',str(w)],check=True)
