import session as s
r=s.c['records'][5]
def p(n,path,role,reason):
 u=r['property_table'][n];return dict(unit_id=u['unit_id'],value_path=path,value=s.w.resolve(u['value'],path),semantic_role=role,reason=reason)
def pk(n,why):return p(n,[],'content_operand',why)
def og(n,why):
 v=r['property_table'][n]['value']; paths=[]
 for k,x in v.items():
  if isinstance(x,dict):paths.extend([[k,z] for z in x])
  else:paths.append([k])
 return [p(n,x,'context_only',why) for x in paths]
def ev(path,g,m,pr,why):return dict(evidence_path=path,grounding=g,source_meaning=m,properties=pr,notes=why)
cn=[
('features','C','supported','The current red n0 singleton and candidate red n3 singleton identify content. The latter witness is mis-timed current, although this does not remove the explicit content in the full text.'),
('features','O','unsupported','Copied-prior and inherited-current provenance and global coordinate orientation do not express content organization beyond coordinate lists or an access/presentation status. Candidate represented_time 13 also contradicts its current evidence time.'),
('features','D','unsupported','The n0 red coordinate and separate copied-record origin distinguish content from provenance only. They do not independently attribute an organization/access/presentation status.'),
('features','A','supported','k0/k1/k2 conditional command estimates occur without an explicit relation from processing/allocation to organization, emergence, access or presentation of the content.'),
('auxiliary','basic_spatial_relation','supported','The opening convention specifies downward-increasing rows and rightward columns.'),
('auxiliary','identity_or_location_uncertainty','supported','Many maxima are listed for n1/n2, but no identity/location uncertainty claim is expressed.'),
('auxiliary','answer_confidence','supported','Neither the coordinate sets nor origin fields contain an answer-confidence judgment.'),
('auxiliary','recovery_or_success_forecast','supported','Future candidates contain predicted coordinates, without any recovery/success proposition.'),
('auxiliary','source_process_distinction','supported','The text explicitly distinguishes copied_prior_record, history_estimate, inherited_current and command_estimate origins. The cited candidate has time 13; its evidence entry incorrectly calls this current.'),
('auxiliary','experiential_assertion','supported','There is no subjective experience or perceptual-appearance assertion.'),
('composites','COD','unsupported','The current n0 red singleton gives C only; neither its origin line nor its repetition gives O/D. Metadata separation is not the required status differentiation.'),
('composites','CA','supported','Explicit content is present but no qualifying content-processing/attention relation is stated.')]
cr=[dict(component_path=[a,b],original_status=r['canonical_first_annotation'][a][b]['status'],assessment=x,notes=n) for a,b,x,n in cn]
cur=[pk(4,'Current n0 red singleton is the content operand.'),*og(2,'The shared p0/current/n0 prefix binds this copied-record provenance and represented time to the red singleton outside whichever of the two adjacent tokens is quoted.')]
inh=[*og(26,'The nested fields explicitly distinguish the inherited record at 13 from its copied W record at 11 representing 12; no containing object is treated as an undifferentiated field.'),pk(28,'The same p0/candidate/k0/n0 prefix binds the red singleton to this inheritance outside the origin quote.'),pk(30,'The same prefix also binds the blue singleton to this inheritance outside the origin quote.')]
cand=[pk(46,'The k0/n3 red singleton is the candidate content operand.'),*og(44,'Outside the red quote, the matching p0/candidate/k0/n3 prefix binds command_estimate, feedback through 11 and represented time 13 to that content.')]
head=[p(0,[],'context_only','The header contains the orientation convention plus candidate introduction; only its row/column clause is the quoted meaning.')]
er=[
ev(['features','C','evidence',0],'supported','source_specific',cur,'Identifies current n0 red at (0,3), with copied W11 provenance resolved by the index.'),
ev(['features','C','evidence',1],'unsupported','source_specific',cand,'The red singleton is identified, but this evidence records current time. The adjacent origin and conditional candidate prefix instead give represented time 13; the recorded time binding is unsupported.'),
ev(['features','O','evidence',0],'unsupported','source_specific',cur,'Copied-prior provenance at represented time 12 does not establish organization, presentation or access of n0 red.'),
ev(['features','O','evidence',1],'unsupported','source_specific',inh,'Inheritance from copied W11 is provenance, not an O description. Its outer represented time 13 also contradicts the evidence current time; the nested 12 belongs to inherited origin.'),
ev(['features','O','evidence',2],'unsupported','global_convention',head,'The display coordinate convention specifies orientation without a content-specific O relation.'),
ev(['features','D','evidence',0],'unsupported','source_specific',cur,'The n0 red singleton supplies identity/location only; neighboring provenance is not an independently attributed O status.'),
ev(['features','D','evidence',1],'unsupported','source_specific',cur,'The same-prefix origin/content association exists, but describes copying rather than the required content-status distinction.'),
ev(['auxiliary','basic_spatial_relation','evidence',0],'supported','global_convention',head,'The quote explicitly describes how the display axes increase.'),
ev(['auxiliary','source_process_distinction','evidence',0],'unsupported','source_specific',inh,'The quote genuinely distinguishes inherited_current from nested copied_prior_record, but the original current time misstates a candidate represented at 13. Provenance remains copied W11, not ordinary-source content.'),
ev(['composites','COD','bindings',0,'evidence',0],'unsupported','source_specific',cur,'The current n0 red singleton supports C but cannot by itself ground COD.'),
ev(['composites','COD','bindings',0,'evidence',1],'unsupported','source_specific',cur,'The copied-record origin supplies no O/D relation even though identity and time are compatible with the red singleton.'),
ev(['composites','COD','bindings',0,'evidence',2],'unsupported','source_specific',cur,'The separately retained duplicate red token is not evidence of separate_identity_and_status: the additional status is absent from its meaning.')]
br=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',j] for j in range(3)],assessment='unsupported',notes='All own evidence points to current p0/n0 red represented at 12, so the content/provenance prefix and time association is compatible. However no candidate feature witness for O or D qualifies: the inherited candidate is another view/time, the header is global convention and current origin is provenance only. The three own positions, including duplicate red, do not complete COD.')]
s.save(5,dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['Candidate evidence is recorded as current in C, O and source_process_distinction despite outer represented_time 13. Nested inherited-origin time 12 cannot change that time.','The ordinary panel includes copied W11; attribution follows indexed origin rather than the panel label.','No coordinate list or origin kind establishes perceptual appearance.'],notes='Full report, all 12 components, 12 distinct evidence positions and the one binding were inspected; original statuses and recorded time errors are preserved rather than repaired.'))
