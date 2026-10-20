import json,sys,pathlib
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_11.json','861514d2ea78d2073085b44a0042c46c17bc60a0205e9640bb087b5067d1d6a2');r=c['records'][0]
assert r['text_sha256']==__import__('hashlib').sha256(r['text'].encode()).hexdigest()
assert r['canonical_first_annotation']['unit_id']==r['unit_id']
x={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
judgments=[
('supported','Red and blue coordinate-set content at p0/current/n0 and red at n3 are explicitly identified; this establishes C without organization or appearance.'),
('unsupported','The n0 copied_prior_record and n3 history_estimate labels specify record provenance; the n0 red set only enumerates coordinates. Neither states a differentiated organization, presentation or access status under the frozen O rule.'),
('unsupported','At n3, red={(0,2)} identifies location and history_estimate identifies provenance. Their separate fields do not separately express organization/presentation/access status for that red content.'),
('supported','Conditional k0/k1/k2 peak and origin lists contain no allocation/content-processing condition explicitly related to organization or access. A is absent despite command_estimate labels.'),
('supported','The opening sentence explicitly fixes rows downward and columns rightward, a basic spatial orientation convention.'),
('supported','The report gives candidate peak sets but does not assert uncertainty about identity or location; multiple maxima alone do not supply that assertion.'),
('supported','No confidence in an answer is expressed; the report lists peaks and provenance, without answer confidence values or prose.'),
('supported','Conditional peaks for k0/k1/k2 do not assert recovery or success probability, so this auxiliary absence is warranted.'),
('supported','The inherited_current origin at k0/n1 explicitly retains a copied_prior_record origin, distinguishing record provenance. Its evidence time=current is inaccurate for a conditional represented_time=13 record.'),
('supported','No experience or awareness is asserted anywhere in the complete report.'),
('unsupported','The claimed n3 red COD binding has identified red location but no qualifying organization/access status or content-status differentiation; coordinate listing and history_estimate do not suffice.'),
('supported','C is present, but no content-processing-to-organization/access relation binds it to A. CA absence is warranted.')]
x['component_reviews']=[dict(component_path=p['path'],original_status=p['recorded_status'],assessment=j[0],notes=j[1]) for p,j in zip(r['components'],judgments)]
def prop(n,path,role,reason):
 u=r['property_table'][n];return dict(unit_id=u['unit_id'],value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def peaks(n,reason):return [prop(n,[],'content_operand',reason)]
def origin(n,reason):
 u=r['property_table'][n];out=[]
 def walk(v,p):
  if isinstance(v,dict):
   for k,t in v.items():walk(t,p+[k])
  else:out.append(prop(n,p,'context_only',reason))
 walk(u['value'],[]);return out
p0=peaks(4,'Exact n0 red content set, at the same current-node prefix as origin u0002.')
p1=peaks(6,'Exact n0 blue content set, bound by the current-node prefix.')
p2=peaks(22,'Exact n3 red content at represented time 12; same n3 prefix binds origin u0020 outside this quote.')
o0=origin(2,'Each field specifies copied record provenance/time, not organization/access; n0 prefix links this origin to red u0004 and blue u0006.')
o3=origin(20,'These exact n3 provenance/time fields bind to red u0022 at the same prefix; history_estimate is not a differentiated access status.')
sp=origin(32,'Nested copied_prior_record at time 11 representing 12 is inherited into conditional time 13, not current; each field is cited separately rather than attributing the containing token wholesale.')
items=[
('supported','source_specific',p0,'C: this quote identifies red content by a four-coordinate set, at current n0. Indexed provenance is copied_W11, not the ordinary panel label.'),
('supported','source_specific',p1,'C: this separate evidence position identifies blue content at current n0; its copied_W11 provenance is explicit in the indexed origin.'),
('supported','source_specific',p2,'C: red at n3 is identified by its singleton coordinate, with source_current ordinary provenance.'),
('unsupported','source_specific',o0+p0+p1,'O: copied_prior_record, record_time=11 and represented_time=12 name provenance and time; binding to the n0 content outside the quote follows the exact n0 prefix but adds no organization/access statement.'),
('unsupported','source_specific',o3+p2,'O: history_estimate with feedback_through=11 and represented_time=12 is an origin label. The same-node red outside the quote is location content, not an organization/access witness.'),
('unsupported','source_specific',p0,'O: four maximal coordinates are a listing, not an expressed organized representation or accessibility status. The presentation_pattern attribute does not change the literal meaning.'),
('unsupported','source_specific',p2+o3,'D: red coordinate content and its same-node origin are separately listed but do not differentiate content from organization/presentation/access.'),
('unsupported','source_specific',o3+p2,'D: the origin fields do not specify an access/presentation status for n3 red. The binding to red outside the quote is only the common n3 prefix.'),
('supported','global_convention',[prop(0,[],'context_only','The header is an authored orientation convention; its full value is cited for the initial directional sentence only, not as an A-content property.')],'The opening 34 characters establish grid orientation. The remaining header describes conditional records; neither portion supplies source-specific content.'),
('unsupported','source_specific',sp,'This nested provenance distinguishes copied from inherited origin, but the recorded evidence time=current is incompatible with the conditional candidate record representing 13. It is copied_W11 provenance, not A content.'),
('unsupported','source_specific',p2+o3,'COD own evidence 0 identifies n3 red but supplies neither qualifying O nor D; its same-node origin outside the quote is provenance only.'),
('unsupported','source_specific',o3+p2,'COD own evidence 1 supplies n3 history_estimate provenance, not an organization/access status, despite compatible node/time with red.'),
('unsupported','source_specific',peaks(24,'Exact n3 blue coordinate set; this is blue content, not the recorded origin attribute.'),'COD own evidence 2 is blue location, mislabeled attribute=origin and incompatible with the binding red attribute as its status witness.')]
x['evidence_reviews']=[dict(evidence_path=e['path'],grounding=a,source_meaning=s,properties=p,notes=n) for e,(a,s,p,n) in zip(r['evidence'],items)]
x['binding_reviews']=[dict(binding_path=r['bindings'][0]['path'],feature_witnesses=[dict(feature='C',evidence_paths=[r['evidence'][2]['path']]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[e['path'] for e in r['evidence'][10:13]],assessment='unsupported',notes='C evidence /features/C/evidence/2 identifies current n3 red. The O n3 origin quote and both D quotes do not express organization/access; O coordinate evidence concerns n0 rather than n3. Blue binding evidence also has the wrong attribute. No compatible C/O/D conjunction exists for red/current n3/model_access.')]
x['additional_concerns']=['Origin and command_estimate labels must not become access/organization or attention witnesses merely because this packet comes from a model.','The source_process_distinction candidate evidence represents time 13, although its original evidence labels it current.','No perceptual appearance or felt truth follows from these location sets.']
x['notes']='Established original retained without status repair. All 12 components, 13 evidence-list positions and the single COD binding reviewed; repeated quotes remain separate. Current n0 content is copied_W11 while n3 is ordinary source_current, as indexed; header convention is not source-specific content.'
p=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_11/work/record0_input.json');p.write_text(json.dumps(x,indent=2)+'\n')
print(w.save_first_record(c,r['id'],p.parent,p.read_bytes()))
