import sys,json
sys.path.insert(0,'audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_12/work')
sys.argv=['','4','noop'];import review_tools as t
r=t.r
notes=[
('supported','Red at (0,2) and blue at (1,0) identify represented contents and their coordinate attributes at p0/current/n1.'),
('unsupported','copied_prior_record with record_time 11 and represented_time 12 describes provenance and temporal indexing. It states no spatial/constituent organization or differentiated access/presentation status of red or blue beyond their coordinates.'),
('unsupported','A separate provenance entry does not separately attribute an organization/presentation/access status. Red coordinates plus copied-record metadata do not satisfy D.'),
('supported','The command_estimate entries and conditional k labels provide candidate peak coordinates, with no explicit processing-condition relation to organization or access.'),
('supported','Rows down and columns right explicitly specify a layout spatial orientation convention; this is not a content-specific O claim.'),
('supported','No prose explicitly attributes uncertainty about identity or location; multiple peak coordinates alone are not an uncertainty assertion.'),
('supported','No confidence in an answer is stated anywhere in this coordinate/origin inventory.'),
('supported','Candidate peaks at time 13 forecast coordinates, not recovery or success likelihood.'),
('supported','copied_prior_record and record_time 11 versus represented_time 12 explicitly distinguish the stored source record from the represented current node.'),
('supported','No experience or felt appearance is asserted by the report.'),
('unsupported','Both COD bindings join coordinates to provenance metadata, leaving O and D without semantic witnesses even though n1 and current indexing match.'),
('supported','C is present but there is no qualifying A relation, so absent CA is warranted.')]
components=[dict(component_path=c['path'],original_status=c['recorded_status'],assessment=a,notes=n) for c,(a,n) in zip(r['components'],notes)]
def prop(u,path,role,reason):
 v=next(p['value'] for p in r['property_table'] if p['unit_id']==u)
 for k in path:v=v[k]
 return dict(unit_id=u,value_path=path,value=v,semantic_role=role,reason=reason)
def origin(u):
 return [prop(u,['kind'],'context_only','The copied_prior_record kind describes source provenance, not organization or current accessibility.'),prop(u,['record_time'],'context_only','The source record is indexed at 11.'),prop(u,['represented_time'],'context_only','The report indexes this copied entry as representing time 12; temporal indexing does not establish O/D.')]
red=prop('u0010',[],'content_operand','Complete red peak set for p0/current/n1, from copied_W11 provenance.')
blue=prop('u0012',[],'content_operand','Complete blue peak set for the same p0/current/n1, from copied_W11 provenance.')
evidence=[]
for j,e in enumerate(r['evidence']):
 if j in [0,1]:
  ps=[red if j==0 else blue];a='supported';meaning='source_specific';n='The complete peak set identifies '+('red' if j==0 else 'blue')+' content at p0/current/n1, current represented time 12. Indexed provenance is copied_W11, not analytic A-content.'
 elif j==5:
  ps=[prop('u0000',[],'context_only','The quoted first sentence of the header specifies row/column orientation only; the remaining header is a candidate convention.')];a='supported';meaning='global_convention';n='The quote supports layout orientation in display scope and current convention, with no source-specific organization or A relation.'
 elif j==6:
  ps=origin('u0002');a='supported';meaning='source_specific';n='At n0, copied source kind and record_time 11 versus represented_time 12 establish source/represented-record distinction. This is copied_W11 provenance despite analytic panel naming.'
 else:
  ps=origin('u0008')+[red if j not in [9,10] else blue];a='unsupported';meaning='source_specific';n='For p0/current/n1, the full report places origin metadata beside the '+('blue' if j in [9,10] else 'red')+' peak entry. That textual adjacency binds the operand outside this quote to this node/time, but kind/record_time/represented_time supply provenance, not an organization/access status. Neither coordinates nor provenance establish the recorded '+('D' if j in [3,4] else 'O' if j==2 else 'COD')+' role. Both operands resolve to copied_W11; no A-content is implicated.'
 evidence.append(dict(evidence_path=e['path'],grounding=a,source_meaning=meaning,properties=ps,notes=n))
bindings=[]
for j,b in enumerate(r['bindings']):
 witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',j]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])]
 bindings.append(dict(binding_path=b['path'],feature_witnesses=witnesses,binding_evidence_paths=[e['path'] for e in r['evidence'] if e['binding_id']==b['binding_id']],assessment='unsupported',notes='The '+('red' if j==0 else 'blue')+' coordinate witness and origin quote concern p0/current/n1 at represented time 12, but coordinate and copied-record provenance are different attributes. Provenance has no stated organization/presentation/access consequence. No compatible O or D semantic witness exists, so the shared node/time cannot establish COD.'))
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=components,evidence_reviews=evidence,binding_reviews=bindings,additional_concerns=['Do not treat copied W11 peak content as analytic A-content because of panel source naming.','Conditional command labels and changed peak sets do not state an attention-to-organization/access relation.','Provenance status is not perceptual appearance or a differentiated organization/access status.'],notes='Reviewed full 5046-character report, canonical annotation, all 12 components, 11 evidence positions and both bindings. Source-specific operands here are copied_W11; no claim of felt truth or current execution is established.')
print(json.dumps(t.w.save_first_record(t.x,t.i,t.WORK,(json.dumps(out,indent=2)+'\n').encode())))
