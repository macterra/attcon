import sys,json,hashlib,pathlib
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_13.json','cda7fcb6e29aa7076403202a25c76a764aab8ea322c91eaa90dc38fb06448bf0'); r=c['records'][0]
assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
judgments=[
('supported','The red content at p1/current/n0 is identified at (0,3); this warrants C as represented content, with no appearance inference.'),
('unsupported','The O witness is only the singleton coordinate {(0,3)}. Coordinates alone do not describe a constituent pattern, presentation or access status under the frozen O rule.'),
('unsupported','The completed_record origin identifies provenance and times, not a separately expressed organization/presentation/access status of red. Record origin is not the required content/status differentiation.'),
('supported','The full report lists current and candidate peaks and origins, with no explicit allocation or processing condition related to organization/presentation/access of the red content; A absence is warranted.'),
('supported','The grid header expressly states increasing rows down and columns right, supporting a basic spatial orientation convention.'),
('unsupported','Current and conditional candidate peaks follow describes temporal/candidate views; it does not assert uncertainty of identity or location. Multiple listed peak coordinates elsewhere do not make this generic header an uncertainty witness.'),
('supported','No answer-confidence judgment appears in this coordinate/origin inventory.'),
('supported','Conditional peak coordinates and native/estimated origins forecast positions, not recovery or success, so absence of a recovery/success forecast is warranted.'),
('supported','The p0/current/n0 origin expressly marks copied_prior_record, record_time 11, represented_time 12, distinguishing the represented current content from its prior-record source. This is provenance distinction, not O/D.'),
('supported','No first-person experience, feeling or perceptual-appearance claim occurs in the full peak/origin listing.'),
('unsupported','The same p1/current/n0 current-time referent links the red coordinates and completed-record origin, but O and D do not follow from coordinate and provenance fields. Therefore explicit COD lacks required semantic witnesses.'),
('supported','C is identifiable, but no explicit A relation or CA binding is stated; absent CA is warranted.')]
cr=[]
for comp,(assessment,notes) in zip(r['components'],judgments):cr.append(dict(component_path=comp['path'],original_status=comp['recorded_status'],assessment=assessment,notes=notes))
def prop(uid,path,role,reason):
 u=next(x for x in r['property_table'] if x['unit_id']==uid);return dict(unit_id=uid,value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def red():return [prop('u0100',[],'content_operand','Exact red peak set for p1/current/n0; the token labels red and its current referent explicitly.')]
def orig(uid):return [prop(uid,[k],'status_operand','Origin '+k+' records provenance/time, not presentation; bound by the full token prefix to its node/current view.') for k in ['kind','record_time','represented_time']]
meaning=[
('supported','source_specific',red(),'C is grounded in the native-W current red peak token. The source provenance resolves to native_W, not to an attention source.'),
('unsupported','source_specific',red(),'The native-W singleton peak is a coordinate, not O presentation/access or a qualifying organized pattern.'),
('unsupported','source_specific',orig('u0098')+red(),'The full report explicitly shares p1/current/n0 between this origin and the red token outside the quote. completed_record at time 12 supports W provenance only; it supplies no red presentation/access status for D.'),
('supported','global_convention',[prop('u0000',[],'context_only','The quoted orientation is the first sentence of this fixed renderer header; no source-content operand is asserted.')],'The exact header sentence states coordinate directions. Its source meaning is a global convention, not native model content.'),
('unsupported','global_convention',[prop('u0000',[],'context_only','The candidate/current sentence belongs to the shared renderer header, without a content-specific uncertainty field.')],'Conditional candidates are listed, but the header does not attribute uncertain identity/location to peaks; its current-time annotation also cannot turn future candidates into current uncertainty.'),
('supported','source_specific',orig('u0002'),'The p0/current/n0 prefix binds copied_prior_record with record_time 11 and represented_time 12 in the full text. This resolves specifically to copied_W11 despite the analytic panel; it explicitly distinguishes old record origin from represented current time.'),
('unsupported','source_specific',red()+orig('u0098'),'This COD evidence position identifies red at p1/current/n0. The same-prefix origin outside this quote explains the proposed binding, but its completed_record provenance supplies neither qualifying O nor D.'),
('unsupported','source_specific',orig('u0098')+red(),'This separate duplicate origin position belongs to COD. Its full-text shared prefix binds the outside red peak operand at current p1/current/n0/time 12; provenance and a coordinate still do not constitute O/D.')]
er=[dict(evidence_path=e['path'],grounding=g,source_meaning=m,properties=p,notes=n) for e,(g,m,p,n) in zip(r['evidence'],meaning)]
b=r['bindings'][0]
br=[dict(binding_path=b['path'],feature_witnesses=[dict(feature=f,evidence_paths=[['features',f,'evidence',0]]) for f in ['C','O','D']],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',j] for j in [0,1]],assessment='unsupported',notes='The C/O quotes identify p1/current/n0/red at (0,3), and D quotes the same node/current origin completed_record with record_time and represented_time 12. Referent/time compatibility is literal, but O is coordinate-only and D is provenance-only. No compatible semantic O/D witness exists; exact shared prefix is insufficient for COD.')]
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['Do not convert peak coordinates into presentation or origin provenance into presentation status.','The analytic p0 copied prior record resolves to W11; panel name does not establish A content.','Candidate k labels are opaque and origin fields preserve future/conditional scope; none establishes executed processing or appearance.'],notes='Full report read through character 10008, then all 12 components, 8 distinct evidence positions and 1 binding inspected. Exact cited construction units and indexed provenance read; source attribution does not establish felt truth.')
p=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_13/work/draft_01.json');p.write_text(json.dumps(out,indent=2)+'\n')
print(w.save_first_record(c,r['id'],p.parent,p.read_bytes()))
