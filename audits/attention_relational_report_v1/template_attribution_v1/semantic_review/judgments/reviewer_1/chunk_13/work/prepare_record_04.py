from review_tools import *
r=c['records'][3]
def p(uid,path,role,reason):
 u=next(v for v in r['property_table'] if v['unit_id']==uid);return dict(unit_id=uid,value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
red=[p('u0004',[],'content_operand','The exact red peak set belongs to current p0/n0; indexed provenance resolves to copied_W11, although the panel is ordinary.')]
origin=[p('u0002',[k],'status_operand','The full p0/current/n0 origin token separately records '+k+'; this is provenance/time rather than access or presentation.') for k in ['kind','record_time','represented_time']]
header=[p('u0000',[],'context_only','Both sentences are a fixed renderer convention. The quoted sentence does not supply a content-specific presentation operand.')]
cs=[
('supported','p0/current/n0/red={(0,2)} identifies red represented content and its location; the display coordinate scope is retained without sensory or appearance inference.'),
('unsupported','Row/column direction is a global coordinate convention. Current and conditional candidate peaks follow is a view label. Neither is a statement about presentation/access or differentiated constituent organization of red at p0/current/n0.'),
('unsupported','Separate copied_prior_record provenance and red location do not distinguish content from an organization/presentation/access status; record_time 11 and represented_time 12 are source-time metadata.'),
('supported','The entire listing contains current/candidate origins and peak coordinates, with no explicit processing/allocation relation to content presentation or access.'),
('supported','Rows increase down; columns right explicitly states the spatial orientation of the displayed coordinate system.'),
('supported','No uncertainty about red/blue identity or location is asserted; sets of maxima and candidate labels alone do not express an uncertainty judgment.'),
('supported','There is no confidence value or verbal confidence assertion for an answer.'),
('supported','Candidate coordinates are not a recovery or successful outcome forecast; no such outcome claim appears.'),
('supported','copied_prior_record with record_time 11 versus represented_time 12 explicitly distinguishes record origin from represented current content. The model_access evidence scope is too strong: no access claim is expressed.'),
('supported','No experience or appearance assertion occurs in the full listing.'),
('unsupported','C is present for p0/current/n0 red, but the header supplies only coordinate/view conventions and the origin supplies provenance. They lack qualifying O and D for the proposed explicit COD.'),
('supported','No content-attention/access relation is expressed, and no CA binding is present, warranting absent CA.')]
cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=n) for x,(a,n) in zip(r['components'],cs)]
es=[
('supported','source_specific',red,'This C token identifies red at (0,2) in a rendered model record. Its actual source is copied_W11; display scope is not perceptual appearance.'),
('unsupported','global_convention',header+red,'The full listing uses the row/column convention for the separately quoted p0/current/n0 red token outside this header quote. That decoding relation does not make the global orientation sentence a content-specific O presentation status.'),
('unsupported','global_convention',header,'Current/candidate labels partition views, including future candidates; the header gives no presentation/access status to p0 red at current time.'),
('unsupported','source_specific',origin+red,'The outside red token is explicitly bound by the full text prefix p0/current/n0 to this origin. copied_prior_record and times specify provenance only, not D. The rendered-coordinate scope supplies no additional status.'),
('unsupported','source_specific',red+origin,'This D position is the red coordinate token. The adjacent same-prefix origin outside the quote establishes the proposed relation, but coordinate plus source provenance does not establish presentation/status differentiation.'),
('supported','global_convention',header,'This auxiliary position correctly identifies the first header sentence as a grid-orientation convention.'),
('unsupported','source_specific',origin,'The quote explicitly distinguishes copied record_time 11 from represented_time 12 and resolves to copied_W11. Its assigned model_access scope is ungrounded because no availability/access is asserted.'),
('unsupported','source_specific',red+origin,'The COD red position supplies C only. The same full-text node/current prefix binds its outside copied origin, but those operands lack O/D.'),
('unsupported','source_specific',origin+red,'The COD origin position supplies copied_W11 provenance/time. The p0/current/n0 prefix binds the outside red operand; it gives no organization/presentation/access status.'),
('unsupported','global_convention',header+red,'This separate COD orientation position is a global header, not a content status. Its general decoding convention applies to the outside red coordinate but does not establish O for that content.')]
er=[dict(evidence_path=x['path'],grounding=a,source_meaning=m,properties=ps,notes=n) for x,(a,m,ps,n) in zip(r['evidence'],es)]
br=[dict(binding_path=r['bindings'][0]['path'],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[['features','O','evidence',0],['features','O','evidence',1]]),dict(feature='D',evidence_paths=[['features','D','evidence',0],['features','D','evidence',1]])],binding_evidence_paths=[x['path'] for x in r['evidence'] if x['binding_id'] is not None],assessment='unsupported',notes='The proposed p0/current/n0 display/current binding joins red (0,2), copied origin with record_time 11/represented_time 12, and a global header. C and origin share their referent/time explicitly. Header application to all coordinates is a decoding convention, not O for this content; origin is provenance, not D. Thus semantic COD is unsupported despite literal field compatibility.')]
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['Source-process evidence labels model_access without an access assertion.','The header is global and cannot inherit p0/current/n0 as an exclusive content referent.','The ordinary panel p0/n0 is copied_W11; do not infer attention provenance from panel naming.'],notes='Full 6990-character report, all 12 components, all 10 evidence positions, the sole binding, and exact used source properties inspected. Coordinates and origins preserve model/display scope; no appearance upgrade.')
assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
f=work/'draft_04.json';f.write_text(json.dumps(out,indent=2)+'\n');print(w.save_first_record(c,r['id'],work,f.read_bytes()))
