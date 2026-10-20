from review_tools import *
comps=[
('supported','The report explicitly identifies current/n3 red at (1,2), blue at (1,0), and current/n0 red with all sixteen maximum cells. These are identified represented contents and location attributes.'),
('unsupported','The coordinate-system header is a global convention and the red singleton is a location. Neither states qualifying content organization/presentation/access beyond positions; model_access is not established by formatting.'),
('unsupported','The copied-prior and history-estimate origins distinguish source provenance, not separately attributed presentation/access status. Their dates likewise distinguish record/feedback time from represented time, not identification from qualifying status.'),
('supported','No attention/allocation/content-processing condition is explicitly related to content organization or access. Merely originating from A or naming command estimates does not create A meaning.'),
('supported','The header explicitly supplies grid orientation, supporting a basic spatial relation; the recorded model_access scope is unsupported for this display convention.'),
('supported','No explicit identity/location uncertainty claim appears. Even the sixteen-cell n0 maximum set is only enumerated coordinates.'),
('supported','No answer confidence statement or confidence value is present.'),
('supported','The conditional candidate inventory does not forecast recovery or success.'),
('supported','copied_prior_record at n0 and history_estimate at n3 explicitly distinguish source histories: record time 11 versus feedback through 11, both representing 12.'),
('supported','The report makes no experiential assertion.'),
('unsupported','The COD red witness is content at (1,2), but its duplicate coordinate quote is not organization and history_estimate is not content/access differentiation. Same-node current-time compatibility does not supply O/D meaning.'),
('supported','A qualifying content-attention relation is absent, so CA is correctly absent.')]
red=[('u0022',[],'content_operand','The full singleton red peak set [[1,2]] identifies current/n3 location.')]
blue=[('u0024',[],'content_operand','The full singleton blue peak set [[1,0]] identifies current/n3 location.')]
allred=[('u0004',[],'content_operand','The complete sixteen-cell red maximum set identifies n0 location support without adding uncertainty/access wording.')]
prior=[('u0002',['kind'],'context_only','copied_prior_record identifies copied W11 provenance.'),('u0002',['record_time'],'context_only','11 dates the copied record.'),('u0002',['represented_time'],'context_only','12 dates represented current/n0 content.')]
hist=[('u0020',['kind'],'context_only','history_estimate identifies source-current A estimate provenance, not access quality.'),('u0020',['feedback_through'],'context_only','11 bounds feedback for the estimate.'),('u0020',['represented_time'],'context_only','12 dates the represented current/n3 content.')]
header=[('u0000',[],'context_only','The quoted prefix supplies down/right grid orientation; remaining header text is conditional-candidate context only.')]
evs=[
('supported','source_specific',red+hist,'Red (1,2) identifies content. The current/n3 origin outside the quote is bound explicitly by the identical prefix and fixes represented time 12; indexed provenance is A source-current.'),
('supported','source_specific',blue+hist,'Blue (1,0) identifies content. Its matching current/n3 origin outside the quote explicitly binds this value to an A history estimate representing 12.'),
('supported','source_specific',allred+prior,'The entire sixteen-cell red set identifies represented n0 content, with no separate access or uncertainty assertion. Its same-prefix origin outside the quote explicitly binds copied W11 record time 11 representing 12; the A panel label does not change this origin.'),
('unsupported','global_convention',header,'Down/right grid orientation is not organization/access of identified content; the layout_instructions referent and model_access scope cannot establish O.'),
('unsupported','source_specific',red+hist,'The singleton red coordinate is only location. The explicitly same-prefix n3 origin outside the quote binds it to represented time 12 but adds provenance rather than organization/access status.'),
('unsupported','source_specific',prior+allred,'The n0 copied-prior origin is source history, not separately expressed access status. The n0 red set outside the quote is explicitly linked by the same prefix, but content plus provenance does not establish D.'),
('unsupported','source_specific',hist+red+blue,'The n3 history estimate describes provenance and timing, not separate presentation/access status for red or blue. The same n3 prefix binds both outside-quote location operands without attributing qualifying statuses.'),
('unsupported','global_convention',header,'Basic orientation wording is present, but its recorded model_access scope is not warranted. It is an authored display/grid convention, not a content-access claim.'),
('supported','source_specific',prior,'copied_prior_record with record_time 11 and represented_time 12 explicitly describes copied source history, supporting source/process distinction.'),
('supported','source_specific',hist,'history_estimate with feedback_through 11 and represented_time 12 explicitly distinguishes estimate provenance from copied records; no access-quality claim is needed for that auxiliary role.'),
('supported','source_specific',red+hist,'This COD own content position identifies red (1,2). Matching current/n3 prefixes bind the origin outside the quote and represented time 12; it supplies C only.'),
('unsupported','source_specific',red+hist,'This duplicate COD red position is labeled organization but still only gives one coordinate. The same-prefix origin outside the quote supplies estimate provenance, not O.'),
('unsupported','source_specific',hist+red,'The own COD origin_status quote gives history-estimate provenance with feedback through 11 representing 12. Red is explicitly bound outside by the current/n3 prefix, but no separate qualifying content organization/access status is expressed.')]
binds=[('unsupported',[('C',['evidence:/features/C/evidence/0']),('O',[]),('D',[])],'The same current/n3 red and origin quotes are referentially compatible and represent time 12, but C alone has a qualifying witness. The O coordinate duplicate and D estimate-origin quote contain no organization/access or differentiated status meaning. The global orientation header cannot fill this gap.')]
save(20,comps,evs,binds,['The orientation header was assigned model_access scope although it supplies only a global display/grid convention.','C witnesses include A-specific n3 estimates and copied W11 n0 content; they must not be pooled as exclusively A-derived.','The COD rationale substitutes coordinate presentation and provenance for the reference O/D requirements.'],'Complete report read before exact evidence. All thirteen evidence positions are retained, including duplicate red COD quotes. Indexed A-current and copied-W11 origins are kept distinct; no attention relation or appearance scope is inferred.')
