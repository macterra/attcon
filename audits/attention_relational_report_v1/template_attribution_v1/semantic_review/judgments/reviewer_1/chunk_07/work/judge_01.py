from review_io import *
r=c['records'][1]
def p(i,path,role,reason):return prop(r,i,path,role,reason)
red=p(4,[],'content_operand','Identifies the complete red peak set for p0/current/n0; singleton location does not by itself express an access status.')
ori=[p(2,['kind'],'status_operand','The same p0/current/n0 prefix binds copied-prior representation status to its red content.'),p(2,['record_time'],'status_operand','The copied observation was recorded at 11.'),p(2,['represented_time'],'status_operand','That prior record is supplied as the representation for 12; the distinct times qualify its representation status.')]
head=p(0,[],'context_only','Authored header convention, whose quoted subsection supplies only coordinate orientation or conditional-candidate vocabulary.')
est=[p(20,['kind'],'status_operand','History-estimate provenance for p0/current/n3, explicitly separated from the n0 copied record.'),p(20,['feedback_through'],'status_operand','History estimate uses feedback through 11.'),p(20,['represented_time'],'status_operand','Estimate represents time 12.'),p(22,[],'content_operand','Outside the origin quote, the identical p0/current/n3 prefix binds the history status to red peak (1,2) in the full text.')]
sp=[p(32,['kind'],'status_operand','Candidate n1 inherits the current representation.'),p(32,['represented_time'],'status_operand','The candidate represents 13, so the recorded current-time annotation is unsupported.'),p(32,['inherited_origin','kind'],'status_operand','Nested actual inherited origin is a copied prior record.'),p(32,['inherited_origin','record_time'],'status_operand','That underlying record is from 11.'),p(32,['inherited_origin','represented_time'],'status_operand','The inherited current representation was for 12.')]
comps=[
('supported','Red n0 singleton and red n2 complete-grid set explicitly identify represented content and the red attribute.'),
('supported','n0 copied_prior_record with record_time 11 versus represented_time 12 differentiates representation status; n3 separately states history_estimate with feedback horizon 11. This supports model representation/access structure without appearance.'),
('supported','The red n0 content is listed separately from a copied-prior status with distinct record and represented times under the same current/n0 prefix.'),
('supported','k0/k1/k2 are arbitrary candidate labels; the complete text supplies no functional attention/allocation condition related to an access or organization change.'),
('supported','The header explicitly maps increasing rows down and columns right.'),
('unsupported','Conditional candidate vocabulary identifies alternative branches, not uncertainty about identity or location; the header supplies no location uncertainty judgment.'),
('supported','No answer-confidence statement appears in the complete report.'),
('supported','Candidate peaks and command_estimate origins assert no recovery or success outcome or likelihood.'),
('unsupported','The nested provenance does distinguish inherited from prior record, but the sole evidence is a candidate representation for 13, incorrectly annotated current. The recorded application including time is unsupported.'),
('supported','There is no assertion of experience or felt appearance.'),
('unsupported','Compatible n0 red content and copied-prior status exist, but binding evidence position 2 is blue while labelled red. The complete original binding is not grounded as recorded.'),
('supported','A is absent and no content-attention relation is bound.')]
evs=[
(0,'supported','source_specific',[red],'C identifies red singleton at p0/current/n0, inherited from copied W11 according to the indexed origin; no perceptual appearance is inferred.'),
(1,'supported','source_specific',[p(16,[],'content_operand','All sixteen cells are the complete red peak set for current n2.')],'C identifies the red attribute and complete set, without converting uniform maxima to uncertainty or appearance.'),
(2,'supported','source_specific',ori+[red],'The origin and red peak tokens share p0/current/n0 in the full text; copied record time 11 versus represented time 12 separately qualifies that content in model representation scope.'),
(3,'supported','source_specific',est,'Origin and red content share p0/current/n3, binding feedback horizon 11 and represented time 12 to a history-based representation; this is not felt access.'),
(4,'supported','source_specific',[red]+ori,'The red quote supplies the content operand; the immediately preceding same-node origin supplies a separate copied-prior status and its two times.'),
(5,'supported','source_specific',ori+[red],'Copied-prior status is separated from the red location under the same current/n0 prefix and target time 12.'),
(6,'supported','global_convention',[head],'Only the header subsection about increasing rows and columns supports coordinate orientation; it is a renderer-wide convention.'),
(7,'unsupported','global_convention',[head],'Current and conditional candidate peaks is a generic branch convention and says nothing about uncertainty of an identity or location.'),
(8,'unsupported','source_specific',sp,'Nested inherited-current/copied-prior provenance is real, but candidate represented_time 13 contradicts the recorded current time; no generic header can resolve that mismatch.'),
(9,'supported','source_specific',[red]+ori,'Binding position 0 identifies n0 red content, with outside-quote same-prefix copied status at represented time 12.'),
(10,'supported','source_specific',ori+[red],'Binding position 1 supplies n0 copied-prior representation status, bound by the full-text same-node red token.'),
(11,'unsupported','source_specific',[p(6,[],'content_operand','This is the blue peak set (0,2), not red n0 content.'),red],'Binding position 2 explicitly quotes blue but records attribute red. Same node/time does not make different role attributes identical.')]
b=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[['features','O','evidence',0]]),dict(feature='D',evidence_paths=[['features','D','evidence',0],['features','D','evidence',1]])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',i] for i in range(3)],assessment='unsupported',notes='C/O/D feature witnesses can bind red n0 and its copied-prior representation at time 12 in model_access scope. Own evidence position 2 instead names blue at (0,2) while the binding calls it red; every own evidence position must retain its attribute. This original binding is unsupported as recorded.')]
save(1,comps,evs,b,['The panel label A does not make copied W11 origins or peaks A-specific content.','The candidate source-process evidence represents 13 despite its current-time annotation.','COD binding position 2 misattributes blue content to red; no binding repair is performed.'],'Complete report read before evidence. Twelve components, all twelve evidence positions and the sole original binding judged. Source-specific means provenance-specific, not source-causal or experiential validity.')
