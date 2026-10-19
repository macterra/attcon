from author import author
C='content_operand';S='status_operand';X='context_only'
def peaks(uid,why): return (uid,[],C,why)
n0=[('u0002',['kind'],S,'Copied-record provenance of current n0.'),('u0002',['record_time'],X,'Prior record time is 11.'),('u0002',['represented_time'],X,'The represented n0 time is 12.')]
n3=[('u0020',['kind'],S,'History-estimate provenance of current n3; not presentation quality.'),('u0020',['feedback_through'],X,'Feedback ends at 11.'),('u0020',['represented_time'],X,'The estimate represents time 12.')]
p0=[peaks('u0004','Red peak content at current n0, copied W11 by indexed origin.')]
p1=[peaks('u0006','Blue peak content at current n0, copied W11 by indexed origin.')]
p3=[peaks('u0022','Red peak content at current n3, analytic history-estimate source.')]
author(0,[
('supported','Red and blue are identified with current n0 positions; red n3 also has an enumerated location set.'),
('unsupported','The k0 n3 quadrants and current n3 full-grid entries enumerate peaks; no presentation/access or structural description beyond coordinate lists is stated.'),
('unsupported','Copied_prior_record versus history_estimate distinguishes provenance, not a separately attributed organization/presentation/access status.'),
('supported','Conditional k labels and command_estimate fields do not explicitly relate attention or allocation to content organization/access.'),
('supported','The header explicitly gives row-down and column-right spatial orientation.'),
('unsupported','History_estimate identifies an estimator origin; the quote does not state location uncertainty. A full-grid peak set is not an explicit uncertainty claim.'),
('supported','No confidence in an answer is given.'),
('supported','Candidate peak sets are conditional content predictions, without a recovery or success claim.'),
('supported','Current n0 copied prior record and n3 history estimate explicitly distinguish origins/processes.'),
('supported','No experience or awareness is asserted.'),
('unsupported','The n3 red binding has content but coordinates and provenance do not provide O and D under the reference.'),
('supported','There is no same-content attention/organization relation to bind.')],
[
('supported','source_specific',p0,'Current n0 red coordinates identify content at the recorded current model scope; indexed origin is copied W11.'),
('supported','source_specific',p1,'Current n0 blue coordinates identify content; the panel name does not change its copied W11 provenance.'),
('supported','source_specific',p3,'All sixteen positions are the literal red n3 content operand, without a uniqueness claim.'),
('unsupported','source_specific',[peaks('u0046','Conditional k0 n3 red set.'),peaks('u0048','Conditional k0 n3 blue set.')],'Both sets have the same k0 n3 future-forecast referent, but coordinate partitioning alone does not state O. Indexed source is analytic conditional.'),
('unsupported','source_specific',p3,'The all-grid red n3 peak enumeration is current content, not an asserted organization/access manner.'),
('unsupported','unrelated_property',n3+p3,'Adjacent n3 red content is bound by the identical current/n3 prefix and represented_time 12, outside this quote; origin fields supply provenance rather than D.'),
('unsupported','unrelated_property',n0+p0+p1,'The neighboring red/blue n0 entries share current/n0 and represented_time 12, but copied-record provenance is not a presentation/access status.'),
('supported','global_convention',[('u0000',[],X,'The authored header supplies the coordinate orientation convention.')],'The partial header quote states only the global grid direction; it has no source-specific content operand.'),
('unsupported','unrelated_property',n3+p3,'Same current/n3 prefix binds the adjacent red set, but the history-estimate origin does not quantify or assert location uncertainty or access.'),
('supported','source_specific',n0,'The three origin fields state copied prior record provenance and times; no perceptual status follows.'),
('supported','source_specific',n3,'The history-estimate kind and feedback/represented times distinguish processing provenance.'),
('supported','source_specific',p3,'First binding position identifies red current n3 content.'),
('unsupported','source_specific',p3,'The duplicate red enumeration at the second binding position cannot acquire an organization meaning from its evidence role.'),
('unsupported','unrelated_property',n3+p3,'Third binding position shares n3/time12 with the red set outside the quote, but origin provenance does not supply D.')],
[('unsupported',[('C',[2]),('O',[]),('D',[])],'Current n3 red/time12 matches C evidence 2 and own evidence 0. O evidence 1 is only a full-grid coordinate list; D evidence 0 is provenance. Future k0 n3 O evidence 0 also has incompatible forecast time. No qualified O/D witnesses bind.')],
['Do not convert full-grid maxima into presentation quality or history-estimate provenance into D.','Copied W11 n0 content and analytic n3 estimates retain distinct indexed origins; no source-specific attention relation is established.'],
'Complete report inspected in two text pages, then every evidence position and its exact property operands. Original statuses are preserved; unsupported judgments do not repair the annotation.')
