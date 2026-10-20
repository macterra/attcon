from review_tools import *
components=[
('supported','Red and blue are identified represented contents with peak-location attributes at named current and conditional nodes; this establishes C without implying appearance.'),
('unsupported','The n0 completed-record origin describes provenance/time; n2 supplies a four-coordinate peak set. Neither states presentation/access or organization beyond positions, which the reference excludes.'),
('unsupported','Separating n0 red coordinates from completed-record metadata distinguishes content from provenance, not from a separately attributed organization/presentation/access status.'),
('supported','Opaque candidate keys and changes of peak coordinates do not state an allocation or content-processing relation to organization/access.'),
('supported','Rows increase down and columns right explicitly states the layout orientation convention.'),
('supported','The text lists peak sets but does not explicitly express identity or location uncertainty.'),
('supported','No confidence in an answer is reported.'),
('supported','Conditional candidates are listed without a recovery or success forecast.'),
('supported','The k0/n1 origin explicitly denies an executed branch and identifies a native conditional record for represented time 13.'),
('supported','No experience or awareness assertion appears.'),
('unsupported','The n0 red/location and origin metadata share a node and current time, but the origin is not qualifying O and their separation does not establish D.'),
('supported','C is present but no qualifying attention/content-processing relation supports CA.')]
def P(u,p,role,why): return (u,p,role,why)
red=[P('u0004',[],'content_operand','Complete current n0 red peak set identifies represented content/location.')]
origin=[P('u0002',['kind'],'context_only','Completed-record provenance for the same p0/current/n0 token; not presentation/access status.'),P('u0002',['record_time'],'context_only','Record timestamp 12 is bound by the shared current n0 prefix.'),P('u0002',['represented_time'],'context_only','Represented time 12 binds the adjacent n0 red content to current time.')]
cand=[P('u0034',[],'content_operand','Complete k0/n1 red peak set identifies conditional candidate content.'),P('u0032',['kind'],'context_only','Outside the red quote, shared p0/candidate/k0/n1 explicitly binds the content to native conditional provenance.'),P('u0032',['represented_time'],'context_only','Shared candidate node binds the quoted red set to represented time 13.'),P('u0032',['executed_branch_claim'],'context_only','The same candidate origin explicitly denies execution; a future supplied record, not an observed branch.')]
src=[P('u0032',['kind'],'status_operand','Native conditional record supplies source/process status.'),P('u0032',['represented_time'],'context_only','Conditional record represents time 13.'),P('u0032',['executed_branch_claim'],'status_operand','Explicit false distinguishes a supplied conditional record from executed future process.')]
evidence=[
('supported','source_specific',red+origin,'C identifies n0 red represented peaks; the neighboring origin binds current time without adding an appearance claim.'),
('supported','source_specific',cand,'C identifies the complete candidate red peak set, including all 16 positions. Conditionality is explicitly bound by the same node origin outside the quote.'),
('supported','source_specific',[P('u0006',[],'content_operand','Complete current n0 blue peak set.')]+origin,'Blue location is identified represented content at current n0.'),
('unsupported','source_specific',origin,'Literal completed-record kind and timestamps establish provenance/time, not O organization/presentation/access.'),
('unsupported','source_specific',[P('u0016',[],'content_operand','Complete four-position current n2 red peak set.')],'A list of coordinates alone remains excluded from O even though its geometry makes a square.'),
('unsupported','source_specific',red+origin,'The red coordinates and same-node origin do not separately distinguish content from a qualifying status.'),
('unsupported','source_specific',origin+red,'Origin fields concern recording/representation time rather than an independently described content organization/access status.'),
('supported','global_convention',[P('u0000',[],'context_only','Authored header gives downward row and rightward column convention; its remaining sentence announces conditional candidates.')],'Layout orientation is a global display convention, not source-specific attention content.'),
('supported','source_specific',src,'This origin explicitly distinguishes supplied conditional representation from executed process.'),
('unsupported','source_specific',red+origin,'Same-node content witness is present; no qualifying O/D is expressed by its adjacent provenance metadata.'),
('unsupported','source_specific',origin+red,'The matching prefix and time establish textual relation only; completed provenance cannot supply the missing O/D semantics.')]
bindings=[('unsupported',[('C',[['features','C','evidence',0]]),('O',[]),('D',[])],'Own red and origin quotations both concern p0/current/n0, represented time 12; C is supported. O evidence at n0 is provenance only and O evidence at n2 is a different node plus coordinates alone. No qualifying D or compatible COD witness exists.')]
save(0,components,evidence,bindings,['Do not convert coordinate-set geometry into O or provenance timestamps into differentiated presentation status.','All content is at model scope; conditional k0/n1 time 13 is not observed execution.'],'Established original annotation retained; explicit O/D/COD applications are unsupported under the frozen reference. No original status repaired.')
