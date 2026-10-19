from author_util import *
n=0
def p(u,path,role,why):return prop(n,u,path,role,why)
def peak(u,why):return p(u,[],'content_operand',why)
def origin(u,why):
 return [p(u,[f],'context_only' if f!='kind' else 'status_operand',why+' Field '+f+'.') for f in row(n)['property_table'][int(u[1:])]['value']]
c=[
('supported','The n0 red/blue and n3 red peak entries identify color content with location attributes at represented time 12.'),
('unsupported','The candidate k0 partition and current full-grid n3 entries enumerate maximal coordinates. They never state a constituent organization or presentation/access status beyond those positions, excluded by O.'),
('unsupported','history_estimate versus copied_prior_record records provenance. Neither origin separately states how red content is organized, presented or accessible; provenance must not be promoted to D.'),
('supported','No command is related explicitly to a content organization/access change; k labels and different conditional coordinates alone cannot establish A.'),
('supported','The header explicitly specifies downward rows and rightward columns as a grid convention.'),
('unsupported','history_estimate names an estimation process; it states no uncertainty in identity/location. The tied coordinate list gives no explicit uncertainty language.'),
('supported','No answer certainty or response confidence is stated.'),
('supported','Conditional candidate peak inventories do not assert recovery or success.'),
('supported','n0 is labeled copied_prior_record from time 11 and n3 history_estimate with feedback through 11, explicitly distinguishing their origins.'),
('supported','No experience or felt-state assertion occurs in the exact report.'),
('unsupported','The n3 red current binding has identified content but no qualifying O or D; a full-grid peak set plus an origin does not establish COD.'),
('supported','No qualifying content-processing/organization relation supports CA.')]
e=[
('supported','source_specific',[peak('u0004','Current n0 red location content; indexed origin is copied W11, despite the analytic panel.')],'The quote identifies red at (0,3), current represented time 12; the model scope denotes represented coordinates only.'),
('supported','source_specific',[peak('u0006','Current n0 blue location content from copied W11.')],'The separate blue evidence position identifies (0,1) for n0 at time 12.'),
('supported','source_specific',[peak('u0022','Analytic current n3 red complete peak set.')],'All sixteen coordinates are asserted for red n3 at time 12; content identification qualifies C without qualifying O.'),
('unsupported','source_specific',[peak('u0046','Conditional k0 n3 red maxima.'),peak('u0048','Conditional k0 n3 blue maxima.')]+origin('u0044','Shared p0/candidate/k0/n3 prefix binds outside-quote origin to both peak sets; it establishes conditional time 13.'),'The red/blue coordinate partition is real analytic conditional content, but it is still a peak list, not stated presentation/organization. The separating space adds no operand.'),
('unsupported','source_specific',[peak('u0022','Full-grid current analytic red maxima.')],'The exact current list exhausts all cells; that is not itself a presentation/access claim under O.'),
('unsupported','source_specific',origin('u0020','n3 current origin')+[peak('u0022','Outside quote: the shared p0/current/n3 prefix binds this red content to the origin at represented time 12.')],'The origin establishes history-estimate provenance, not an organization/presentation/access status for the same red content.'),
('unsupported','source_specific',origin('u0002','n0 current copied origin')+[peak('u0004','Outside quote: shared p0/current/n0 binds red content to the copied record at represented time 12.')],'Copied prior record is source history; it does not distinguish red content from a qualifying representation/access status.'),
('supported','global_convention',[p('u0000',[],'context_only','Only the first header sentence is cited: it defines row and column directions; the remainder names current/conditional inventories.')],'The partial header overlap supports grid orientation as a global renderer convention, with no source-content attribution.'),
('unsupported','source_specific',origin('u0020','n3 history-estimate origin')+[peak('u0022','Outside quote: common current n3 prefix ties red coordinates to this estimate.')],'The kind field says estimate and the feedback field says through 11; neither asserts location uncertainty or model access.'),
('supported','source_specific',origin('u0002','n0 copied-prior provenance'),'The three fields identify a copied time-11 record represented at 12; source/process distinction is warranted, not perceptual access.'),
('supported','source_specific',origin('u0020','n3 analytic history provenance'),'The estimate kind, feedback cutoff 11 and represented time 12 identify a different source process from n0.'),
('supported','source_specific',[peak('u0022','Binding content: current n3 red maxima.')],'This duplicate quotation is a separate COD C-role witness and identifies n3 red content at time 12.'),
('unsupported','source_specific',[peak('u0022','Binding proposed organization operand: current n3 full peak set.')],'This duplicate list has no additional meaning when used as binding O evidence; all-cell maxima remain coordinates.'),
('unsupported','source_specific',origin('u0020','Binding proposed status: analytic history origin')+[peak('u0022','Outside quote: same current n3 prefix and represented time 12 bind the red list to this origin.')],'The binding origin is compatible in node/time with red, but provenance alone does not supply D.')]
b=[('unsupported',[('C',[2]),('O',[]),('D',[])],[11,12,13],'C evidence /features/C/evidence/2 and own evidence/0 identify the same p0/current/n3 red, time 12. Proposed O evidence/1 and D evidence/0 share that node/time but only list peaks and provenance. No qualifying O or D witness exists; candidate k0 O evidence is additionally future time 13 and cannot repair this current binding.')]
author(n,c,e,b,['Do not interpret sixteen tied maxima as experienced diffusion or uncertainty.','The analytic panel n0 content originates in copied W11; only n3 current/conditional values are analytic-source content.','Candidate time 13 remains conditional, not executed or current.'],'Full 5040-character report inspected before evidence; all 14 evidence positions, including duplicate binding quotes, individually retained.')
