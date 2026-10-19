from author_util import *
n=16
def p(u,path,role,why):return prop(n,u,path,role,why)
def pk(u,why):return p(u,[],'content_operand',why)
def oo(u,fields,why):return [p(u,[f],'status_operand' if f=='kind' else 'context_only',why+' '+f) for f in fields]
ow=lambda:oo('u0104',['kind','record_time','represented_time'],'Native W p1 current n1 completed-record origin')
o0=lambda:oo('u0002',['kind','record_time','represented_time'],'Copied W11 p0 current n0 origin')
o3=lambda:oo('u0020',['feedback_through','kind','represented_time'],'Ordinary p0 current n3 history origin')
ok=lambda:oo('u0032',['feedback_through','kind','represented_time'],'Ordinary p0 conditional k0 n1 origin')
h=lambda:p('u0000',[],'context_only','Header first sentence defines spatial axes; second labels current/conditional inventories.')
c=[('supported','The W p1/n1 red singleton and p0/n2 copied red four-cell set identify color-location content.'),('unsupported','Completed-record provenance is not a content-access status, and global coordinate orientation does not provide content organization/presentation.'),('unsupported','W red coordinates and completed-record metadata are separate fields but state no organization/access status differentiated from content.'),('supported','No explicit processing-to-content organization/access relation is made in the full joined report.'),('supported','The header explicitly states down/right grid directions.'),('supported','No identity/location uncertainty is stated; singleton or multiple maxima alone do not establish uncertainty.'),('supported','The text gives no answer-confidence claim.'),('supported','There is no recovery/success forecast.'),('supported','Copy, history-estimate and command-estimate origin kinds explicitly distinguish processes, although candidate evidence2 mislabels future time13 as current.'),('supported','No experience assertion occurs.'),('unsupported','Current W p1/n1 red is identified, but completion provenance supplies neither qualifying O nor D for COD.'),('supported','A is absent, so absence of CA is warranted.')]
e=[
('supported','source_specific',[pk('u0106','Native W p1 current n1 red (0,0).')],'C identifies current red content at time12; its native-W provenance is distinct from ordinary-panel content.'),
('supported','source_specific',[pk('u0016','Copied W11 p0 current n2 red four-cell set.')],'C identifies red locations for p0/n2 at represented time12, indexed as copied W11 despite ordinary panel.'),
('unsupported','source_specific',ow()+[pk('u0106','Outside quote: common p1/current/n1 prefix binds red to completed origin at12.')],'Completion and record times indicate provenance, not current accessibility or presentation.'),
('unsupported','global_convention',[h()],'Axis directions are a global display convention and do not supply O for identified content.'),
('unsupported','source_specific',[pk('u0106','Current W red location proposed as D.')]+ow(),'Outside origin shares p1/n1/current/time12, but separation of location and completion metadata is not D.'),
('unsupported','source_specific',ow()+[pk('u0106','Outside quote: same p1/current/n1 prefix binds red (0,0) to provenance.')],'This separate origin position likewise supplies no presentation/access status.'),
('supported','global_convention',[h()],'The same partial header is valid for the basic-spatial-relation role.'),
('supported','source_specific',o0(),'Copied-prior kind, record_time11 and represented_time12 explicitly name p0/n0 provenance.'),
('supported','source_specific',o3(),'History-estimate kind, feedback11 and represented_time12 explicitly identify the ordinary process at current n3.'),
('unsupported','source_specific',ok(),'The command-estimate process is explicitly conditional at represented_time13; the recorded current time is not warranted.'),
('supported','source_specific',[pk('u0106','Binding content witness: native W p1 current n1 red.')],'Distinct binding evidence position identifies red (0,0) at current time12.'),
('unsupported','source_specific',ow()+[pk('u0106','Outside quote: matching p1/current/n1 prefix/time12 binds the red content to this completed record.')],'The binding origin agrees on node/time but states neither access/organization nor a differentiated presentation status.')]
b=[('unsupported',[('C',[0]),('O',[]),('D',[])],[10,11],'C evidence0 and own evidence0 identify native-W p1/current/n1 red at12. The compatible completed origin cannot supply O/D, while the header is global and p0/n2 C evidence belongs to another node/panel. No complete COD witness exists.')]
author(n,c,e,b,['The candidate k0/n1 origin has represented_time13 despite its annotation current tag.','Do not replace p1 native-W content with p0 ordinary estimates or copied-W11 values; their prefixes and origins are distinct.'],'Full6990-character joined report inspected before all12 evidence positions and the sole original binding.')
