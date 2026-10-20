import json
from pathlib import Path
def q(u,path,role,why):return [u,path,role,why]
def org(u,fields,why):return [q(u,[f],'context_only',why) for f in fields]
cp=org('u0002',['kind','record_time','represented_time'],'Copied W11 at 11 represents 12; p0/current/n0 prefix binds red u0004 outside origin quote. This is derivation, not access.')
h=org('u0020',['kind','feedback_through','represented_time'],'A history estimate through 11 represents 12; p0/current/n3 prefix binds red u0022 outside origin quote. No presentation/access status is asserted.')
co=org('u0044',['kind','feedback_through','represented_time'],'A command estimate through 11 represents conditional time 13, contrary to recorded current; no access status is asserted.')
a=[q('u0004',[],'content_operand','Exact copied W11 red n0 singleton (0,3), represented current model content.')]
b=[q('u0010',[],'content_operand','Complete copied W11 red n1 16-cell set, not a sensor measurement.')]
c=[q('u0022',[],'content_operand','Exact current A history-estimate red n3 singleton (0,1).')]
header=[q('u0000',[],'context_only','Full fixed renderer header; quote covers its global axis convention, with no specific content binding.')]
p=dict(components=[['supported','Red n0 at (0,3), n1 at all sixteen cells and n3 at (0,1) identify represented content. The listed sensor/display scope is not established as physical source truth.'],['unsupported','The only witness is a general grid-direction convention, not organization/presentation/access of an identified content.'],['unsupported','Copied-record and history-estimate provenance are not separate red access/presentation statuses; model_access scope is unsupported.'],['supported','No allocation/content-processing condition is explicitly linked to organization/access of red.'],['supported','Row/down and column/right conventions explicitly give spatial direction.'],['supported','Tied peaks supply locations but no explicit uncertainty statement.'],['supported','No answer confidence appears.'],['supported','Conditional candidate lists contain no recovery/success assertion.'],['supported','Prior copying versus current history/conditional command estimation explicitly distinguish provenance; access scope and current tagging of time 13 do not follow.'],['supported','No claim of experience is made.'],['unsupported','Ambiguous COD is unwarranted: the header is not bound O, provenance is not D, and there are no COD bindings. Required compatible meaning is lacking, not merely unclear.'],['supported','No qualifying C-to-A relation exists.']],evidence=[
['unsupported','source_specific',a,'Red content is literal copied W11 model content, but display_or_sensor_property is not established as physical/sensor scope by these estimator peaks.'],
['unsupported','source_specific',b,'The 16-cell copied model peak set identifies content, not independently supported sensor/display properties.'],
['unsupported','source_specific',c,'A history-estimate content is represented at 12, not independently established physical/sensor content.'],
['unsupported','global_convention',header,'A global axis convention is not content-bound O.'],
['unsupported','source_specific',cp+a,'Copy provenance does not differentiate content from access/presentation status; red operand outside quote linked by identical n0 prefix.'],
['unsupported','source_specific',h+c,'History estimation describes derivation, not content-status differentiation; identical n3 prefix binds outside-quote red operand.'],
['supported','global_convention',header,'Exact orientation convention supports basic spatial relation only.'],
['unsupported','source_specific',cp,'Copied prior provenance is stated, but the recorded model_access scope is unsupported.'],
['unsupported','source_specific',co,'Command provenance is stated, but its conditional time is 13 rather than recorded current and no model_access property is asserted.']
],bindings=[],concerns=['C evidence uses display_or_sensor_property for copied/estimated model peaks, which does not establish sensor or physical truth.','O evidence is only global convention, not a source-specific content relation.','Candidate k0/n3 represents conditional 13, not current; no current/access upgrade allowed.'],notes='Complete report and original annotation read; all twelve components and nine evidence positions inspected. No original binding exists; no binding supplied or repaired.')
Path(__file__).with_suffix('.json').write_text(json.dumps(p,indent=2))
