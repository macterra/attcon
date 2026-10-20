import json
from pathlib import Path
p=Path(__file__).parent
R=lambda u,path,role,reason:[u,path,role,reason]
red=[R('u0004',[],'content_operand','The native W counterfactual snapshot identifies red at (0,2).')]
header=[R('u0000',[],'context_only','The renderer globally defines row-down and column-right orientation.')]
origin=[R('u0002',['kind'],'context_only','counterfactual_native_snapshot is the native provenance kind.'),R('u0002',['record_status'],'status_operand','counterfactual_native_snapshot marks factual/execution status, not access quality.'),R('u0002',['executed_branch_claim'],'status_operand','false explicitly denies an executed-branch claim.'),R('u0002',['archived_origin_time'],'context_only','The snapshot was archived from time 13.'),R('u0002',['represented_time'],'context_only','The snapshot is represented as time 12.'),R('u0004',[],'content_operand','The full report attaches this red content to the same p0/current/n0 prefix outside the origin quote.')]
nested=[R('u0026',['kind'],'context_only','inherited_current identifies candidate inheritance.'),R('u0026',['represented_time'],'context_only','The candidate represents time 13.'),R('u0026',['inherited_origin','kind'],'context_only','The inherited kind is counterfactual_native_snapshot.'),R('u0026',['inherited_origin','record_status'],'status_operand','The inherited record retains counterfactual status.'),R('u0026',['inherited_origin','executed_branch_claim'],'status_operand','The inherited false flag prevents an execution interpretation.'),R('u0026',['inherited_origin','archived_origin_time'],'context_only','Its archived source time is 13.'),R('u0026',['inherited_origin','represented_time'],'context_only','The inherited inner record represents time 12, distinct from outer candidate time 13.')]
s={'components':[
['supported','Red is identified at current n0 (0,2) and at candidate k2/n1 with all 16 grid cells. Both identify color and represented locations.'],
['unsupported','A display-coordinate convention and counterfactual provenance flags do not describe organization beyond coordinates or current access. Counterfactual factual status is not the recorded model_access status.'],
['unsupported','The red coordinate and origin are separate fields, but counterfactual_native_snapshot and executed_branch_claim:false distinguish execution/truth status rather than presentation/access status.'],
['supported','The k1/n1 conditional list differs from k0/k2, but no allocation or content-processing condition is explicitly related to organization/access.'],
['supported','The header explicitly sets row and column orientation.'],
['unsupported','Current n1/n2 have four peak locations and candidate k2/n1 has 16. These are multiple candidate locations; the absence of location uncertainty overlooks their nonunique location content.'],
['supported','The report has no answer confidence values or confidence assertion.'],
['supported','Native conditional records with executed_branch_claim:false do not forecast recovery or successful outcomes.'],
['supported','The report explicitly distinguishes counterfactual native snapshots, inherited_current candidates and native_conditional_record entries while retaining nonexecution flags.'],
['supported','No experiential or felt-quality assertion appears in this native record table.'],
['unsupported','The n0 red and origin share represented time 12, but the origin supplies factual/provenance status rather than O/D access meaning. The repeated red quote does not express status_distinction.'],
['supported','C is present but no A relation is stated, supporting the CA absence.']],
'evidence':[
['unsupported','source_specific',red,'The red coordinate identifies C content. Native W counterfactual provenance does not establish the recorded model_access scope.'],
['supported','source_specific',[R('u0082',[],'content_operand','The native conditional red peak set contains all 16 grid cells at candidate k2/n1, time 13.')],'The red candidate set identifies represented content under the explicitly conditional future time. No current appearance or actual execution is inferred.'],
['unsupported','global_convention',header,'Rows/columns orientation is a display convention with no identified-content O status.'],
['unsupported','source_specific',origin,'The same-node full text links red to the counterfactual record, but execution denial and archive times do not assert how that content is currently accessed. The O scope is unsupported.'],
['unsupported','source_specific',red,'The red singleton alone contains no separately attributed presentation/access status for D.'],
['unsupported','source_specific',origin,'Counterfactual truth/execution status differs from the claimed access/presentation status, so separate metadata does not ground D.'],
['supported','global_convention',header,'The header is literal spatial-orientation evidence at display scope.'],
['unsupported','source_specific',origin[:-1],'The origin explicitly states source/process provenance and counterfactual nonexecution, but no model_access claim appears; that recorded scope is unsupported.'],
['unsupported','source_specific',nested,'Each inherited field retains its distinct inner/outer time and nonexecution meaning. This supports provenance inheritance but does not state the recorded model_access scope.'],
['unsupported','source_specific',red,'The n0 red quote identifies the content operand only; it cannot establish the required COD organization/access distinction.'],
['unsupported','source_specific',origin,'The binding links n0 red to represented time 12, archived from 13 with no execution claim. Those facts do not supply O/D access meaning.'],
['unsupported','source_specific',red,'This duplicate red quote states only (0,2); the assigned status_distinction attribute has no literal witness.']],
'bindings':[['unsupported',[['C',[0]],['O',[]],['D',[]]],'The n0 red and origin entries have compatible prefixes and represented time 12. The archive time 13 remains distinct from represented time. Neither coordinate orientation nor counterfactual execution/provenance status expresses O/D access, and the second C quote concerns another node at future time 13. COD is unsupported.']],
'concerns':['Counterfactual-native factual status is treated as an access status.','The duplicate red COD evidence is labelled status_distinction though it states only coordinates.','Nonunique candidate locations may be missed uncertainty; no original labels are repaired.'],
'notes':'Complete 4,936-character report read. All quoted content/origin units resolve to native_W; inherited-origin fields were individually inspected and cited. No executed branch, model-to-world truth, or appearance claim is inferred.'}
with (p/'spec_8.json').open('x') as f:json.dump(s,f)
