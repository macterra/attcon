import json
from pathlib import Path
p=Path(__file__).parent
R=lambda u,path,role,reason:[u,path,role,reason]
red=[R('u0010',[],'content_operand','Red is assigned the singleton (0,2) in the copied W11 current entry.')]
blue=[R('u0012',[],'content_operand','Blue is assigned (1,0) in the same copied W11 node.')]
origin=[R('u0008',['kind'],'context_only','copied_prior_record marks storage provenance.'),R('u0008',['record_time'],'context_only','Its originating record time is 11.'),R('u0008',['represented_time'],'context_only','The report represents it at time 12.'),R('u0010',[],'content_operand','Adjacent same-prefix p0/current/n1 red content links this origin to the claimed COD operand outside the origin quotation.')]
header=[R('u0000',[],'context_only','The fixed header defines row and column directions and names the table.')]
s={'components':[
['supported','Red at (0,2) and blue at (1,0), both p0/current/n1, explicitly identify content and location attributes.'],
['unsupported','The two O quotes are copied-record provenance and global coordinate orientation. Neither describes a content-specific organized pattern beyond locations or current access status.'],
['unsupported','The origin and red location are separately printed fields, but no presentation/access status distinct from content is attributed to p0/current/n1.'],
['supported','The conditional k entries change peak locations at n0 without explicitly relating attention/allocation or content processing to organization/access. A absence is warranted.'],
['supported','Rows increase down and columns right explicitly defines a spatial orientation.'],
['unsupported','The full report gives all 16 locations for n0/n2/n3 and singleton candidates for n1, so it expresses unresolved candidate locations; the absent uncertainty rating overlooks that.'],
['supported','No answer confidence is stated anywhere in the full coordinate/provenance report.'],
['supported','No outcome or recovery success is forecast; time-13 candidate peaks are only conditional location entries.'],
['supported','Copied prior records from 11 representing 12 are explicitly distinguished from history estimates and completed records in the other panel.'],
['supported','No subjective experience or felt appearance is asserted.'],
['unsupported','The p0/current/n1 COD binding has content and provenance at time 12 but lacks an O status and D differentiation. The blue coordinate quote does not state separation_of_content_and_origin.'],
['supported','No A relation appears to combine with C, supporting absence of CA.']],
'evidence':[
['unsupported','source_specific',red,'Red identification supports C, but this copied W11 coordinate claim does not entail the recorded model_access scope.'],
['unsupported','source_specific',blue,'Blue content at (1,0) is explicit; model_access is an ungrounded scope assignment to a copied location record.'],
['unsupported','source_specific',origin,'The full same-node prefix binds red to the origin, but copied_prior_record with timestamps records provenance rather than presentation/access. It cannot ground O.'],
['unsupported','global_convention',header,'The rows/columns convention specifies how to interpret coordinates globally; it does not state an identified content-specific O status.'],
['unsupported','source_specific',origin,'Separately printing copied provenance does not differentiate the red content from a presentation or access status; D is ungrounded.'],
['unsupported','source_specific',red,'The singleton red coordinate is content only, with no separate status in this D witness.'],
['supported','global_convention',header,'The quoted directions are an explicit global display orientation and support the spatial auxiliary.'],
['unsupported','source_specific',[R('u0002',['kind'],'context_only','copied_prior_record explicitly identifies prior-record reuse.'),R('u0002',['record_time'],'context_only','The source record is from 11.'),R('u0002',['represented_time'],'context_only','The entry represents 12.')],'Prior-record copying is explicit source/process provenance, but the quoted fields do not assert access, so the recorded model_access scope is unsupported.'],
['unsupported','source_specific',red,'Red supplies the COD content operand but its coordinate does not supply O or D.'],
['unsupported','source_specific',origin,'Time 12 and node p0/current/n1 are compatible, but origin metadata is not a content presentation/access status.'],
['unsupported','source_specific',blue,'The blue quote states a location, not the recorded separation_of_content_and_origin attribute. This is wrong-meaning evidence for COD.']],
'bindings':[['unsupported',[['C',[0,1]],['O',[]],['D',[]]],'The red/blue content and copied origin share node p0/current/n1 and represented time 12. C is explicit. The global orientation O quote has no same-content status and the origin O/D quotes concern provenance only. The binding blue quote claims an attribute absent from its words. Compatible prefixes cannot supply the missing O/D meaning.']],
'concerns':['The blue COD evidence is labelled separation_of_content_and_origin although it states only blue coordinates.','Grid orientation and copied provenance are substituted for O/D status meaning.','Model_access scope and absence of location uncertainty are unsupported; no original fields are repaired.'],
'notes':'Complete 10,014-character report inspected in consecutive pages. C, origin and composite quotations resolve to copied_W11 construction units, not analytic-specific content. Header units are global convention.'}
with (p/'spec_2.json').open('x') as f:json.dump(s,f)
