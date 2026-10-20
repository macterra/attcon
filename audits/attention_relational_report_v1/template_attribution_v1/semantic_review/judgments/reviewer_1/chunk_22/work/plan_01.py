import json
from pathlib import Path
P=Path(__file__).with_suffix('.json')
def peak(u,reason):return [u,[],'content_operand',reason]
def origin(u,fields,reason):return [[u,[f],'context_only',reason] for f in fields]
h=origin('u0020',['kind','feedback_through','represented_time'],'History estimation through 11 represents time 12; this is provenance, not a separately asserted organization/access status. Same p0/current/n3 prefix binds it to red u0022 outside the origin quote.')
c=origin('u0044',['kind','feedback_through','represented_time'],'Command estimate represents time 13 conditional on k0, not current time. Same p0/candidate/k0/n3 binds red u0046 outside the quote; estimate label supplies no organization status.')
w=origin('u0002',['kind','record_time','represented_time'],'Copied record at 11 representing 12; same p0/current/n0 binds red u0004 outside the quote. Distinguishes provenance but states no presentation/access property.')
p=dict(components=[
['supported','Red at (0,1), blue at (1,2) for p0/current/n3 and red at (0,3) for n0 identify represented contents and location attributes.'],
['unsupported','history_estimate, command_estimate and copied_prior_record label derivation. They do not describe a spatial/constituent pattern or differentiated organization/access status; candidate k0/n3 represents time 13 and is mislabeled current.'],
['unsupported','Separately printing red coordinates and history_estimate provenance does not separately attribute an organization/presentation/access status to red.'],
['supported','No attention/allocation/content-processing condition is related to organization or access; k0 command labels and peaks alone do not supply that relation.'],
['supported','Rows down and columns right explicitly define spatial orientation.'],
['supported','The report lists peak sets, including ties, but does not explicitly hedge identity or location; estimator provenance alone is not expressed uncertainty.'],
['supported','No answer confidence or response certainty is stated.'],
['supported','Conditional candidate peak lists provide no recovery or success probability or forecast claim.'],
['supported','copied_prior_record at record_time 11 versus represented_time 12 explicitly distinguishes record provenance from represented content time.'],
['supported','No claim of experience or consciousness appears.'],
['unsupported','The n3 red content is identified, but its origin kind is not an O or D witness. Shared prefix and time cannot establish COD without those meanings.'],
['supported','C is present; no qualifying content-attention relation or binding is stated.']
],evidence=[
['supported','source_specific',[peak('u0022','Exact red peak set for p0/current/n3, ordinary source current history estimate.')],'Identifies red location at current represented time 12; no appearance claim.'],
['supported','source_specific',[peak('u0024','Exact blue peak set for the same n3 current history estimate.')],'Identifies blue location at current represented time 12.'],
['supported','source_specific',[peak('u0004','Exact red peak for n0, indexed provenance copied W11 despite the ordinary panel.')],'Content identification is grounded in copied W11; panel identity does not make this an ordinary-specific estimate.'],
['unsupported','source_specific',h+[peak('u0022','Outside-quote red operand bound by identical current n3 prefix.')],'history_estimate is derivation metadata, not organization or access.'],
['unsupported','source_specific',c+[peak('u0046','Outside-quote red operand bound by identical candidate k0 n3 prefix.')],'command_estimate conveys conditional provenance at 13, not current organization at 12.'],
['unsupported','source_specific',w+[peak('u0004','Outside-quote red operand bound by identical current n0 prefix.')],'Copied provenance is not a differentiated presentation or access status.'],
['unsupported','source_specific',[peak('u0022','Exact content operand; origin u0020 outside quote is linked by current n3 prefix.')]+h,'Coordinates identify red but no separate organization/access status is attributed.'],
['unsupported','source_specific',h+[peak('u0022','Same prefix explicitly binds history origin to the red content outside this quote.')],'Content/provenance separation does not satisfy D.'],
['supported','global_convention',[['u0000',[],'context_only','Complete fixed header; quoted first sentence gives grid orientation, not source-content evidence.']],'Grid direction is an authored global convention; not an A-content witness.'],
['supported','source_specific',w,'The precise copied kind and two times support the source/process provenance distinction.'],
['unsupported','source_specific',[peak('u0022','Binding red content operand; matching n3 origin outside quote is u0020.')]+h,'COD binding has C but no qualifying O or D.'],
['unsupported','source_specific',h+[peak('u0022','Binding red operand outside quote linked by shared p0/current/n3 prefix and represented time 12.')],'Origin identifies how the record was generated, not red presentation/access.']
],bindings=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',0],['composites','COD','bindings',0,'evidence',1]],assessment='unsupported',notes='Same p0/current/n3 prefix joins red (0,1) to history_estimate, feedback_through 11 and represented_time 12. This establishes content and provenance compatibility, but history_estimate is neither an organization/access status nor its differentiation from content. No compatible O/D feature witness exists; candidate k0/n3 also concerns conditional time 13.')],concerns=['The O candidate quote representing 13 is recorded as current; preserve this mismatch.','Copied W11 content is not source-specific to the ordinary estimator merely because it appears in p0.','Generic origin labels must not become presentation or attention relations.'],notes='Complete report, 12 components, all 12 evidence positions and single COD binding inspected. Source-aware model scope only; no source arithmetic required.')
P.write_text(json.dumps(p,indent=2))
