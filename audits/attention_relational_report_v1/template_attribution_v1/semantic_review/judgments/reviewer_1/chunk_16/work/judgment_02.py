import runpy,sys,json
from pathlib import Path
p=str(Path(__file__).with_name('review_ops.py'));sys.argv=[p,'2','build'];d=runpy.run_path(p);v=lambda p:d['v'](p)['value']
def P(n,path,role,reason):
 u=v(['property_table',n]);val=u['value']
 for k in path:val=val[k]
 return dict(unit_id=u['unit_id'],value_path=path,value=val,semantic_role=role,reason=reason)
r=P(4,[],'content_operand','Complete current p0/n0 red peaks are (0,1).');b=P(6,[],'content_operand','Complete current p0/n0 blue peaks are (0,3).');fr=P(40,[],'content_operand','Conditional k0/n2 red peaks for represented time 13 are (0,1).');fb=P(42,[],'content_operand','Conditional k0/n2 blue peaks for represented time 13 are (0,2).')
h=P(0,[],'context_only','Authored header convention supplies grid orientation, not a source-specific organization/access status.')
o=[P(2,['kind'],'status_operand','Copied prior record is provenance rather than an organization/access claim.'),P(2,['record_time'],'context_only','Copied record time is 11.'),P(2,['represented_time'],'context_only','Current representation time is 12; p0/current/n0 binds origin to the red and blue entries outside its quote.')]
ho=[P(20,['kind'],'status_operand','History-estimate kind distinguishes how n3 content was obtained.'),P(20,['feedback_through'],'context_only','History estimate uses feedback through 11.'),P(20,['represented_time'],'context_only','History estimate concerns time 12.')]
fo=[P(38,['kind'],'processing_relation_operand','Command-estimate label identifies a conditional estimate, but supplies no relation to organization/access.'),P(38,['feedback_through'],'context_only','Forecast uses feedback through 11.'),P(38,['represented_time'],'context_only','Candidate is a time-13 forecast; common p0/candidate/k0/n2 prefix connects red and blue operands outside origin quote.')]
cr=[]
def C(path,a,n):
 c=next(x for x in v(['components']) if x['path']==path);cr.append(dict(component_path=path,original_status=c['recorded_status'],assessment=a,notes=n))
C(['features','C'],'supported','Current p0/n0 red (0,1) and blue (0,3), plus candidate k0/n2 red (0,1), identify represented content and attributes at their separate current/forecast times.')
C(['features','O'],'unsupported','Grid orientation is global display convention; copied_prior_record with record_time11/represented_time12 states provenance. Neither supplies a bound spatial pattern beyond coordinates or a current access status.')
C(['features','D'],'unsupported','p0/current/n0 content coordinates and copied-record provenance are separate fields, but provenance does not express the organization/presentation/access status required for D.')
C(['features','A'],'unsupported','command_estimate at p0/candidate/k0/n2 and its red coordinate are a conditional forecast, without an allocation/content-processing relation to organization or access. The command label alone cannot qualify.')
C(['auxiliary','basic_spatial_relation'],'supported','Rows down and columns right are explicitly stated in the header.')
C(['auxiliary','identity_or_location_uncertainty'],'supported','Four tied peaks at current n2 are listed without uncertainty language or probabilities; coordinate sets alone do not assert uncertainty.')
C(['auxiliary','answer_confidence'],'supported','No answer confidence is supplied alongside coordinate and origin serialization.')
C(['auxiliary','recovery_or_success_forecast'],'supported','Candidate forecasts list positions, without recovery or success judgments.')
C(['auxiliary','source_process_distinction'],'supported','The explicit copied_prior_record, history_estimate and command_estimate origin kinds distinguish copied history from current estimation and conditional forecasts.')
C(['auxiliary','experiential_assertion'],'supported','No experience or consciousness is asserted by the peak and provenance table.')
C(['composites','COD'],'unsupported','The current p0/n0 coordinate and origin entries identify content but fail to supply qualifying O/D. The header orientation cannot supply missing bound meaning.')
C(['composites','CA'],'unsupported','Conditional k0/n2 content is identified, but command_estimate provenance plus coordinates lacks a content-processing relation to organization/access, so A and CA are unsupported.')
er=[]
def E(path,a,s,p,n):er.append(dict(evidence_path=path,grounding=a,source_meaning=s,properties=p,notes=n))
E(['features','C','evidence',0],'supported','source_specific',[r],'Red p0/current/n0 is identified at (0,1), a copied_W11 property despite the A panel.')
E(['features','C','evidence',1],'supported','source_specific',[b],'Blue p0/current/n0 is identified at (0,3), also copied_W11 provenance.')
E(['features','C','evidence',2],'supported','source_specific',[fr],'Red candidate k0/n2 is identified at (0,1) in source-conditional A forecast; this does not assert current appearance.')
E(['features','O','evidence',0],'unsupported','global_convention',[h],'Rows/columns orientation is a global display convention with no bound content organization/access status.')
E(['features','O','evidence',1],'unsupported','source_specific',[r,b]+o,'Shared p0/current/n0 prefix relates outside-quote peaks to copied provenance at time12. That relation is not O.')
E(['features','D','evidence',0],'unsupported','source_specific',[r]+o,'The full-text same-node prefix binds red (0,1) to origin; provenance is not a qualifying organization/access status.')
E(['features','D','evidence',1],'unsupported','source_specific',[r]+o,'Red identifies content, while same-node origin gives copied-record lineage; a content/provenance distinction does not meet D.')
E(['features','A','evidence',0],'unsupported','source_specific',[fr,fb]+fo,'The shared candidate/k0/n2 prefix explicitly binds outside-quote peaks to command_estimate at time13. No organization/access relation is expressed.')
E(['features','A','evidence',1],'unsupported','source_specific',[fr,fb]+fo,'Red coordinate and the adjacent same-node command-estimate origin describe a forecast, not how processing conditions relate to content organization/access.')
E(['auxiliary','basic_spatial_relation','evidence',0],'supported','global_convention',[h],'Header gives exact row-down, column-right spatial convention.')
E(['auxiliary','source_process_distinction','evidence',0],'supported','source_specific',o,'Copied W11 record provenance is explicitly named, distinct from history/command estimation elsewhere in full text.')
E(['auxiliary','source_process_distinction','evidence',1],'supported','source_specific',ho,'History-estimate origin identifies the source-current A process, feedback cutoff and represented time.')
E(['auxiliary','source_process_distinction','evidence',2],'supported','source_specific',fo,'Source-conditional A command estimate is explicitly a forecast at13 based on feedback through11.')
E(['composites','COD','bindings',0,'evidence',0],'unsupported','source_specific',[r,b]+o,'The full-text same-node prefix binds both peaks to origin at12, but no qualifying O or D is stated.')
E(['composites','COD','bindings',0,'evidence',1],'unsupported','source_specific',[r,b]+o,'Blue coordinates are identified content only; same-node copied provenance does not establish organization/access status.')
E(['composites','COD','bindings',0,'evidence',2],'unsupported','source_specific',[r,b]+o,'Outside-quote red/blue peaks share p0/current/n0 with quoted copied origin. Compatible referent/time cannot replace missing O/D meanings.')
E(['composites','CA','bindings',0,'evidence',0],'unsupported','source_specific',[fr,fb]+fo,'Same candidate/k0/n2 prefix connects outside-quote red/blue peaks with forecast origin; command_estimate is insufficient for A.')
E(['composites','CA','bindings',0,'evidence',1],'unsupported','source_specific',[fr,fb]+fo,'Conditional red peak and same-node origin identify a forecast without processing-to-organization/access relation.')
E(['composites','CA','bindings',0,'evidence',2],'unsupported','source_specific',[fr,fb]+fo,'Conditional blue peak is content, and full-text same-node origin gives time13 forecast provenance; no qualifying A relation.')
br=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0],['features','C','evidence',1]]),dict(feature='O',evidence_paths=[['features','O','evidence',1]]),dict(feature='D',evidence_paths=[['features','D','evidence',0],['features','D','evidence',1]])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',i] for i in range(3)],assessment='unsupported',notes='Current p0/n0 red/blue and copied origin share referent and represented time12; O header evidence concerns the global grid, and copied origin does not assert organization/access. Thus O/D are not grounded despite node/time compatibility.'),dict(binding_path=['composites','CA','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',2]]),dict(feature='A',evidence_paths=[['features','A','evidence',0],['features','A','evidence',1]])],binding_evidence_paths=[['composites','CA','bindings',0,'evidence',i] for i in range(3)],assessment='unsupported',notes='Candidate k0/n2 red/blue and command_estimate share forecast time13. C supports red content; command labeling and coordinates lack processing-to-organization/access meaning, so the A witness cannot establish CA. Current C paths concern n0/time12 and cannot substitute.')]
rj=dict(id=d['i'],unit_id=v(['unit_id']),text_sha256=v(['text_sha256']),reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['Do not make copied W11 properties A-specific because panel_source_key is A.','Preserve time13 candidates as forecasts; grid orientation and peak coordinates do not establish perceptual appearance.'],notes='Complete report, canonical annotation, all12 components, all19 evidence positions, both bindings and every used property unit inspected before saving this first judgment.')
print(json.dumps(d['w'].save_first_record(d['c'],d['i'],d['W'],(json.dumps(rj,indent=2)+'\n').encode())))
