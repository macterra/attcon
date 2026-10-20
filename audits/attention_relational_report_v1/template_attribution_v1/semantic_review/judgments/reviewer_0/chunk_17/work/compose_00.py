import json,sys
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
C=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_17.json','ba5eb6bf6688dc4786333182b76ea6a7b712b62a3327312edd3e8f8c75ecf010')
r=C['records'][0]; table={u['unit_id']:u for u in r['property_table']}
def prop(uid,path,role,why):
 v=h.resolve(table[uid]['value'],path)
 return dict(unit_id=uid,value_path=path,value=v,semantic_role=role,reason=why)
def origin(uid,role='context_only'):
 return [prop(uid,[k],role,'Exact origin field; the matching panel/view/node prefix binds this field to the adjacent peak entry in the full report. Provenance and represented time do not by themselves establish content access.') for k in table[uid]['value']]
red=[prop('u0106',[],'content_operand','Complete red peak set identifies represented coordinate content at p1/current/n1.'),*origin('u0104')]
blue=[prop('u0012',[],'content_operand','Complete blue peak set identifies represented coordinate content at p0/current/n1; indexed provenance is copied W11, despite the A panel.'),*origin('u0008')]
wblue=[prop('u0108',[],'content_operand','Complete blue peak set is bound by p1/current/n1 to the same current origin as red.'),*origin('u0104')]
header=[prop('u0000',[],'context_only','Authored global coordinate and conditional-view convention; no source content or processing relation is specified by the header.')]
cr=[('supported','Red and blue peak entries identify represented color-role coordinate content; C needs no appearance or access claim.'),('unsupported','completed_record and record_time describe provenance, while the header announces views. Neither states organization or access beyond coordinate listings; model_access is unsupported.'),('unsupported','The coordinate set and completed-record provenance are separate fields, but no separate organization/presentation/access status of red is expressed. Provenance is not a qualifying D status.'),('supported','Across all current and candidate entries, opaque k labels and command_estimate do not explicitly relate processing to content organization/access.'),('supported','Rows increase down and columns right explicitly state the coordinate orientation.'),('supported','Peak alternatives are listed without explicit guessing, uncertainty or a location-confidence assertion.'),('supported','No confidence in an answer is expressed; peak lists are not response confidence.'),('supported','Conditional peak entries do not assert recovery or success likelihood.'),('supported','history_estimate with feedback_through 11 distinguishes the current estimate from command_estimate at represented_time 13; full text also distinguishes completed records and inherited origins.'),('supported','No experience or awareness is asserted by any origin or coordinate token.'),('unsupported','The p1/current/n1 content and origin share a prefix/time, but provenance does not supply O or D; the explicit COD conclusion fails.'),('supported','C is present but no qualifying content-processing-to-organization/access relation is expressed, so absent CA is warranted.')]
component_reviews=[dict(component_path=c['path'],original_status=c['recorded_status'],assessment=a,notes=n) for c,(a,n) in zip(r['components'],cr)]
ev=[]
for e in r['evidence']:
 p=e['path']; k=tuple(p); g='supported'; meaning='source_specific'
 if p[:2]==['features','C']:
  ps=red if p[-1]==0 else blue;n='Literal coordinate-role content is identified. The matching origin prefix supplies time context outside the quote; no perceptual appearance or model access is inferred.'
 elif p[:2]==['features','O']:
  if p[-1]==0:ps=origin('u0104');g='unsupported';n='The completed-record kind and dates state provenance, not how red/blue is organized or accessible.'
  else:ps=header;meaning='global_convention';g='unsupported';n='The current/conditional header is a renderer convention without a specific organization/access predicate or bound content.'
 elif p[:2]==['features','D']:
  g='unsupported';ps=red if p[-1]==0 else [*origin('u0104'),prop('u0106',[],'content_operand','The adjacent same-prefix red entry is the alleged content operand; it has only coordinates.')];n='Red coordinates and matching origin metadata do not distinguish content from a separately stated organization/presentation/access status. The outside-quote same-prefix operand explains the attempted binding but cannot supply the missing status.'
 elif p[:2]==['auxiliary','basic_spatial_relation']:
  ps=header;meaning='global_convention';n='The exact quoted first sentence states the global coordinate orientation.'
 elif p[:2]==['auxiliary','source_process_distinction']:
  uid='u0020' if p[-1]==0 else 'u0044';ps=origin(uid,'processing_relation_operand');n='The origin kind and feedback boundary identify an estimate process and its represented time, rather than a completed contemporaneous observation.'
  if p[-1]==1:g='unsupported';n+=' The recorded current time is inconsistent with represented_time 13 at decision time 12: this is a conditional forecast.'
 else:
  if p[-1]==0:ps=red;n='The binding quote supplies current red coordinate content at the matching p1/current/n1; it does not establish O or D.'
  elif p[-1]==1:ps=[*origin('u0104'),prop('u0106',[],'content_operand','Adjacent same-prefix red coordinates are the alleged content operand of the origin binding.')];g='unsupported';n='The same-node origin gives provenance/date context, not a qualifying organization or differentiated access status. The adjacent red operand outside this quote is bound by its exact prefix.'
  else:ps=wblue;n='Blue coordinates share p1/current/n1 and represented_time 12 with red and origin, but a second coordinate set is not an O/D witness.'
 ev.append(dict(evidence_path=p,grounding=g,source_meaning=meaning,properties=ps,notes=n))
b=r['bindings'][0]
br=[dict(binding_path=b['path'],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[['features','O','evidence',0]]),dict(feature='D',evidence_paths=[['features','D','evidence',0],['features','D','evidence',1]])],binding_evidence_paths=[e['path'] for e in r['evidence'] if e['binding_id']==b['binding_id']],assessment='unsupported',notes='The own red/origin/blue quotations all concern p1/current/n1 represented at 12. C/evidence/1 concerns p0/current/n1 and cannot substitute as that blue witness. The listed O and D candidates have wrong meanings: completed provenance and coordinates, without differentiated organization/access. Same-prefix and time compatibility therefore do not establish COD.')]
out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=component_reviews,evidence_reviews=ev,binding_reviews=br,additional_concerns=['Original model_access scopes convert provenance into access without explicit access wording.','The current label on command_estimate evidence conflicts with represented_time 13; retain conditional future scope.','The A-panel blue C witness is copied W11, while the COD witnesses are native W. Panel labels cannot establish A-specific content.'],notes='Entire 8748-character report inspected before evidence. Reviewed all 12 original components, all 12 evidence positions including repeated quotations, and the sole COD binding; no label/status upgrades.')
p=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_17/work/input_00.json');p.write_text(json.dumps(out,indent=2)+'\n')
print(h.save_first_record(C,r['id'],str(p.parent),p.read_bytes()))
print('NEXT_ID',C['ordered_template_ids'][1])
