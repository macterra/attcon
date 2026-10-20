import sys,json,runpy
sys.argv=['session.py','10','view','["reference_established"]'];s=runpy.run_path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_21/work/session.py');r=s['r'];w=s['w'];c=s['c'];i=s['i'];W=s['W'];t={p['unit_id']:p for p in r['property_table']}
def prop(u,path,role,reason):
 v=t[u]['value']
 for p in path:v=v[p]
 return dict(unit_id=u,value_path=path,value=v,semantic_role=role,reason=reason)
def peak():return prop('u0106',[],'content_operand','The native-W n1 red four-location set identifies the represented content; coordinates alone do not state organization/access.')
def orig(u):
 paths=[]
 def walk(v,p):
  if type(v)==dict:
   for k,x in v.items():walk(x,p+[k])
  else:paths.append(p)
 walk(t[u]['value'],[])
 return [prop(u,p,'context_only','Individual origin field states inheritance, provenance or time; it is not a presentation/access-status judgment.') for p in paths]
cs=[('supported','p1/current/n1 red is identified at the four top-left cells.'),('unsupported','Calling the four coordinates presentation_positions adds no literal organization/access claim. The unchanged reference excludes coordinate lists alone.'),('unsupported','completed_record with record_time=represented_time=12 states origin chronology, not separate differentiated presentation/access status for red n1.'),('supported','The report enumerates command alternatives and estimates without relating allocation/processing to content organization/access.'),('supported','The explicit row-down and column-right convention supplies spatial directions.'),('supported','Estimate labels leave the uncertainty interpretation ambiguous: history/command estimates suggest provisional knowledge but no explicit identity/location uncertainty is asserted. Ambiguous is retained.'),('supported','No answer-confidence assertion is present.'),('supported','Conditional peak lists forecast no recovery or success probability.'),('supported','The nested n0 origin explicitly distinguishes inherited_current at 13 from a copied prior record at 11 representing 12. The original evidence time current is a separate grounding defect.'),('supported','No assertion about subjective experience occurs.'),('unsupported','C is explicit for p1 red n1, but duplicated coordinate evidence and completed-record provenance cannot supply O/D semantics.'),('supported','No qualifying A relation is present, warranting CA absence.')]
cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=n) for x,(a,n) in zip(r['components'],cs)]
er=[]
for idx,e in enumerate(r['evidence']):
 p=e['path'];g='unsupported';m='source_specific'
 if idx==0:g='supported';ps=[peak()];note='Native-W p1/current/n1 red top-left four-cell set identifies content at current time 12.'
 elif idx in [1,7,8]:
  ps=[peak()];note='This exact four-cell list identifies positions only. Repeating it under presentation_positions or a COD binding does not provide O or D, even though the p1/current/n1 referent/time match.'
 elif idx in [2,9]:
  ps=orig('u0104')+[peak()];note='The common p1/current/n1 prefix binds the adjacent native-W completed_record origin at time 12 to red n1; this explains the outside-quote operand. Origin chronology is not separate organization/presentation/access status, so '+('D' if idx==2 else 'COD')+' grounding fails.'
 elif idx==3:
  g='supported';m='global_convention';ps=[prop('u0000',[],'context_only','The fixed global header supplies row/column direction conventions.')];note='The initial sentence grounds spatial direction only at display scope.'
 elif idx==4:
  g='unresolved';ps=orig('u0020');note='history_estimate with feedback_through 11 and represented_time 12 is source-specific analytic provenance. An uncertainty reading is possible but no identity/location uncertainty is actually stated, so the ambiguous component has no decisive uncertainty witness.'
 elif idx==5:
  ps=orig('u0032');note='command_estimate is analytic conditional provenance at represented_time 13. The evidence records time current, which conflicts with candidate/future meaning. It also does not explicitly assert identity/location uncertainty.'
 elif idx==6:
  ps=orig('u0026');note='Nested inherited_current at represented_time 13 and inherited copied_prior_record record_time 11 / represented_time 12 explicitly distinguish provenance. Each actual nested field is retained. Recorded evidence time current conflicts with this candidate forecast; provenance does not repair the recorded time.'
 er.append(dict(evidence_path=p,grounding=g,source_meaning=m,properties=ps,notes=note))
br=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',x] for x in range(3)],assessment='unsupported',notes='All own evidence points to p1/current/n1 at represented time 12, and C is supported by native-W red peaks. The duplicate O quote is still coordinate-only and the D quote is provenance-only, leaving no qualifying compatible O/D witnesses.')]
j=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['The COD witnesses here are native W, not analytic-source attention content.','Two auxiliary evidence entries annotate candidate time 13 as current; they must retain that error rather than be silently corrected.','A possible uncertainty implication of estimate labels is distinct from any organization/access status.'],notes='Full report, original rationales, ten evidence positions including duplicate n1 quotes, and nested origin fields reviewed. No status/time repair performed.')
print(json.dumps(w.save_first_record(c,i,W,(json.dumps(j,indent=2)+'\n').encode())))
