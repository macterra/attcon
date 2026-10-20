import sys,json,runpy
sys.argv=['session.py','5','view','["reference_established"]'];s=runpy.run_path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_21/work/session.py');r=s['r'];w=s['w'];c=s['c'];i=s['i'];W=s['W'];t={p['unit_id']:p for p in r['property_table']}
def prop(u,path,role,reason):
 v=t[u]['value']
 for p in path:v=v[p]
 return dict(unit_id=u,value_path=path,value=v,semantic_role=role,reason=reason)
def peaks(u):return prop(u,[],'content_operand','The complete peak set identifies the named colour and node location(s); it does not express availability or manner of representation.')
def origin(u):
 return [prop(u,[k],'context_only','Exact origin provenance/time field; the matching p0/node and current/candidate prefix binds this field to that node\'s content in the full text. Provenance is not an access or presentation rating.') for k in t[u]['value']]
cs=[('supported','Named red content is identified at current n0 (0,3), inherited candidate n0 (0,3), and current n3 (0,1); C permits coordinate content.'),('unsupported','The cited single positions and all-sixteen peak list state coordinates only. No separate organized presentation or access status is stated.'),('unsupported','copied_prior_record, history_estimate and command_estimate distinguish origin provenance/time. These generic estimate labels do not separately rate the content\'s organization, presentation or access.'),('supported','Candidate command labels and equal n3 current/candidate peaks do not explicitly relate processing allocation to content organization or access.'),('supported','Rows increase down and columns right explicitly specifies a spatial coordinate convention.'),('supported','Peak lists include ties, but no statement explicitly judges identity/location uncertainty; no uncertainty wording is supplied.'),('supported','No answer choice confidence is reported.'),('supported','Conditional candidate records provide location sets without a recovery/success forecast.'),('supported','History estimates through feedback 11, command estimates at represented time 13 and copied records at time 11 explicitly distinguish source/process provenance.'),('supported','No experience, feeling or consciousness assertion occurs.'),('unsupported','Its n0 red coordinate and copied-prior origin identify content and provenance, but neither gives O/D access or presentation witnesses; therefore the explicit COD binding fails.'),('supported','There is no explicit processing-to-organization/access relation to supply A, so CA absence is warranted.')]
cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=n) for x,(a,n) in zip(r['components'],cs)]
er=[]
for e in r['evidence']:
 p=e['path'];g='supported';m='source_specific';ps=[];notes=''
 if p[:2]==['auxiliary','basic_spatial_relation']:
  m='global_convention';ps=[prop('u0000',[],'context_only','Authored global header states coordinate directions; no source-specific colour content or attention relation is expressed.')];notes='The initial row/column direction sentence supports only the coordinate convention at display scope.'
 elif p[0]=='features' and p[1]=='C':
  u=['u0004','u0028','u0022'][p[-1]];ps=[peaks(u)];notes='The quoted named red peak set identifies represented content at the recorded node/time. Source provenance is '+t[u]['source_resolution']['origin_class']['category']+'; this is not an appearance or access-quality claim.'
 elif p[0]=='features' and p[1]=='O':
  g='unsupported';u=['u0004','u0010','u0024'][p[-1]];ps=[peaks(u)];notes='The quote is a peak-coordinate enumeration, including a full 4x4 tie set at n1. Coordinates alone do not state organization/presentation/access status under O; the recorded scope does not add that meaning.'
 elif (p[0]=='features' and p[1]=='D') or (p[0]=='auxiliary' and p[1]=='source_process_distinction'):
  u=(['u0002','u0020','u0044'] if p[0]=='features' else ['u0020','u0044','u0002'])[p[-1]];ps=origin(u)
  if p[0]=='features':
   g='unsupported';v={'u0002':'u0004','u0020':'u0022','u0044':'u0046'}[u];ps.append(peaks(v));notes='The full text joins this origin and red peaks by the same exact p0/node/current-or-k0 prefix. The content operand lies outside the quote. Origin kind and feedback/record/represented times describe provenance, not a differentiated organization/presentation/access status, so D is unsupported.'
  else:
   notes='The origin kind and its distinct feedback/record and represented times explicitly state source/process provenance at the recorded current or conditional node. They do not describe felt quality.'
 else:
  g='unsupported'
  if p[-1]==0:ps=[peaks('u0004')]
  else:ps=origin('u0002')+[peaks('u0004')]
  notes='The same p0/current/n0 prefix binds copied-prior origin to the red (0,3) content at represented time 12; the outside-quote operand is explicit in the adjacent full text. These are content/provenance facts only, with no O/D access or presentation status, so they cannot ground COD.'
 er.append(dict(evidence_path=p,grounding=g,source_meaning=m,properties=ps,notes=notes))
br=[dict(binding_path=['composites','COD','bindings',0],feature_witnesses=[dict(feature='C',evidence_paths=[['features','C','evidence',0]]),dict(feature='O',evidence_paths=[]),dict(feature='D',evidence_paths=[])],binding_evidence_paths=[['composites','COD','bindings',0,'evidence',0],['composites','COD','bindings',0,'evidence',1]],assessment='unsupported',notes='n0 red (0,3) and its copied-prior provenance share current represented time 12, supporting C only. The O coordinate witness and D provenance witness do not have the required status meaning; n1 red and n3 blue do not rescue this n0 red binding.')]
j=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=['Do not convert copied W11 properties into ordinary-source attention content merely because the panel source key is ordinary.','Do not convert candidate coordinates or command_estimate provenance into an allocation-to-access relation.','Model_access and model_organization labels on coordinate-only quotes exceed the literal semantic content.'],notes='Full report read. Source-specific attribution follows the indexed copied_W11/source_current/source_conditional provenance; no annotation status or scope has been repaired.')
raw=(json.dumps(j,indent=2)+'\n').encode();print(json.dumps(w.save_first_record(c,i,W,raw)))
