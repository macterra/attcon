import runpy,sys,json
from pathlib import Path
class Review:
 def __init__(self,n):
  p=str(Path(__file__).with_name('review_ops.py'));sys.argv=[p,str(n),'build'];self.d=runpy.run_path(p);self.cr=[];self.er=[];self.br=[]
 def v(self,p):return self.d['v'](p)['value']
 def P(self,n,path,role,reason):
  u=self.v(['property_table',n]);val=u['value']
  for k in path:val=val[k]
  return dict(unit_id=u['unit_id'],value_path=path,value=val,semantic_role=role,reason=reason)
 def C(self,path,a,n):
  c=next(x for x in self.v(['components']) if x['path']==path);self.cr.append(dict(component_path=path,original_status=c['recorded_status'],assessment=a,notes=n))
 def E(self,path,a,s,p,n):self.er.append(dict(evidence_path=path,grounding=a,source_meaning=s,properties=p,notes=n))
 def B(self,path,fw,ep,a,n):self.br.append(dict(binding_path=path,feature_witnesses=[dict(feature=f,evidence_paths=paths) for f,paths in fw],binding_evidence_paths=ep,assessment=a,notes=n))
 def save(self,concerns,notes):
  r=dict(id=self.d['i'],unit_id=self.v(['unit_id']),text_sha256=self.v(['text_sha256']),reference_established=True,component_reviews=self.cr,evidence_reviews=self.er,binding_reviews=self.br,additional_concerns=concerns,notes=notes)
  print(json.dumps(self.d['w'].save_first_record(self.d['c'],self.d['i'],self.d['W'],(json.dumps(r,indent=2)+'\n').encode())))
