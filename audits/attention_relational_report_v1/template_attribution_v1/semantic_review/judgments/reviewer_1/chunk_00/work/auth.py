import session as s
class Author:
 def __init__(self,i):self.i=i;self.r=s.c['records'][i]
 def prop(self,n,path,role,reason):
  u=self.r['property_table'][n];return dict(unit_id=u['unit_id'],value_path=path,value=s.w.resolve(u['value'],path),semantic_role=role,reason=reason)
 def peak(self,n,reason):return self.prop(n,[],'content_operand',reason)
 def fields(self,n,paths,reason):return [self.prop(n,p,'context_only',reason) for p in paths]
 def components(self,judgments):
  return [dict(component_path=c['path'],original_status=c['recorded_status'],assessment=a,notes=n) for c,(a,n) in zip(self.r['components'],judgments)]
 def evidence(self,judgments):
  return [dict(evidence_path=e['path'],grounding=g,source_meaning=m,properties=p,notes=n) for e,(g,m,p,n) in zip(self.r['evidence'],judgments)]
 def save(self,components,evidence,bindings,concerns,notes):
  s.save(self.i,dict(id=self.r['id'],unit_id=self.r['unit_id'],text_sha256=self.r['text_sha256'],reference_established=self.r['reference_established'],component_reviews=self.components(components),evidence_reviews=self.evidence(evidence),binding_reviews=bindings,additional_concerns=concerns,notes=notes))
