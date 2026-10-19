from author import c,h
import json,sys
n=int(sys.argv[1]);mode=sys.argv[2];r=c['records'][n];i=r['id']
def v(p,offset=0,limit=100):
 z=h.source_view(c,i,p,offset,limit)
 print(json.dumps(dict(path=p,offset=offset,total=z['total_items'],end=z['page_end'],next=z['next_offset'],value=z['value']),separators=(',',':')))
print(i)
if mode=='text':
 for o in range(0,len(r['text']),3000): v(['text'],o,3000)
elif mode=='details':
 for p in [['reference_established'],['unit_id'],['text_sha256']]:v(p)
 if not r['reference_established']:
  for k in ['reference_measurement','components','evidence','bindings']:v([k],0,20)
 else:
  for o in range(0,len(r['components']),4):v(['components'],o,4)
  for j,e in enumerate(r['evidence']):
   v(['evidence',j,'path']);v(['evidence',j,'recorded_evidence'],0,20)
   v(['evidence',j,'geometry','intersections'],0,10)
  for j in range(len(r['bindings'])):v(['bindings',j],0,20)
  for comp in ['features','auxiliary','composites']:
   # Rationale and original binding fields; evidence already individually inspected.
   for name,val in r['canonical_first_annotation'][comp].items():
    if 'rationale' in val:v(['canonical_first_annotation',comp,name,'rationale'],0,2000)
    if 'bindings' in val:
     for j,b in enumerate(val['bindings']):
      for k in ['referent','attribute','scope','time']:v(['canonical_first_annotation',comp,name,'bindings',j,k],0,200)
elif mode=='props':
 for s in sys.argv[3:]:v(['property_table',int(s)],0,30)
