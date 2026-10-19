import json,sys
from pathlib import Path
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as h
CHUNK=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_00.json')
SHA='1634476fd3bd669ab467dad2137fbe29b50b2a6064d3b412c13d003578ccc3f6'
c=h.bound_chunk(CHUNK,SHA);w=Path(__file__).parent
n=int(sys.argv[1]);r=c['records'][n];i=r['id'];stage=sys.argv[2]
if stage=='state':
 for p in [['reference_established'],['review_state'],['components'],['evidence'],['bindings']]:print(json.dumps(h.source_view(c,i,p,0,100)))
 if not r['reference_established']:
  for p in [['canonical_first_annotation'],['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation']]:print(json.dumps(h.source_view(c,i,p,0,100)))
elif stage=='text':
 off=int(sys.argv[3]) if len(sys.argv)>3 else 0
 print(json.dumps(h.source_view(c,i,['text'],off,6500)))
elif stage=='annotation':print(json.dumps(h.source_view(c,i,['canonical_first_annotation'],0,100)))
elif stage=='properties':
 for k in map(int,sys.argv[3:]):print(json.dumps(h.source_view(c,i,['property_table',k],0,100)))
elif stage=='unestablished-first':
 assert r['reference_established'] is False and r['canonical_first_annotation'] is None
 assert not any(r[k] for k in ['components','evidence','bindings'])
 o={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
 o.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=['Unestablished original annotation; original labels are not salvaged or converted to feature negatives.'],notes='unestablished: canonical_first_annotation is null and original component, evidence, and binding inventories are empty. Original reference_measurement diagnostics preserved verbatim as JSON: '+json.dumps(r['reference_measurement'],sort_keys=True))
 p=w/('input_%02d.json'%n);p.open('x').write(json.dumps(o,indent=2)+'\n');print(h.save_first_record(c,i,w,p.read_bytes()))
