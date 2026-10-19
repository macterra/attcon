"""Independent stdlib packet-first numerical verification; no production imports."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
from collections import Counter

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
INPUTS = {
 'bank': ('audits/attention_relational_report_v1/template_attribution_v1/property_pairs/first_pairs.json', 'c944d239cd52863fefa9331fc91800d048362d38d2fd7ee2659dc3b41d5706a5'),
 'index': ('audits/attention_relational_report_v1/template_attribution_v1/construction_index/first_index.json', 'c7b36e2275099ddf5470fb7f5063b85f51d8de401e874f93d65799ec66a747d3'),
 'exports': ('audits/attention_relational_report_v1/run/export/first_exports.json', '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837')}


def encode(x):
 return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def digest(x):
 return hashlib.sha256(encode(x)).hexdigest()


def require(condition, message):
 if not condition:
  raise ValueError(message)


def by_id(rows, key='id'):
 result = {r[key]: r for r in rows}
 require(len(result) == len(rows), 'duplicate inventory '+key)
 return result


def atomic_fields(obj, prefix=()):
 if isinstance(obj, dict):
  require(bool(obj), 'empty origin object')
  for key in sorted(obj):
   yield from atomic_fields(obj[key], prefix+(key,))
 elif isinstance(obj, list):
  require(bool(obj), 'empty origin list')
  for i, value in enumerate(obj):
   yield from atomic_fields(value, prefix+(i,))
 else:
  yield list(prefix), obj


def origin_resolution(context, origin, panel_source, kind):
 chain = [origin]
 while chain[-1]['kind'] == 'inherited_current':
  chain.append(chain[-1]['inherited_origin'])
 terminal = chain[-1]['kind']
 if terminal == 'history_estimate':
  category = 'inherited_source_current' if len(chain)>1 else 'source_current'
 elif terminal == 'command_estimate':
  category = 'source_conditional'
 elif terminal == 'copied_prior_record':
  category = 'copied_W11'
 elif terminal in ('completed_record','counterfactual_native_snapshot','native_conditional_record'):
  category = 'native_W'
 elif terminal == 'fixed_observation_record':
  category = 'sensor'
 else:
  raise ValueError('unknown origin kind '+terminal)
 if category in ('source_current','source_conditional','inherited_source_current'):
  require(panel_source in ('A','ordinary','analytic'), 'estimate source inconsistency')
  source = panel_source
 elif category == 'copied_W11':
  require(panel_source in ('A','ordinary','analytic'), 'copied source inconsistency')
  source = 'W'
 else:
  source = 'W' if category == 'native_W' else 'sensor'
  require(source == panel_source, 'native source inconsistency')
 return dict(binding=dict(context,basis='origin' if kind=='origin' else 'peaks'), state='resolved', panel_source_key=panel_source, origin_chain=chain, origin_class=dict(origin_source_key=source, category=category, view=context['view'], origin_kind=origin['kind'], terminal_origin_kind=terminal))


def maxima(grid):
 require(type(grid) is list and len(grid)==4 and all(type(r) is list and len(r)==4 for r in grid), '4x4 required')
 cells = [(v,r,c) for r,row in enumerate(grid) for c,v in enumerate(row)]
 require(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v,_,_ in cells), 'finite unit interval required')
 require(all(struct.unpack('<f',struct.pack('<f',v))[0]==v for v,_,_ in cells), 'lossless float32 required')
 require(abs(math.fsum(v for v,_,_ in cells)-1)<=1e-6+1e-12, 'unit mass required')
 maximum = sorted(v for v,_,_ in cells)[-1]
 return [[r,c] for v,r,c in cells if v==maximum]


def compare_tables(left,right):
 result=[]
 for identity in sorted(left.keys() | right.keys()):
  a,b=left.get(identity),right.get(identity)
  matched = a is not None and b is not None
  kind=(a if a is not None else b)['identity']['kind']
  equal=encode(a['value'])==encode(b['value']) if matched else None
  raw=(a['raw_distribution_content_sha256']==b['raw_distribution_content_sha256']) if matched and kind=='role_peaks' else None
  result.append(dict(property_identity_sha256=identity, state='matched' if matched else 'first_only' if a is not None else 'second_only', property_kind=kind, value_unchanged=equal, raw_distribution_unchanged=raw, peak_null_with_raw_distribution_change=(equal is True and raw is False) if matched and kind=='role_peaks' else None, source_meaning_established=False, asserted_predicate_tracking=None))
 return result


def counts(rows):
 return dict(total=len(rows),matched=sum(r['state']=='matched' for r in rows),unmatched=sum(r['state']!='matched' for r in rows),value_unchanged=sum(r['value_unchanged'] is True for r in rows),value_changed=sum(r['value_unchanged'] is False for r in rows),peak_null_with_raw_distribution_change=sum(r['peak_null_with_raw_distribution_change'] is True for r in rows))


def restored(rows,raw=False):
 subset=[r for r in rows if r['property_kind']=='role_peaks'] if raw else rows
 field='raw_distribution_unchanged' if raw else 'value_unchanged'
 return len(subset)>0 and all(r['state']=='matched' and r[field] is True for r in subset)


def main():
 loaded={}
 for name,(path,expected) in INPUTS.items():
  raw=(ROOT/path).read_bytes(); require(hashlib.sha256(raw).hexdigest()==expected, 'input hash mismatch '+name)
  loaded[name]=json.loads(raw)
 bank,index,exports=loaded['bank'],loaded['index'],loaded['exports']
 variants=('baseline','current_donor','current_restored','prospective_donor','prospective_restored','W_changed','W_restored')
 scheduled=[]; pair_inventory=[]; baselines={}
 for seed in (1301,1311,1321):
  for world in range(4):
   groups=[(source,view,variants) for source in ('A','ordinary','analytic') for view in ('A_only','joined')]+[('W','W_only',('baseline','W_changed','W_restored')),('sensor','sensor_only',('baseline',))]
   for source,view,arms in groups:
    case=[seed,world,source,view]
    ids={v:f's{seed}_w{world:02d}_t12_{source}_{view}_{v}' for v in arms}
    scheduled.extend(ids.values()); baselines[tuple(case)]=ids['baseline']
    opportunities=[]
    if source in ('A','ordinary','analytic'):
     for intervention in ('current','prospective'):
      opportunities.extend([(intervention+'_donor','baseline',intervention+'_donor'),(intervention+'_donor_to_restored',intervention+'_donor',intervention+'_restored'),(intervention+'_baseline_to_restored','baseline',intervention+'_restored')])
    if source!='sensor':
     opportunities.extend([('W_changed','baseline','W_changed'),('W_changed_to_restored','W_changed','W_restored'),('W_baseline_to_restored','baseline','W_restored')])
    for kind,a,b in opportunities:
     pair_inventory.append(dict(kind=kind,case=case,first_id=ids[a],second_id=ids[b]))
 require(len(scheduled)==552 and len(pair_inventory)==684,'fixed schedule count')
 require([r['id'] for r in exports['requests']]==scheduled==index['ordered_source_ids'],'ordered552 IDs differ')
 require(index['pairs']==pair_inventory,'index pair inventory differs')
 packets=by_id(exports['packets']); provenance=by_id(exports['provenance']); reports=by_id(exports['templates']); indexed=by_id(index['templates'],'source_id')
 require(all(set(x)==set(scheduled) for x in (packets,provenance,reports,indexed)),'552 inventories differ')
 tables={}; index_property_count=0; distribution_count=0; total_origin_fields=0
 for identifier in scheduled:
  packet=packets[identifier]['content']; record=provenance[identifier]; item=indexed[identifier]['index']
  require(packets[identifier]['content_sha256']==digest(packet)==item['packet_content_sha256'],'packet binding '+identifier)
  require(item['provenance_content_sha256']==digest(record),'provenance binding '+identifier)
  require(indexed[identifier]['case']==record['case'],'case binding '+identifier)
  require(item['text']==reports[identifier]['report']['text'],'literal text binding '+identifier)
  require(item['text_sha256']==hashlib.sha256(item['text'].encode()).hexdigest(),'text hash '+identifier)
  units=item['units']; unit_lookup={}; tables[identifier]={}; expected_tokens=['Rows increase down; columns right. Current and conditional candidate peaks follow.']; atoms=[]
  require(''.join(u['text'] for u in units)==item['text'],'index text partition')
  cursor=0
  for n,u in enumerate(units):
   require(u['unit_id']==f'u{n:04d}' and u['start']==cursor and u['end']==cursor+len(u['text']),'index unit positions')
   cursor=u['end']
   if u['kind'] in ('origin','role_peaks'):
    key=(u['kind'],encode(u['context'])); require(key not in unit_lookup,'duplicate index context'); unit_lookup[key]=u
  require(len(record['panels'])==len(packet['panels']),'panel provenance count')
  source_map={p['panel']:p['source_key'] for p in record['panels']}; require(len(source_map)==len(record['panels']),'duplicate panel provenance')
  for pi,panel in enumerate(packet['panels']):
   contexts=[('current',None,panel['current'],['panels',pi,'current'])]
   if panel['candidates'] is not None:
    require(set(panel['candidates'])=={'k0','k1','k2'},'candidate commands')
    contexts.extend(('candidate',command,panel['candidates'][command],['panels',pi,'candidates',command]) for command in ('k0','k1','k2'))
   for view,command,nodes,prefix in contexts:
    require(len(nodes)==4 and {n['node'] for n in nodes}=={'n0','n1','n2','n3'},'complete nodes')
    for ni,node in enumerate(nodes):
     origin=node['origin']; temporal=[dict(path=p,value=v) for p,v in atomic_fields(origin) if p[-1] in ('represented_time','record_time','feedback_through','shown_at','archived_origin_time','decision_time')]
     require(type(origin['represented_time']) is int and type(packet['decision_time']) is int and all(type(t['value']) is int for t in temporal),'integer times')
     path=prefix+[ni]; text_prefix='/'.join([panel['panel'],view]+([] if command is None else [command])+[node['node']])
     for kind,role in [('origin',None),('role_peaks','red'),('role_peaks','blue')]:
      context=dict(panel=panel['panel'],view=view,command=command,node=node['node'],role=role)
      resolution=origin_resolution(context,origin,source_map[panel['panel']],kind)
      unit=unit_lookup.pop((kind,encode(context)))
      property_path=path+(['origin'] if kind=='origin' else ['derived',role,'peaks'])
      raw_path=path+['distributions',role] if kind=='role_peaks' else None
      if kind=='role_peaks':
       value=maxima(node['distributions'][role]); distribution_count+=1
       require(encode(value)==encode(node['derived'][role]['peaks']),'derived maxima differ')
       raw_sha=digest(node['distributions'][role]); fields=[([],value)]
       token=text_prefix+'/'+role+'={'+','.join('('+str(r)+','+str(c)+')' for r,c in value)+'}'
       atoms.append(dict(kind='peaks',panel=panel['panel'],view=view,command=command,node=node['node'],role=role,cells=value))
       require(unit['distribution_packet_path']==raw_path,'index raw operand path')
      else:
       value=origin; raw_sha=None; fields=list(atomic_fields(origin)); total_origin_fields+=len(fields)
       token=text_prefix+'/origin='+encode(origin).decode('ascii')
      expected_id=f'u{2*len(expected_tokens):04d}'; expected_tokens.append(token); index_property_count+=1
      require(unit['unit_id']==expected_id and unit['text']==token,'independent unit reconstruction')
      require(unit['packet_path']==property_path and encode(unit['value'])==encode(value),'indexed property/value')
      require(encode(unit['source_resolution'])==encode(resolution),'indexed actual source resolution')
      for relative,field_value in fields:
       identity=dict(actual_source=resolution['origin_class']['origin_source_key'],context=context,represented_time=origin['represented_time'],decision_time=packet['decision_time'],origin_category=resolution['origin_class']['category'],origin_kind=origin['kind'],terminal_origin_kind=resolution['origin_class']['terminal_origin_kind'],temporal_coordinates=temporal,kind=kind,operator='exact_argmax' if kind=='role_peaks' else 'identity',value_path=relative)
       key=digest(identity); require(key not in tables[identifier],'duplicate independent property identity')
       tables[identifier][key]=dict(identity=identity,value=field_value,construction_unit_id=expected_id,packet_value_path=property_path+relative,raw_distribution_packet_path=raw_path,raw_distribution_content_sha256=raw_sha,source_resolution=resolution)
  require(not unit_lookup,'unconsumed index properties')
  require(' '.join(expected_tokens)==item['text'],'independent complete literal reconstruction')
  require(encode(atoms)==encode(reports[identifier]['report']['propositions']),'independent propositions')
 require(distribution_count==25440 and index_property_count==38160,'complete peak/origin-unit denominator')
 expected_pairs=[]
 for pair in pair_inventory:
  a,b=pair['first_id'],pair['second_id']; rows=compare_tables(tables[a],tables[b]); restoration=None
  if provenance[b]['case']['variant'].endswith('_restored'):
   baseline=baselines[tuple(pair['case'])]; baseline_rows=compare_tables(tables[baseline],tables[b])
   restoration=dict(baseline_id=baseline,second_indexed_properties_equal_baseline=restored(baseline_rows),second_raw_peak_operands_equal_baseline=restored(baseline_rows,True),second_literal_text_equal_baseline=reports[baseline]['report']['text']==reports[b]['report']['text'],comparison_counts=counts(baseline_rows))
  source_rows={}
  for row in rows:
   key=row['property_identity_sha256']; prop=tables[a][key] if key in tables[a] else tables[b][key]
   source_rows.setdefault(prop['identity']['actual_source'],[]).append(row)
  expected_pairs.append(dict(original_pair=pair,properties=rows,counts=counts(rows),by_actual_source={s:counts(rs) for s,rs in sorted(source_rows.items())},literal_text_equal=reports[a]['report']['text']==reports[b]['report']['text'],restoration_to_own_baseline=restoration,semantic_endpoint_tracking_established=False))
 grouped={}
 for pair in expected_pairs:
  kind=pair['original_pair']['kind']; g=grouped.setdefault(kind,dict(pair_count=0,property_counts=Counter())); g['pair_count']+=1; g['property_counts'].update(pair['counts'])
 for g in grouped.values():g['property_counts']=dict(g['property_counts'])
 recomputed=dict(template_opportunities=552,pair_opportunities=684,property_tables=tables,pairs=expected_pairs,summary_by_pair_kind=grouped)
 mismatches=[]
 for field,expected in recomputed.items():
  observed=bank.get(field)
  if encode(expected)!=encode(observed):
   details=[]
   if field=='property_tables':
    for key in sorted(set(expected)|set(observed or {})):
     if encode(expected.get(key))!=encode((observed or {}).get(key)):details.append(key)
   elif field=='pairs':
    for n in range(max(len(expected),len(observed or []))):
     if encode(expected[n] if n<len(expected) else None)!=encode(observed[n] if n<len(observed or []) else None):details.append(n)
   mismatches.append(dict(field=field,expected_sha256=digest(expected),observed_sha256=digest(observed),differing_tables_or_pair_positions=details))
 require(bank['schema']=='attention_relational_template_property_pairs_v1','bank schema')
 require(all(bank[k] is False for k in ('complete_semantic_analysis','semantic_endpoint_tracking_performed','new_independent_observations','reference_or_source_review_judgments_read')),'semantic scope flags')
 for name in ('index','exports'):require(bank['input_dependencies'][name]['sha256']==INPUTS[name][1],'bank input binding '+name)
 output=encode(recomputed)+b'\n'
 with (OUT/'first_recomputed.json.gz').open('xb') as stream:
  with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:compressed.write(output)
 result=dict(all_checks_passed=not mismatches,mismatches=mismatches,templates_verified=len(tables),pairs_verified=len(expected_pairs),properties_verified=sum(map(len,tables.values())),raw_distributions_independently_checked=distribution_count,individual_origin_fields_verified=total_origin_fields,pair_property_rows_verified=sum(len(p['properties']) for p in expected_pairs),restoration_pairs_verified=sum(p['restoration_to_own_baseline'] is not None for p in expected_pairs),recomputation_uncompressed_sha256=hashlib.sha256(output).hexdigest(),recomputation_gzip_sha256=hashlib.sha256((OUT/'first_recomputed.json.gz').read_bytes()).hexdigest(),summary_by_pair_kind=grouped)
 with (OUT/'first_calculation_result.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps(result,sort_keys=True)); require(not mismatches,'independent bank discrepancies retained')


if __name__=='__main__':main()
