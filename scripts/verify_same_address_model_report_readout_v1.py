"""Separate deterministic bookkeeping cross-check; never grades report meaning."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import re
from copy import deepcopy

BASE = Path('audits/attention_relational_report_v1/same_address_pilot_v1')
def read(name): return json.loads((BASE/name).read_text())
def sha(raw): return hashlib.sha256(raw).hexdigest()
def flatten(x, path=''):
    if isinstance(x,dict):
        return {k:v for a,b in x.items() for k,v in flatten(b,path+'/'+a).items()}
    if isinstance(x,list):
        return {k:v for a,b in enumerate(x) for k,v in flatten(b,path+'/'+str(a)).items()}
    return {path:x}
def pointer(obj, p):
    for part in p.split('/')[1:]:
        obj=obj[int(part)] if isinstance(obj,list) else obj[part]
    return obj

r=read('reporter_collection.json'); v=read('review_collection.json'); a=read('final_accounting.json'); d=read('final_disposition.json')
assign=read('protocol/assignment_inventory.json')['assignments']; lookup={u['unit_id']:u for u in assign}
assert len(r['units'])==72 and len(v['units'])==288
assert len(a['units'])==72 and len(a['pairs'])==84
assert len(d['unit_bindings'])==72 and len(d['pair_dispositions'])==84
for group in ['all','A','ordinary','analytic','W']:
    ids={u['unit_id'] for u in assign if group=='all' or u['source']==group}
    assert a['summary'][group]['planned']==len(ids)
    for stream in ['R','F1','F2','S1','S2']:
        rows=[u for u in (r['units'] if stream=='R' else v['units']) if u['unit_id'] in ids and u['stream']==stream]
        summary=a['summary'][group]['streams'][stream]
        assert summary['availability']==dict(Counter(u['availability'] for u in rows))
        if stream.startswith('F'):
            assert summary['whole_fidelity']==dict(Counter(u['response']['whole_fidelity'] for u in rows if u['availability']=='annotation_recorded'))
        if stream.startswith('S'):
            assert all(u['availability']!='annotation_recorded' for u in rows)
            assert summary['unknown_or_invalid_positions']==len(ids)
    for comp in ['COD','CA']:
        assert d['composite_summary'][group][comp]['unknown']==len(ids)
        assert d['composite_summary'][group][comp]['full_missingness_uncertainty_bound']==[0,len(ids)]
whole={s:{u['unit_id'] for u in v['units'] if u['stream']==s and u['availability']=='annotation_recorded' and u['response']['whole_fidelity']=='supported'} for s in ['F1','F2']}
assert len(whole['F1'])==len(whole['F2'])==41
assert len(whole['F1'] & whole['F2'])==35==d['factual_interpretation']['both_supported_positions']
assert {u['unit_id'] for u in a['units'] if u['both_factual_whole_supported']}==whole['F1']&whole['F2']
packets={x['id']:x['content'] for x in read('source_run_operational_correction_v1/neutral_packets.json')}
frames={u['unit_id']:packets[u['design_opportunity_id']] for u in assign}
for p in a['pairs']:
    left,right=[frames[p[k]] for k in ['first_unit_id','second_unit_id']]
    patched=deepcopy(left)
    changes=p['input_delta']['changes']
    assert len(changes)==p['input_delta']['changed_leaf_count']
    assert len({row['pointer'] for row in changes})==len(changes)
    for row in p['input_delta']['changes']:
        assert row['first']==pointer(left,row['pointer']) and row['second']==pointer(right,row['pointer'])
        parent,key=row['pointer'].rsplit('/',1)
        target=pointer(patched,parent)
        target[int(key) if isinstance(target,list) else key]=deepcopy(row['second'])
    assert patched==right
    assert p['input_delta']['exact_public_content_equal']==(left==right)
    for stream,comparison in p['assertion_delta'].items():
        for endpoint in ['first','second']:
            u=pointer(a,comparison[endpoint+'_annotation_pointer'])
            assert u['unit_id']==p[endpoint+'_unit_id'] and u['stream']==stream
assert sum(p['input_delta']['exact_public_content_equal'] for p in a['pairs'])==28
for collection in [r,v]:
    obs={x['context_id']:x for x in read(Path(collection['terminal_observations']['path']).name)['observations']}
    assert len(obs)==len(collection['contexts'])
    for c in collection['contexts']:
        assert obs[c['context_id']]['terminal'] and c['terminal_observation']==obs[c['context_id']]
        if c['raw_sha256']:
            import base64
            assert sha(base64.b64decode(c['raw_base64']))==c['raw_sha256']
for name in ['review_collection.json','final_accounting.json']:
    meta=read(name+'.archive.json');original=(BASE/name).read_bytes();compressed=Path(meta['gzip_path']).read_bytes()
    assert sha(original)==meta['source_sha256'] and len(original)==meta['source_size_bytes']
    assert sha(compressed)==meta['gzip_sha256'] and len(compressed)==meta['gzip_size_bytes']
    assert gzip.decompress(compressed)==original
assert d['accounting_sha256']==sha((BASE/'final_accounting.json').read_bytes())
for name in ['reporter_run/manifest.json','review_run/manifest.json']:
    manifest=read(name)
    assert sha(Path(manifest['code']['path']).read_bytes())==manifest['code']['sha256']
    for context in manifest['contexts']:
        for desc in context['allowed_files'].values():
            assert sha(Path(desc['path']).read_bytes())==desc['sha256']
result={'schema':'same_address_pilot_v1_final_verification','status':'passed','performed_by':'root deterministic cross-check, not an independent semantic auditor','planned_reports':72,'planned_review_positions':288,'planned_pairs':84,'available_reports':54,'whole_supported_each_factual_stream':41,'whole_supported_both':35,'valid_semantic_annotations':0,'exact_baseline_restoration_pairs':28,'archive_round_trips_verified':2,'frozen_actor_packet_hashes_verified':True,'first_check_correction':'Initial cross-check incorrectly assumed identical origin metadata keys across controls. It stopped before writing a verification result. Corrected check verifies every recorded replacement against both endpoints and applies the complete change list to reconstruct the second packet exactly. No accounting or first-response bytes changed. changed_leaf_count counts terminal nodes of the recursive comparison; a changed-shape origin object or list is one replacement.','collection_suite':{'command':'PYTHONPATH=/usr/lib/python3/dist-packages .venv/bin/python -m pytest -q tests/test_same_address_model_report_workspace_v1.py','passed':16,'elapsed_seconds':5.78},'scientific_entailment_or_semantic_reliability_tested':False}
with (BASE/'final_verification.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
