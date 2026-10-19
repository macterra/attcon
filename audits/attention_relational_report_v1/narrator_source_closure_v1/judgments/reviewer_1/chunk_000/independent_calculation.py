"""First independent arithmetic; no report semantics or ownership is inferred here."""
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = Path('audits/attention_relational_report_v1/narrator_source_closure_v1/preparation/chunk_000.json')
EXPECTED = '37c75224b4c27e38a8d3794612867f1808d34169e8d88dab5ec01844d0bf26d4'
raw = INPUT.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED
chunk = json.loads(raw)

def summarize(m):
    cells = [(r, c, p) for r, row in enumerate(m) for c, p in enumerate(row)]
    peak = max(p for r, c, p in cells)
    return {
        'peak_probability': peak,
        'peaks': [[r, c] for r, c, p in cells if p == peak],
        'entropy_nats': -sum(p * math.log(p) for r, c, p in cells if p != 0),
        'expected_row': sum(r*p for r, c, p in cells),
        'expected_column': sum(c*p for r, c, p in cells),
    }

def relation(red, blue):
    pairs = list(itertools.product(red, blue))
    h = ['left_of', 'same_column', 'right_of']
    v = ['above', 'same_row', 'below']
    return {
        'horizontal': [s for i,s in enumerate(h) if any((a[1]>b[1])-(a[1]<b[1])+1 == i for a,b in pairs)],
        'vertical': [s for i,s in enumerate(v) if any((a[0]>b[0])-(a[0]<b[0])+1 == i for a,b in pairs)],
        'coincident_all': all(a == b for a,b in pairs),
        'coincident_possible': any(a == b for a,b in pairs),
    }

results = []
for ri, record in enumerate(chunk['records']):
    entries = []
    comparisons = []
    packet = record['complete_source_packet']
    for pi, panel in enumerate(packet['panels']):
        for view, values in [('current',panel['current'])] + list(panel['candidates'].items()):
            for ni,e in enumerate(values):
                base = ['panels',pi,'current',ni] if view == 'current' else ['panels',pi,'candidates',view,ni]
                recomputed = {role: summarize(m) for role,m in e['distributions'].items()}
                checks = {role: {k: recomputed[role][k] == value for k,value in d.items()} for role,d in e['derived'].items()}
                rr = relation(recomputed['red']['peaks'], recomputed['blue']['peaks'])
                entries.append({'entry_path':base,'chunk_entry_path':['records',ri,'complete_source_packet']+base,
                    'raw_maps':e['distributions'],'origin':e['origin'],'recomputed':recomputed,
                    'supplied':e['derived'],'exact_checks':checks,'recomputed_relation':rr,
                    'supplied_relation':e['red_blue_peak_relation'],'relation_equal':rr == e['red_blue_peak_relation'],
                    'uniform':{role:len(set(p for row in m for p in row))==1 for role,m in e['distributions'].items()},
                    'peak_mass':{role:sum(m[r][c] for r,c in recomputed[role]['peaks']) for role,m in e['distributions'].items()}})
                if view != 'current':
                    comparisons.append({'left_path':base+['distributions'],'right_path':['panels',pi,'current',ni,'distributions'],
                        'maps_equal':e['distributions']==panel['current'][ni]['distributions']})
    if len(packet['panels']) == 2:
        for ni in range(4):
            comparisons.append({'left_path':['panels',0,'current',ni,'distributions'],
                'right_path':['panels',1,'current',ni,'distributions'],
                'maps_equal':packet['panels'][0]['current'][ni]['distributions']==packet['panels'][1]['current'][ni]['distributions']})
    results.append({'id':record['id'],'entries':entries,'equalities':comparisons})

out = {'input_path':str(INPUT),'input_sha256':EXPECTED,
    'script_path':str(Path(__file__).resolve()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'operator':'Exact max and argmax; -sum(p*ln(p)); sum(row*p), sum(column*p), no normalization; Cartesian product of all red and blue maxima; exact full-map equality.',
    'records':results}
with (HERE/'independent_calculation_results.json').open('x') as f:
    json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print('entries', sum(len(r['entries']) for r in results))
print('scalar_exact_mismatches',sum(not b for r in results for e in r['entries'] for checks in e['exact_checks'].values() for b in checks.values()))
print('relation_mismatches',sum(not e['relation_equal'] for r in results for e in r['entries']))
for r in results:
    print(r['id'])
    for e in r['entries']:
        if e['entry_path'][2] == 'current' or e['entry_path'][-1] == 0:
            print(e['entry_path'],e['recomputed'], 'uniform',e['uniform'],'peak_mass',e['peak_mass'])
    print('equalities',r['equalities'])
