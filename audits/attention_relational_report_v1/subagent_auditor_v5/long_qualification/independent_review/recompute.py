"""Independent qualification calculator: allowed local inputs only; no author import.

The narrative predicates and operand inventories below are independently specified.
Selected pure functions from the two permitted public interface sources are executed
only to establish exposure, after independent raw-matrix recomputation.
All outputs use exclusive creation and remain in this script's directory.
"""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from types import SimpleNamespace

BASE = Path('audits/attention_relational_report_v1/subagent_auditor_v5/long_qualification')
OUT = BASE / 'independent_review'
PATHS = [Path('scripts/build_attention_relational_auditor_v5_long_qualification.py'),
         Path('scripts/attention_relational_grounded_auditor_v4.py'),
         Path('scripts/attention_relational_report_v1.py'), BASE / 'authoring_notes.md',
         BASE / 'first_materialization/public_cases.json',
         BASE / 'first_materialization/hidden_atomic_key.json',
         BASE / 'first_materialization/receipt.json']

def enc(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')

def sha(value):
    return hashlib.sha256(value).hexdigest()

def follow(value, path):
    for part in path:
        value = value[part]
    return value

def demand(condition, message):
    if not condition:
        raise ValueError(message)

def path(p, n, role=None, c=None):
    value = ['panels', p] + (['current'] if c is None else ['candidates', c]) + [n]
    return value + (['origin'] if role is None else ['distributions', role])

def origin_expectation(p, n, c=None):
    current = ({'kind': 'completed_record', 'record_time': 40, 'represented_time': 40}
               if p == 1 else {'kind': 'history_estimate', 'represented_time': 40,
                                'feedback_through': 39} if n == 1 else
               {'kind': 'copied_prior_record', 'record_time': 39, 'represented_time': 40})
    if c is None:
        return current
    if n != 2:
        return {'kind': 'inherited_current', 'represented_time': 41, 'inherited_origin': current}
    return ({'kind': 'native_conditional_record', 'represented_time': 41,
             'executed_branch_claim': False} if p == 1 else
            {'kind': 'command_estimate', 'represented_time': 41, 'feedback_through': 39})

def summarize(grid):
    demand(len(grid) == 4 and all(len(row) == 4 for row in grid), 'matrix shape')
    flat = [v for row in grid for v in row]
    demand(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
               for v in flat), 'matrix cell')
    demand(sum(flat) == 1, 'matrix mass')
    demand(list(struct.unpack('<16f', struct.pack('<16f', *flat))) == flat, 'float32')
    largest = max(flat)
    return {'peaks': [[i // 4, i % 4] for i in range(16) if flat[i] == largest],
            'peak_probability': largest,
            'entropy_nats': -math.fsum(v * math.log(v) for v in flat if v != 0),
            'expected_row': math.fsum((i // 4) * flat[i] for i in range(16)),
            'expected_column': math.fsum((i % 4) * flat[i] for i in range(16))}

def relation(red, blue):
    pairs = [(r, b) for r in red for b in blue]
    hs = {'left_of' if r[1] < b[1] else 'right_of' if r[1] > b[1] else 'same_column'
          for r, b in pairs}
    vs = {'above' if r[0] < b[0] else 'below' if r[0] > b[0] else 'same_row'
          for r, b in pairs}
    return {'horizontal': [v for v in ('left_of', 'same_column', 'right_of') if v in hs],
            'vertical': [v for v in ('above', 'same_row', 'below') if v in vs],
            'coincident_possible': any(r == b for r, b in pairs),
            'coincident_all': all(r == b for r, b in pairs)}

def extracted(source, names, namespace):
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    demand({n.name for n in selected} == set(names), 'missing pure interface functions')
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<permitted-pure-functions>', 'exec'), namespace)
    return namespace

raws = {str(p): p.read_bytes() for p in PATHS}
hashes = {name: {'sha256': sha(raw), 'bytes': len(raw)} for name, raw in raws.items()}
public = json.loads(raws[str(PATHS[4])])
key = json.loads(raws[str(PATHS[5])])
author_receipt = json.loads(raws[str(PATHS[6])])
demand(raws[str(PATHS[4])] == enc(public) and raws[str(PATHS[5])] == enc(key), 'JSON policy')
for name in ('public_cases.json', 'hidden_atomic_key.json'):
    demand(author_receipt['files'][name] == hashes[str(BASE / 'first_materialization' / name)], 'receipt file hash')
demand(author_receipt['actual_auditors_run'] is False and
       author_receipt['semantic_key_independently_reviewed'] is False, 'first materialization flags')

# Execute no imports, archive-loading code, build_request, runner, or author code.
ens = extracted(raws[str(PATHS[2])].decode(),
                ['require', '_numbers', 'localization_summary', 'peak_relation'], {'math': math})
exporter = SimpleNamespace(**{n: ens[n] for n in ('localization_summary', 'peak_relation')},
                           ROLES=('red', 'blue'), COMMANDS=('k0', 'k1', 'k2'),
                           NODES=('n0', 'n1', 'n2', 'n3'), PACK_SCHEMA='attention_relational_report_v1_packet')
vns = extracted(raws[str(PATHS[1])].decode(), ['source_contexts', 'partition_spans'],
                {'pipeline': SimpleNamespace(require=demand), 'exporter': exporter,
                 'deepcopy': deepcopy, '_BOUNDARY': re.compile(r'\r\n|[\r\n]|[.!?](?=\s)')})

results = []
for case, gold in zip(public['cases'], key['cases']):
    packet, text = case['packet'], case['report_text']
    demand(case['case_id'] == gold['case_id'], 'case alignment')
    demand(sha(enc(packet)) == gold['packet_sha256'], 'packet hash')
    demand(sha(text.encode()) == gold['report_text_sha256'], 'report hash')
    demand(packet['decision_time'] == 40 and packet['previous_node'] == 'n1' and
           packet['candidate_node'] == 'n2' and packet['queried_role'] == 'red', 'metadata')
    demand(packet['fictional'] is True and packet['fictional_identifier'] == case['case_id'], 'fictional provenance')
    demand(packet['coordinate_system'] == {'rows': 4, 'columns': 4, 'row_direction': 'down',
           'column_direction': 'right', 'separate_local_grids': True}, 'geometry metadata')
    demand(packet['candidate_semantics'] == 'conditional supplied record or estimate, not an executed branch', 'conditional semantics')
    matrices, origins, rels = {}, {}, {}
    for p in range(2):
        demand(packet['panels'][p]['panel'] == f'p{p}', 'panel name')
        demand(set(packet['panels'][p]['candidates']) == {'k0', 'k1', 'k2'}, 'commands')
        for c in (None, 'k0', 'k1', 'k2'):
            for n in range(4):
                pp = path(p, n, c=c)
                node = follow(packet, pp[:-1])
                demand(node['node'] == f'n{n}', 'node address')
                origins[tuple(pp)] = node['origin']
                demand(node['origin'] == origin_expectation(p, n, c), 'complete temporal origin')
                for role in ('red', 'blue'):
                    rp = path(p, n, role, c)
                    calc = summarize(follow(packet, rp))
                    matrices[tuple(rp)] = calc
                    demand(calc == node['derived'][role], 'raw-derived mismatch')
                calc_rel = relation(matrices[tuple(path(p,n,'red',c))]['peaks'],
                                    matrices[tuple(path(p,n,'blue',c))]['peaks'])
                demand(calc_rel == node['red_blue_peak_relation'], 'raw relation mismatch')
                rels[tuple(pp[:-1])] = calc_rel
    demand(len(matrices) == 64 and len(origins) == 32 and len(rels) == 32, 'full traversal')
    contexts = vns['source_contexts'](packet)
    spans = vns['partition_spans'](text)
    demand(len(contexts) == 131, 'public contexts')
    demand(''.join(s['text'] for s in spans) == text, 'exact public text partition')
    bypath = {tuple(c['packet_path']): c for c in contexts if 'packet_path' in c}
    for rp, calc in matrices.items():
        demand(bypath[rp]['calculations'] == calc, 'public role calculation exposure')
    for op, value in origins.items():
        demand(bypath[op]['origin'] == value, 'public origin exposure')

    O = lambda p,n,c=None: path(p,n,c=c)
    R = lambda p,n,role='red',c=None: path(p,n,role,c)
    M = lambda *parts: list(parts)
    oo = lambda p,n,c=None: follow(packet,O(p,n,c))
    peak = lambda p,n,role='red',c=None: matrices[tuple(R(p,n,role,c))]['peaks']
    pair = [R(0,0), R(0,0,'blue')]
    three = [R(0,n) for n in range(3)]
    local = [M('coordinate_system','separate_local_grids')]
    conditional = [M('candidate_semantics')]
    experience = [M('schema'), M('fictional')]
    all_candidates = [O(p,n,c) for p in range(2) for c in ('k0','k1','k2') for n in range(4)]
    allp0 = [O(0,n,c) for c in ('k0','k1','k2') for n in range(4)]
    non2 = [O(0,n,c) for c in ('k0','k1','k2') for n in (0,1,3)]
    n1s = [O(0,1,c) for c in ('k0','k1','k2')]
    columns = [peak(0,n)[0][1] for n in range(3)]
    rows = [peak(0,n)[0][0] for n in range(3)]
    rr, bb = peak(0,0), peak(0,0,'blue')
    distinct_singletons = len(rr) == len(bb) == 1 and rr != bb
    every41 = all(follow(packet,p)['represented_time'] == 41 for p in all_candidates)
    # Whole-packet review establishes the closed supplied inventory: no physical
    # transforms, actual branch observations or experience observations exist.
    closed_inventory = (set(packet) == {'schema','fictional','fictional_identifier','decision_time',
      'previous_node','candidate_node','queried_role','coordinate_system','candidate_semantics','derivation','panels'}
      and all(set(follow(packet,p[:-1])) == {'node','origin','distributions','derived','red_blue_peak_relation'}
              for p in origins))
    demand(closed_inventory, 'absence reasoning requires closed packet review')
    conditional_ok = packet['candidate_semantics'].startswith('conditional supplied')
    undecidable = closed_inventory and conditional_ok
    narrative = []
    def S(ok, operands, reason):
        narrative.append(('supported', bool(ok), operands, reason))
    def C(ok, operands, reason):
        narrative.append(('contradicted', bool(ok), operands, reason))
    def U(ok, operands, reason):
        narrative.append(('not_auditable', bool(ok), operands, reason))

    if case['case_id'] == 'q7m2':
        S(oo(0,1)['represented_time']==40,[O(0,1)],'outer represented_time 40')
        S(oo(0,1)['kind']=='history_estimate',[O(0,1)],'history kind')
        S(oo(0,1)['feedback_through']==39,[O(0,1)],'feedback cutoff 39')
        S(oo(0,0)['kind']=='copied_prior_record',[O(0,0)],'copy kind')
        S(oo(0,0)['record_time']==39,[O(0,0)],'record_time 39')
        S(oo(0,0)['represented_time']==40,[O(0,0)],'represented_time 40')
        S(every41,all_candidates,'all 24 wrappers, not nested origins, represent 41')
        S(oo(0,0,'k1')['inherited_origin']==oo(0,0),[O(0,0,'k1'),O(0,0)],'inheritance equality')
        S(oo(0,0,'k1')['represented_time']==41,[O(0,0,'k1')],'outer 41')
        S(oo(0,0,'k1')['inherited_origin']['record_time']==39,[O(0,0,'k1')],'nested record remains 39')
        S(rr[0][1]<bb[0][1],pair,'red column smaller')
        S(rr[0][0]==bb[0][0],pair,'equal peak rows')
        S(distinct_singletons,pair,'all maxima inspected; distinct singleton sets')
        S(columns==[0,1,2],three,'ordered red columns 0,1,2')
        S(sum(columns)==3,three,'sum local columns 3')
        S(undecidable,local+three,'local arithmetic provides no physical distance')
        S(oo(1,2,'k2')['kind']=='native_conditional_record' and conditional_ok,[O(1,2,'k2')]+conditional,'conditional origin')
        S(oo(1,2,'k2')['executed_branch_claim'] is False and undecidable,[O(1,2,'k2')]+conditional,'claim flag does not observe nonexecution')
        S(undecidable,conditional,'actual execution unresolved by entire supplied packet')
        S(closed_inventory,experience,'actual experience unresolved; no experience observation/rule')
    elif case['case_id'] == 'q4v9':
        S(oo(1,0)['record_time']==40,[O(1,0)],'completed record time 40')
        S(oo(1,0)['represented_time']==40,[O(1,0)],'represented time 40')
        S(oo(1,0,'k0')['represented_time']==41,[O(1,0,'k0')],'outer represented time 41')
        S(oo(1,0,'k0')['kind']=='inherited_current',[O(1,0,'k0')],'inherited wrapper')
        S(oo(1,0,'k0')['inherited_origin']=={'kind':'completed_record','record_time':40,'represented_time':40},[O(1,0,'k0')],'nested completed origin retains both times 40')
        S(every41,all_candidates,'all 24 candidate nodes')
        S(all((follow(packet,p)['kind']=='command_estimate')==(p[-2]==2) for p in allp0),allp0,'exactly n2 directly estimated across all three commands')
        S(all(follow(packet,p)['kind']=='inherited_current' for p in non2),non2,'all nine other p0 candidate nodes inherited')
        S(all(follow(packet,p)['inherited_origin']['feedback_through']==39 for p in n1s),n1s,'all three inherited n1 feedback cutoffs')
        S(rr[0][0]>bb[0][0],pair,'red row larger: below')
        S(rr[0][1]==bb[0][1],pair,'same column')
        S(distinct_singletons,pair,'complete singleton sets differ')
        S(columns==[0,1,2],three,'local column order')
        S(sum(columns)==3,three,'local sum 3')
        S(undecidable,local+three,'no supplied physical alignment')
        S(oo(1,2,'k1')['executed_branch_claim'] is False,[O(1,2,'k1')],'explicit claim flag false')
        S(undecidable,conditional,'actual execution unknown, not false')
        S(closed_inventory,experience,'actual experience unknown, not false')
    elif case['case_id'] == 'q8h1':
        S(oo(0,0)['represented_time']==40,[O(0,0)],'represented time 40')
        S(oo(0,0)['kind']=='copied_prior_record',[O(0,0)],'copied record premise')
        C(oo(0,0)['record_time']==39,[O(0,0)],'40 contradicts supplied record time 39')
        S(oo(0,1)['kind']=='history_estimate' and oo(0,1)['represented_time']==40,[O(0,1)],'history estimate represents 40')
        S(oo(0,1)['feedback_through']==39,[O(0,1)],'feedback cutoff 39')
        S(every41,all_candidates,'all 24 outer candidate times')
        S(oo(0,0,'k2')['kind']=='inherited_current' and oo(0,0,'k2')['inherited_origin']['kind']=='copied_prior_record' and oo(0,0,'k2')['inherited_origin']['record_time']==39,[O(0,0,'k2')],'inherited copied record time 39')
        S(oo(0,0,'k2')['represented_time']==41,[O(0,0,'k2')],'outer represented time 41')
        S(rr[0][1]>bb[0][1],pair,'red column larger: right')
        S(rr[0][0]==bb[0][0],pair,'same row')
        S(distinct_singletons,pair,'distinct singleton full maxima')
        S(columns==[3,2,1],three,'strictly descending local columns')
        S(sum(columns)==6,three,'sum 6')
        S(packet['coordinate_system']['separate_local_grids'] and columns==[3,2,1],local+three,'local index inequality premise supported')
        U(undecidable,local+three,'physical trajectory unobserved; local inequality neither entails nor refutes it')
        S(oo(1,2,'k0')['executed_branch_claim'] is False,[O(1,2,'k0')],'packet claim flag false')
        U(undecidable,[O(1,2,'k0')]+conditional,'actual nonexecution unobserved')
        U(closed_inventory,experience,'actual experience unobserved')
        S(packet['derivation']['peaks'].startswith('all cells exactly') and conditional_ok,[M('derivation','peaks')]+conditional,'ledger is supplied numerical maxima, not a motion observation')
        S(oo(0,2,'k0')['kind']=='command_estimate' and oo(1,2,'k0')['kind']=='native_conditional_record' and conditional_ok,[O(0,2,'k0'),O(1,2,'k0')]+conditional,'conditional future supply is not a completed future record')
    elif case['case_id'] == 'q2x6':
        nested = oo(0,1,'k1')['inherited_origin']
        S(oo(0,1,'k1')['represented_time']==41,[O(0,1,'k1')],'outer time 41')
        S(nested['kind']=='history_estimate',[O(0,1,'k1')],'nested history kind')
        C(nested['represented_time']==40,[O(0,1,'k1')],'nested represents 40, contradicting 41')
        S(nested['feedback_through']==39,[O(0,1,'k1')],'nested cutoff 39')
        S(oo(1,0)=={'kind':'completed_record','record_time':40,'represented_time':40},[O(1,0)],'completed origin and both times 40')
        S(every41,all_candidates,'all 24 wrappers represent 41')
        S(oo(1,2,'k2')['kind']=='native_conditional_record',[O(1,2,'k2')],'conditional kind premise')
        S(oo(1,2,'k2')['executed_branch_claim'] is False,[O(1,2,'k2')],'claim flag false premise')
        C(oo(1,2,'k2')['kind']=='native_conditional_record' and conditional_ok,[O(1,2,'k2')]+conditional,'conditional record explicitly conflicts with completed-observation characterization')
        S(oo(1,2,'k2')['represented_time']==41,[O(1,2,'k2')],'represented time 41 premise kept distinct from false completed status')
        U(undecidable,[O(1,2,'k2')]+conditional,'actual nonexecution unknown')
        S(rr[0][0]<bb[0][0],pair,'red row smaller: above')
        S(rr[0][1]==bb[0][1],pair,'same column')
        S(distinct_singletons,pair,'distinct singleton full maxima')
        S(rows==[0,1,2],three,'local red rows 0,1,2')
        S(sum(rows)==3,three,'local row sum 3')
        S(packet['coordinate_system']['separate_local_grids'],local,'separate grids premise')
        U(undecidable,local+three,'physical displacement and units absent')
        U(closed_inventory,[R(0,0),M('schema')],'numerical localization supplies no experience entailment')
        S(closed_inventory,experience,'whole-packet inspection finds no experience observation')
    else:
        raise ValueError('unrecognized case')

    expected = [('supported', True, [M('coordinate_system','row_direction')], 'row direction down'),
                ('supported', True, [M('coordinate_system','column_direction')], 'column direction right'),
                ('supported', True, local, 'separate local grids'),
                ('supported', len(matrices)==64, [list(p) for p in matrices], '64 complete peak-set rows collectively inventory all roles')]
    ledger_rows = []
    for rp, calc in matrices.items():
        p, _, view, *tail = list(rp)
        # Address reconstructed independently from the path, not the gold statement.
        pi = rp[1]
        c = None if rp[2]=='current' else rp[3]
        ni = rp[3] if c is None else rp[4]
        role = rp[-1]
        prefix = '/'.join([f'p{pi}','current' if c is None else 'candidate'] + ([] if c is None else [c]) + [f'n{ni}',role])
        row = prefix + '={' + ','.join(f'({r},{col})' for r,col in calc['peaks']) + '}'
        ledger_rows.append(row)
        demand(text.count(row)==1, 'independently rebuilt full ledger row')
        expected.append(('supported', True, [list(rp)], 'all 16 cells scanned; exact complete peak set '+repr(calc['peaks'])))
    expected.extend(narrative)
    atoms = gold['mandatory_atomic_assertions']
    demand(len(atoms)==len(expected), 'full mandatory atom count')
    atom_receipts = []
    operands_seen = 0
    for index, (atom, (verdict, ok, paths, reason)) in enumerate(zip(atoms,expected)):
        aid = f'a{index:03d}'
        demand(atom['assertion_id']==aid and atom['mandatory'] is True, 'mandatory atom ID')
        demand(ok and atom['expected_verdict']==verdict, f'{case["case_id"]}/{aid}: independent semantic verdict')
        e = atom['evidence']
        demand(type(e['start']) is int and type(e['end']) is int and 0<=e['start']<e['end']<=len(text), 'evidence bounds')
        demand(text[e['start']:e['end']]==e['text'], 'exact evidence slice')
        demand(text.count(e['text'])==1, 'unique exact evidence text')
        if 4<=index<68:
            demand(e['text']==ledger_rows[index-4], 'gold peak quote versus recomputed raw maxima')
        facts, scopes = atom['required_packet_facts'], atom['required_operands']
        demand(len(facts)==len(scopes)==len(paths), 'required operand/fact count')
        demand(Counter(tuple(f['packet_path']) for f in facts)==Counter(tuple(p) for p in paths), 'independent required operand inventory')
        exposed = []
        for scope, fact in zip(scopes,facts):
            pp = fact['packet_path']
            demand(follow(packet,pp)==fact['value'], 'exact fact value')
            if pp[0]=='panels':
                p = pp[1]; c = None if pp[2]=='current' else pp[3]
                n = pp[3] if c is None else pp[4]
                isrole = pp[-2]=='distributions'
                wanted = dict(scope='role_distribution' if isrole else 'node_origin',panel=f'p{p}',
                    view='current' if c is None else 'candidate',command=c,node=f'n{n}',
                    role=pp[-1] if isrole else None,field='distributions' if isrole else 'origin')
                ctx = bypath[tuple(pp)]
            else:
                wanted = dict(scope='record_metadata',panel=None,view=None,command=None,node=None,
                              role=None,field='.'.join(pp))
                ctx = bypath[()]
                demand(follow(ctx['metadata'],pp)==fact['value'], 'public actual metadata value')
            demand(scope==wanted, 'scope/packet path binding')
            exposed.append(ctx['context_id'])
        operands_seen += len(paths)
        intersect = [s['span_id'] for s in spans if s['start']<e['end'] and e['start']<s['end']]
        demand(intersect, 'quote accessible through public evidence span')
        atom_receipts.append(dict(assertion_id=aid, independently_expected_verdict=verdict,
          semantic_basis=reason, required_operand_count=len(paths), required_packet_paths=paths,
          legal_context_cover=list(dict.fromkeys(exposed)), public_evidence_spans=intersect,
          exact_gold_quote=e, verdict_matches=True, exact_fact_values_match=True, scope_bindings_match=True))
    first_prose = text.index('Although p0/current/n1') if case['case_id']=='q7m2' else text.index('While p1/current/n0') if case['case_id']=='q4v9' else text.index('Although p0/current/n0') if case['case_id']=='q8h1' else text.index('While p0/k1/n1')
    wc = len(text.split())
    demand(wc==gold['word_count'] and 180<=wc<=220, 'measured word count')
    counts = dict(Counter(a['expected_verdict'] for a in atoms))
    demand(gold['clean']==(counts.get('supported',0)==len(atoms)), 'clean/planted category')
    receipt_case = next(r for r in author_receipt['cases'] if r['case_id']==case['case_id'])
    demand(receipt_case==dict(case_id=case['case_id'], word_count=wc, atomic_assertion_count=len(atoms), required_peak_assertion_count=64), 'measured receipt counts')
    results.append(dict(case_id=case['case_id'], word_count=wc,
       narrative_word_count=len(text[first_prose:].split()), mandatory_atoms=len(atoms),
       narrative_atoms=len(narrative), header_atoms=4, complete_peak_atoms=64,
       verdict_counts=counts, matrix_count=len(matrices), origin_count=len(origins),
       local_relation_count=len(rels), required_operand_occurrences=operands_seen,
       legal_source_context_count=len(contexts), public_span_count=len(spans),
       packet_sha256=sha(enc(packet)), report_text_sha256=sha(text.encode()),
       raw_matrix_recomputations=[dict(packet_path=list(p), **calc) for p,calc in matrices.items()],
       full_origin_recomputations=[dict(packet_path=list(p),origin=v) for p,v in origins.items()],
       full_local_relation_recomputations=[dict(packet_path=list(p),relation=v) for p,v in rels.items()],
       atoms=atom_receipts))

demand(len(results)==4 and sum(r['mandatory_atoms'] for r in results)==350, '350 mandatory atoms')
demand(sum(r['matrix_count'] for r in results)==256, '256 complete role sets')
OUT.mkdir(parents=True, exist_ok=True)
receipt = dict(schema='independent_long_qualification_recomputation_v1',
  input_hashes=hashes, calculator_sha256=sha(Path(__file__).read_bytes()),
  all_checks_passed=True, read_scope=[str(p) for p in PATHS],
  excluded_actions=['target reports','other agent outputs','old qualification/answers/keys',
                    'Git','network','API','model calls','subagents','project imports','tests'],
  interface_execution_scope='AST-extracted source_contexts and partition_spans; exporter require/_numbers/localization_summary/peak_relation only. No full build_request, instruction file, runner, project import, API, or auditor execution.',
  actual_auditor_judgments_commissioned=False, cases=results)
with (OUT/'recomputation_receipt.json').open('xb') as f:
    f.write(enc(receipt))
print(json.dumps(dict(all_checks_passed=True, receipt_sha256=sha(enc(receipt)),
  cases=[{k:r[k] for k in ('case_id','word_count','narrative_word_count','mandatory_atoms',
    'narrative_atoms','verdict_counts','required_operand_occurrences','public_span_count')} for r in results],
  input_hashes=hashes),indent=2))
