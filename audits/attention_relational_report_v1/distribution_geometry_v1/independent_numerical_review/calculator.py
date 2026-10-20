"""Standalone stdlib reconstruction, prepared before opening the actual result.

Method-aware separate computation: production implementation is never imported.
All comparisons and failures are retained from the first invocation.
"""
import copy
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import time
import traceback
from collections import Counter, defaultdict

BASE = Path('audits/attention_relational_report_v1')
HERE = BASE / 'distribution_geometry_v1/independent_numerical_review'
FILES = {
    'exports': (BASE / 'run/export/first_exports.json', '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837'),
    'census': (BASE / 'command_content_diagnostic_v1/first_result.json', '2fc69de904878ddd7ff7432ff7fb3d064dca4dc3a5d14ccd1d4faf0e5e8ff8b6'),
    'pairs': (BASE / 'template_attribution_v1/binding_inventory/first_inventory.json', '9438c05913719a4e94d7e78341d3a9a629226e40aac323322de49850d4afaa02'),
    'result': (BASE / 'distribution_geometry_v1/first_result.json', '1186e955ca9fc2e5228381d3e4dbe0610cf9dc73cc987397f0ac1fda3eb7307c'),
}
SOURCE = Path('scripts/attention_model_distribution_geometry_v1.py')
CONTRACT = Path('docs/ATTENTION_MODEL_DISTRIBUTION_GEOMETRY_V1.md')
COUNTS = Counter()
ERRORS = []
MAX_ERROR = {'absolute': 0.0, 'scaled': 0.0, 'path': None}
BOUND = 256 * sys.float_info.epsilon


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(condition, path, detail):
    COUNTS['invariant_checks'] += 1
    if not condition:
        ERRORS.append({'path': path, 'detail': detail})


def compare(actual, expected, path, numeric=False):
    COUNTS['tree_nodes_checked'] += 1
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            check(False, path, 'expected object'); return
        check(set(actual) == set(expected), path, 'object key set differs')
        for key in expected.keys() & actual.keys():
            compare(actual[key], expected[key], path + '/' + str(key), numeric)
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            check(False, path, 'expected list'); return
        check(len(actual) == len(expected), path, 'list length differs')
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, path + '/' + str(index), numeric)
    elif numeric and type(expected) in (float, int) and type(actual) in (float, int):
        COUNTS['float64_metric_leaves'] += 1
        delta = abs(actual - expected)
        scale = max(1.0, abs(actual), abs(expected))
        if delta > MAX_ERROR['absolute']:
            MAX_ERROR.update(absolute=delta, path=path)
        MAX_ERROR['scaled'] = max(MAX_ERROR['scaled'], delta / scale)
        check(math.isfinite(actual) and delta <= BOUND * scale, path,
              {'actual': actual, 'expected': expected, 'absolute_error': delta, 'allowed': BOUND * scale})
    else:
        COUNTS['exact_leaves'] += 1
        check(type(actual) is type(expected) and actual == expected, path,
              {'actual': actual, 'expected': expected})


def read(name):
    path, expected = FILES[name]
    observed = sha(path)
    check(observed == expected, str(path), 'input/result SHA-256 mismatch')
    if observed != expected:
        raise ValueError('hash binding failure: ' + name)
    with path.open() as stream:
        return json.load(stream)


def measure(grid):
    check(isinstance(grid, list) and len(grid) == 4 and all(isinstance(r, list) and len(r) == 4 for r in grid), 'grid', 'not 4x4')
    cells = [x for row in grid for x in row]
    check(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in cells), 'grid', 'invalid probabilities')
    check(all(struct.unpack('f', struct.pack('f', x))[0] == x for x in cells), 'grid', 'not exact float32')
    mass = sum(cells)
    check(mass > 0 and abs(mass - 1) <= 1e-6 + 1e-12, 'grid', 'invalid mass')
    probabilities = [x / mass for x in cells]
    means = [sum(probabilities[4*r+c] * (r, c)[axis] for r in range(4) for c in range(4)) for axis in (0, 1)]
    # Second-moment identity gives a separate route from centered-squares code.
    second_moment = sum(probabilities[4*r+c] * (r*r+c*c) for r in range(4) for c in range(4))
    maximum = max(cells)
    result = {'mass': mass, 'mass_deviation': mass - 1,
              'raw_maximum_probability': maximum,
              'complete_peak_set': [[r, c] for r in range(4) for c in range(4) if grid[r][c] == maximum],
              'normalized_entropy_nats': sum(-p * math.log(p) for p in probabilities if p > 0),
              'normalized_centroid': means,
              'normalized_variance_trace': second_moment - sum(x*x for x in means)}
    return result, probabilities


def difference(a, b, ga, gb, pa, pb):
    aa = [v for row in a for v in row]; bb = [v for row in b for v in row]
    centroid = [gb['normalized_centroid'][i] - ga['normalized_centroid'][i] for i in (0, 1)]
    marginal_distances = []
    for axis in (0, 1):
        totals = []
        for p in (pa, pb):
            totals.append([sum(p[4*r+c] for r in range(4) for c in range(4) if (r, c)[axis] == coord) for coord in range(4)])
        marginal_distances.append(sum(abs(x-y) for x, y in zip(*totals)) * .5)
    return {'raw_half_L1': sum(abs(x-y) for x, y in zip(aa, bb)) * .5,
            'maximum_absolute_cell_difference': max(abs(x-y) for x, y in zip(aa, bb)),
            'normalized_total_variation': sum(abs(x-y) for x, y in zip(pa, pb)) * .5,
            'row_marginal_total_variation': marginal_distances[0],
            'column_marginal_total_variation': marginal_distances[1],
            'centroid_displacement': centroid,
            'centroid_L1_displacement': sum(abs(x) for x in centroid),
            'centroid_row_displacement': centroid[0], 'centroid_column_displacement': centroid[1],
            'entropy_delta_nats': gb['normalized_entropy_nats'] - ga['normalized_entropy_nats'],
            'variance_trace_delta': gb['normalized_variance_trace'] - ga['normalized_variance_trace'],
            'raw_maximum_probability_delta': gb['raw_maximum_probability'] - ga['raw_maximum_probability'],
            'complete_peak_sets_equal': ga['complete_peak_set'] == gb['complete_peak_set'],
            'raw_grids_equal': a == b}


def without_command(identity):
    value = copy.deepcopy(identity); value['context']['command'] = None; return value


def origin_identity(node, panel_source, context, decision):
    chain = [node['origin']]
    while 'inherited_origin' in chain[-1]:
        chain.append(chain[-1]['inherited_origin'])
    terminal = chain[-1]['kind']
    if terminal == 'copied_prior_record':
        source, category = 'W', 'copied_W11'
    elif terminal == 'history_estimate':
        source, category = panel_source, 'inherited_source_current'
    elif terminal in ('native_conditional_record', 'counterfactual_native_snapshot', 'completed_record'):
        source, category = 'W', 'native_W'
    elif terminal == 'command_estimate':
        source, category = panel_source, 'source_conditional'
    else:
        raise ValueError('unrecognized registered origin: ' + terminal)
    coordinates = []
    def walk(value, path):
        for key in sorted(value):
            v = value[key]
            if isinstance(v, dict): walk(v, path + [key])
            elif type(v) is int: coordinates.append({'path': path + [key], 'value': v})
    walk(chain[0], [])
    identity = {'actual_source': source, 'context': context, 'decision_time': decision,
                'kind': 'role_peaks', 'operator': 'exact_argmax', 'origin_category': category,
                'origin_kind': chain[0]['kind'], 'represented_time': chain[0]['represented_time'],
                'temporal_coordinates': coordinates, 'terminal_origin_kind': terminal, 'value_path': []}
    resolution = {'binding': {'basis': 'peaks', **context}, 'origin_chain': chain,
                  'origin_class': {'category': category, 'origin_kind': chain[0]['kind'], 'origin_source_key': source,
                                   'terminal_origin_kind': terminal, 'view': 'candidate'},
                  'panel_source_key': panel_source, 'state': 'resolved'}
    return identity, resolution


def score(geometries):
    maxima = [g['raw_maximum_probability'] for g in geometries]
    winner_indices = [i for i in range(3) if maxima[i] == max(maxima)]
    first, second = sorted(maxima)[-2:]
    return {'state': 'comparable', 'raw_scores': maxima,
            'tied_winners': ['k' + str(i) for i in winner_indices],
            'first_index_choice': 'k' + str(min(winner_indices)), 'top_two_margin': second-first,
            'scores_differ': not (maxima[0] == maxima[1] == maxima[2]),
            'all_command_peak_sets_equal': all(g['complete_peak_set'] == geometries[0]['complete_peak_set'] for g in geometries)}


def quantile_table(data):
    values = sorted(data)
    output = {'count': len(values)}
    if values:
        for name, q in [('min', 0), ('q25', .25), ('median', .5), ('q75', .75), ('max', 1)]:
            position = q * (len(values)-1); index = int(position); fraction = position-index
            output[name] = values[index] if not fraction else (1-fraction)*values[index] + fraction*values[index+1]
    return output


def audit():
    # Only independent inputs are loaded first; result is opened after preparation.
    exports = read('exports'); census = read('census'); pair_inventory = read('pairs')
    actual = read('result')
    packets = {p['id']: p for p in exports['packets']}
    provenance = {p['id']: p for p in exports['provenance']}
    check(len(packets) == len(exports['packets']), 'exports', 'duplicate packet IDs')
    compare([p['id'] for p in actual['positions']], [p['id'] for p in census['positions']], 'position order')
    compare(len(actual['positions']), 552, 'position count')
    check(set(packets) == {p['id'] for p in census['positions']}, 'inventory', 'packet/census ID set differs')
    accum = defaultdict(lambda: defaultdict(list)); selection = defaultdict(Counter); query = {}; panel_names = {}
    for cp, rp in zip(census['positions'], actual['positions']):
        pid = cp['id']; where = 'position/' + pid; packet = packets[pid]['content']
        COUNTS['positions'] += 1
        compare(digest(packet), packets[pid]['content_sha256'], where + '/packet hash')
        compare(set(rp), {'id', 'panels', 'groups'}, where + '/keys')
        compare(rp['panels'], cp['panels'], where + '/panels')
        check(len(cp['groups']) == len(rp['groups']), where, 'group inventory length')
        panel_names[pid] = {p['panel'] for p in cp['panels']}; query[pid] = {}
        panel_lookup = {p['panel']: i for i, p in enumerate(packet['panels'])}
        source_lookup = {p['panel']: p['source_key'] for p in provenance[pid]['panels']}
        for panel in cp['panels']:
            COUNTS['panels'] += 1; COUNTS['unavailable_panels'] += not panel['conditional_content_available']
            indices = [i for i, g in enumerate(cp['groups']) if g['panel'] == panel['panel']]
            compare(panel['group_indices'], indices, where + '/panel group indices')
            expected = {(n, r) for n in ('n0','n1','n2','n3') for r in ('red','blue')} if panel['conditional_content_available'] else set()
            check({(cp['groups'][i]['node'], cp['groups'][i]['role']) for i in indices} == expected, where, 'complete node-role inventory')
            check((packet['panels'][panel_lookup[panel['panel']]]['candidates'] is not None) == panel['conditional_content_available'], where, 'availability differs from export')
        for cg, rg in zip(cp['groups'], rp['groups']):
            COUNTS['groups'] += 1; gp = where + '/' + cg['panel'] + '/' + cg['node'] + '/' + cg['role']
            compare(set(rg), {'panel','node','role','common_identity','commands','comparisons','packet_candidate_node_and_queried_role','score_rule_reconstruction'}, gp + '/keys')
            for name in ('panel','node','role'): compare(rg[name], cg[name], gp + '/' + name)
            compare(set(cg['commands']), {'k0','k1','k2'}, gp + '/census commands')
            compare(set(rg['commands']), {'k0','k1','k2'}, gp + '/result commands')
            grids=[]; geometries=[]; probabilities=[]; identities=[]
            pi = panel_lookup[cg['panel']]
            for command in ('k0','k1','k2'):
                COUNTS['grids'] += 1; prop = cg['commands'][command]; record = rg['commands'][command]
                nodes = packet['panels'][pi]['candidates'][command]
                ni = [n['node'] for n in nodes].index(cg['node']); node=nodes[ni]
                grid = node['distributions'][cg['role']]
                path = ['panels', pi, 'candidates', command, ni, 'distributions', cg['role']]
                value_path = ['panels', pi, 'candidates', command, ni, 'derived', cg['role'], 'peaks']
                context = {'panel':cg['panel'],'node':cg['node'],'role':cg['role'],'command':command,'view':'candidate'}
                identity, resolution = origin_identity(node, source_lookup[cg['panel']], context, packet['decision_time'])
                compare(prop['identity'], identity, gp + '/' + command + '/origin identity')
                compare(prop['source_resolution'], resolution, gp + '/' + command + '/source resolution')
                compare(identity['decision_time'], 12, gp + '/decision time'); compare(identity['represented_time'], 13, gp + '/represented time')
                compare(prop['packet_value_path'], value_path, gp + '/peak path')
                measure_value, normalized = measure(grid)
                compare(prop['value'], measure_value['complete_peak_set'], gp + '/retained peaks')
                compare(node['derived'][cg['role']]['peaks'], measure_value['complete_peak_set'], gp + '/export peaks')
                for bound in (prop, record):
                    compare(bound['raw_distribution_packet_path'], path, gp + '/grid path')
                    compare(bound['raw_distribution_content_sha256'], digest(grid), gp + '/grid hash')
                compare(set(record), {'grid','geometry','bank_key','raw_distribution_packet_path','raw_distribution_content_sha256'}, gp + '/command keys')
                compare(record['grid'], grid, gp + '/grid exact')
                compare(record['bank_key'], prop['bank_key'], gp + '/bank key')
                compare(record['geometry'], measure_value, gp + '/geometry', True)
                compare(record['geometry']['raw_maximum_probability'], max(max(r) for r in grid), gp + '/exact raw maximum')
                compare(record['geometry']['complete_peak_set'], measure_value['complete_peak_set'], gp + '/exact peaks')
                grids.append(grid); geometries.append(measure_value); probabilities.append(normalized); identities.append(without_command(identity))
            all_comparable = all(identity == identities[0] for identity in identities)
            compare(rg['common_identity'], identities[0] if all_comparable else None, gp + '/common identity')
            requested = cg['node'] == packet['candidate_node'] and cg['role'] == packet['queried_role']
            compare(rg['packet_candidate_node_and_queried_role'], requested, gp + '/query mark')
            rule = score(geometries) if all_comparable else {'state':'unmatched','reason':'Command alternatives have different source/time identities; no qualified preference reconstructed.'}
            compare(rg['score_rule_reconstruction'], rule, gp + '/score rule exact')
            COUNTS['score_rules'] += 1; COUNTS['unmatched_score_rules'] += not all_comparable
            if requested:
                check(cg['panel'] not in query[pid], gp, 'duplicate query panel'); query[pid][cg['panel']] = (identities[0] if all_comparable else None, rule, geometries)
                key = identities[0]['actual_source'] + '/' + identities[0]['origin_category'] if all_comparable else 'unmatched'
                counter = selection[key]; counter['candidate_query_groups'] += 1; counter['unmatched_score_rule'] += not all_comparable
                if all_comparable:
                    counter['unique_winner'] += len(rule['tied_winners']) == 1
                    counter['all_peak_sets_equal'] += rule['all_command_peak_sets_equal']
                    counter['equal_peak_sets_scores_differ'] += rule['all_command_peak_sets_equal'] and rule['scores_differ']
                    counter['choice_' + rule['first_index_choice']] += 1
            compare(len(rg['comparisons']), 3, gp + '/comparison count')
            for actual_pair, (i,j) in zip(rg['comparisons'], [(0,1),(0,2),(1,2)]):
                COUNTS['command_pairs'] += 1; comparable = identities[i] == identities[j]
                metric = difference(grids[i],grids[j],geometries[i],geometries[j],probabilities[i],probabilities[j]) if comparable else None
                expected_pair={'first_command':'k'+str(i),'second_command':'k'+str(j),'state':'comparable' if comparable else 'unmatched','common_identity':identities[i] if comparable else None,'metrics':metric}
                compare({k:v for k,v in actual_pair.items() if k != 'metrics'}, {k:v for k,v in expected_pair.items() if k != 'metrics'}, gp + '/pair identity')
                compare(actual_pair['metrics'], metric, gp + '/pair metrics', True)
                if comparable:
                    COUNTS['comparable_command_pairs'] += 1
                    key = identities[i]['actual_source'] + '/' + identities[i]['origin_category']
                    for subset in ('all','equal_peaks' if metric['complete_peak_sets_equal'] else 'different_peaks'):
                        for name,value in metric.items():
                            if type(value) in (float,int): accum[key+'/'+subset][name].append(value)
    compare(len(actual['original_pairs']), len(pair_inventory['pairs']), 'original pair inventory length')
    for original, retained in zip(pair_inventory['pairs'],actual['original_pairs']):
        COUNTS['original_pairs'] += 1; pair=original['original_pair']; pid=pair['first_id']; qid=pair['second_id']
        expected=[]
        for panel in sorted(panel_names[pid] | panel_names[qid]):
            COUNTS['original_pair_panel_opportunities'] += 1
            x=query[pid].get(panel); y=query[qid].get(panel)
            matched = x is not None and y is not None and x[0] is not None and y[0] is not None and x[0] == y[0]
            COUNTS['original_pair_comparable_panels' if matched else 'original_pair_unmatched_or_unavailable_panels'] += 1
            if x is None and y is None: COUNTS['both_sides_unavailable_pair_panels'] += 1
            expected.append({'panel':panel,'actual_source':x[0]['actual_source'] if matched else None,
                             'state':'comparable' if matched else 'unmatched_or_unavailable',
                             'scores_changed':x[1]['raw_scores'] != y[1]['raw_scores'] if matched else None,
                             'reconstructed_choice_changed':x[1]['first_index_choice'] != y[1]['first_index_choice'] if matched else None,
                             'command_peak_sets_equal':all(x[2][i]['complete_peak_set'] == y[2][i]['complete_peak_set'] for i in range(3)) if matched else None})
        compare(retained, {'original_pair':pair,'literal_template_equal':original['literal_text_equal'],'candidate_query_panels':expected}, 'original pair/'+str(COUNTS['original_pairs']))
    expected_summary={'distribution_metrics':{stratum:{name:quantile_table(values) for name,values in metrics.items()} for stratum,metrics in accum.items()},'candidate_query_score_rule':{k:dict(v) for k,v in selection.items()}}
    compare(actual['summary'],expected_summary,'summary',True)
    # Counts, score decisions and all scalar count summaries must also match exactly.
    compare(actual['summary']['candidate_query_score_rule'],expected_summary['candidate_query_score_rule'],'exact selection summary')
    for stratum,metrics in expected_summary['distribution_metrics'].items():
        COUNTS['summary_strata'] += 1
        for name,stats in metrics.items():
            COUNTS['summary_metric_quantile_tables'] += 1
            compare(actual['summary']['distribution_metrics'][stratum][name]['count'],stats['count'],'exact quantile count')
    counts_expected={'positions':552,'panels':804,'unavailable_panels':12,'groups':6336,'grids':19008,'command_pairs':19008,'score_rules':6336,'original_pairs':684}
    for key,value in counts_expected.items(): compare(COUNTS[key],value,'fixed inventory/'+key)
    metadata={'schema':'attention_model_distribution_geometry_v1','inputs':{name:{'path':str(path),'sha256':expected} for name,(path,expected) in FILES.items() if name != 'result'},'source_sha256':sha(SOURCE),'contract_sha256':sha(CONTRACT),'analysis_type':'informed_development_not_new_system_evidence','source_grids_modified':False,'normalization_for_geometry_only':True,'new_model_or_API_calls':0,'semantic_verdict':None,'goal_achieved':False}
    compare({k:v for k,v in actual.items() if k not in ('positions','original_pairs','summary')},metadata,'metadata')


def main():
    receipt_path=HERE/'first_execution.json'
    if receipt_path.exists(): raise SystemExit('First receipt already exists; refusing rerun')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat(); t0=time.monotonic(); fatal=None
    try: audit()
    except BaseException:
        fatal=traceback.format_exc(); print(fatal,flush=True)
    result={'schema':'attention_distribution_geometry_independent_reconstruction_v1','started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t0,'calculator_sha256':sha(Path(__file__)),'bindings':{name:{'path':str(path),'expected_sha256':expected} for name,(path,expected) in FILES.items()},'counts':dict(COUNTS),'error_count':len(ERRORS),'errors':ERRORS,'fatal_error':fatal,'maximum_float64_error':MAX_ERROR,'tolerance':{'scaled_bound':BOUND,'formula':'256 * sys.float_info.epsilon * max(1, abs(actual), abs(expected))','reason':'Each 16-cell statistic uses at most a few dozen float64 additions/products; independent ordinary sums and second-moment variance introduce roundoff versus compensated centered-squares computation. 256 eps covers those operations and interpolated quantiles at 4x4 bounded scales. Raw grids, hashes, identities, raw maxima, selector scores, exact ties, labels and choices are compared exactly.'},'scope':'Independent computation after method-code review; not blinded methodology, semantic review, or theory evidence.','passed':fatal is None and not ERRORS}
    with receipt_path.open('x') as stream: json.dump(result,stream,indent=2,sort_keys=True,default=repr); stream.write('\n')
    print(json.dumps({'passed':result['passed'],'counts':dict(COUNTS),'error_count':len(ERRORS),'fatal_error':fatal,'maximum_float64_error':MAX_ERROR}),flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__': main()
