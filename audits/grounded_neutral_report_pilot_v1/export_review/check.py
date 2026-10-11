"""Independent standard-library review of saved JSON; never import source modules.

No binary archive/checkpoint reads, inference, fitting, export, or Reporter calls.
First-pass anomalies are accumulated rather than repaired or hidden.
"""
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'audits/grounded_neutral_report_pilot_v1'
TARGETS = BASE/'targets'
OUT = BASE/'export_review'
RECOVERY = Path('/tmp/attcon-review-milestones-vvjs166i')
FREEZE = '9cb69dfd86f5a654bddd822fe9ddcec1b721b546'
SAVED_EXPORT = 'ec65d39537e1b2612c823b65497183464b120ab6'
SEEDS = (1301, 1311, 1321)
ARMS = ('natural', 'memory', 'route', 'memory_route', 'sham', 'restore_memory',
        'restore_route', 'id_remap', 'apparatus', 'tables', 'sensory_only', 'route_only', 'no_external')
LOCS = tuple('l'+str(i) for i in range(4))
CMDS = tuple('k'+str(i) for i in range(4))
GROUPS = ('stored_evidence', 'native_history', 'delivery_forecasts', 'outcome_forecasts', 'external_readouts')
MASKS = {'sensory_only': (9, {'native_history', 'delivery_forecasts', 'outcome_forecasts'}),
         'route_only': (6, {'stored_evidence', 'outcome_forecasts', 'external_readouts'}),
         'no_external': (18, {'external_readouts'})}
ANOMALIES = []
CHECK_COUNTS = collections.Counter()
OBSERVATIONS = collections.Counter()
HASHES = {}


def check(condition, kind, detail):
    CHECK_COUNTS[kind] += 1
    if not condition:
        ANOMALIES.append({'check': kind, 'detail': str(detail)})


def attempt(kind, label, function):
    try:
        function()
    except Exception as error:
        CHECK_COUNTS[kind] += 1
        ANOMALIES.append({'check': kind, 'detail': label, 'exception': type(error).__name__, 'message': str(error)})


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialized(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key '+key)
        result[key] = value
    return result


def read(path):
    raw = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = sha(raw)
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON '+value)))


def blob(commit, relative):
    return subprocess.run(['git', '-C', str(RECOVERY), 'show', commit+':'+relative],
                          check=True, capture_output=True).stdout


def walk(value, mapping):
    if isinstance(value, dict):
        return {mapping.get(key, key): walk(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [walk(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def pointer(value, path):
    for part in path.split('/')[1:]:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate '+key+' address')
    return result


def join_tables(source):
    tables = source['tables']
    check(set(tables) == {'stored_evidence', 'risk_forecasts', 'external_readouts',
                         'delivery_forecasts', 'post_command_forecasts'}, 'table_schema', 'namespace set')
    locations = {}
    commands = {}
    for name in ('stored_evidence', 'risk_forecasts', 'external_readouts'):
        nodes = indexed(tables[name], 'location')
        check(set(nodes) == set(LOCS), 'table_schema', name+' addresses')
        for address, row in nodes.items():
            target = locations.setdefault(address, {})
            for key, value in row.items():
                check(key not in target or target[key] == value, 'table_schema', name+' overlapping fields')
                target[key] = copy.deepcopy(value)
    for name in ('delivery_forecasts', 'post_command_forecasts'):
        nodes = indexed(tables[name], 'command')
        check(set(nodes) == set(CMDS), 'table_schema', name+' addresses')
        for address, row in nodes.items():
            target = commands.setdefault(address, {})
            for key, value in row.items():
                check(key not in target or target[key] == value, 'table_schema', name+' overlapping fields')
                target[key] = copy.deepcopy(value)
    result = {key: copy.deepcopy(value) for key, value in source.items() if key != 'tables'}
    result['locations'] = [locations[address] for address in LOCS]
    result['command_forecasts'] = [commands[address] for address in CMDS]
    return result


def remove_groups(natural, arm):
    expected = copy.deepcopy(natural)
    if arm == 'sensory_only':
        del expected['native_route_history']; del expected['command_forecasts']
        for node in expected['locations']:
            del node['estimated_current_decision_success']; del node['estimated_fresh_glimpse_decision_success']
        for key in ('history', 'forecasts', 'success', 'after_composition'):
            del expected['source_semantics'][key]
    elif arm == 'route_only':
        del expected['locations']; del expected['external_readout_evaluation_summary']
        del expected['frozen_sources']['external_readout']
        for command in expected['command_forecasts']:
            del command['estimated_decision_success_after_command']
        for key in ('stored_content', 'logits', 'success', 'after_composition', 'external_readout'):
            del expected['source_semantics'][key]
    else:
        for node in expected['locations']:
            del node['external_image_statistic_readouts']
        del expected['source_semantics']['external_readout']
        del expected['frozen_sources']['external_readout']; del expected['external_readout_evaluation_summary']
    return expected


def expected_units(source, query):
    units = {}
    def unit(name, group, paths):
        units[name] = (group, set(paths))
    def row_path(collection, address, value, prefix):
        rows = source['tables'][collection] if 'tables' in source else source[collection]
        index = next(i for i, row in enumerate(rows) if row[address] == value)
        return prefix+'/'+str(index)
    location = None; risk = None; external = None; delivery = None; after = None
    if 'tables' in source:
        location = row_path('stored_evidence', 'location', query['location'], '/tables/stored_evidence')
        risk = row_path('risk_forecasts', 'location', query['location'], '/tables/risk_forecasts')
        external = row_path('external_readouts', 'location', query['location'], '/tables/external_readouts')
        delivery = row_path('delivery_forecasts', 'command', query['command'], '/tables/delivery_forecasts')
        after = row_path('post_command_forecasts', 'command', query['command'], '/tables/post_command_forecasts')
    else:
        if 'locations' in source:
            path = row_path('locations', 'location', query['location'], '/locations')
            node = pointer(source, path)
            if 'stored_evidence' in node: location = path
            if 'estimated_current_decision_success' in node: risk = path
            if 'external_image_statistic_readouts' in node: external = path
        if 'command_forecasts' in source:
            delivery = row_path('command_forecasts', 'command', query['command'], '/command_forecasts')
            if 'estimated_decision_success_after_command' in pointer(source, delivery): after = delivery
    if location:
        evidence = location+'/stored_evidence'
        unit('sample_status', 'stored_evidence', [location+'/observed', evidence+'/status'])
        for key in ('evidence_item', 'raw_orientation_logit', 'logit_sign'):
            unit(key, 'stored_evidence', [evidence+'/'+key])
        unit('decision_or_default', 'stored_evidence', [evidence+'/stored_decision', evidence+'/decision_if_no_sample'])
        unit('stored_decision_meaning', 'stored_evidence', ['/source_semantics/stored_content', '/source_semantics/logits'])
    if 'native_route_history' in source:
        path = row_path('native_route_history', 'command', query['command'], '/native_route_history')
        unit('latest_observed_delivery', 'native_history', [path])
    if delivery:
        for address in pointer(source, delivery+'/predicted_delivery_distribution'):
            unit('delivery_'+address, 'delivery_forecasts', [delivery+'/predicted_delivery_distribution/'+address])
        unit('forecast_meaning', 'delivery_forecasts', ['/source_semantics/forecasts'])
    if after:
        for address in pointer(source, after+'/estimated_decision_success_after_command'):
            unit('post_command_'+address, 'outcome_forecasts', [after+'/estimated_decision_success_after_command/'+address])
    if risk:
        for key in ('estimated_current_decision_success', 'estimated_fresh_glimpse_decision_success'):
            unit(key, 'outcome_forecasts', [risk+'/'+key])
    if external:
        path = external+'/external_image_statistic_readouts'
        for key in ('image_value_std_estimate', 'nearest_neighbor_rms_estimate'):
            unit(key, 'external_readouts', [path+'/'+key])
        unit('external_estimate_meaning', 'external_readouts', [path+'/source', path+'/units', path+'/interpretation',
                                                             '/source_semantics/external_readout'])
    return units


def review_inventory(packet, audit):
    facts = audit['required_facts']; source = packet['source_record']; query = packet['indexed_query']
    expected = expected_units(source, query)
    units = facts['required_units']; unit_map = {unit['unit_id']: unit for unit in units}
    check(len(units) == len(unit_map) == len(expected) == facts['required_unit_count'], 'inventory', packet['packet_id']+' count')
    check(set(unit_map) == set(expected), 'inventory', packet['packet_id']+' required unit identities')
    for name, (group, paths) in expected.items():
        unit = unit_map[name]; saved = unit['source_facts']
        check(unit['source_group'] == group, 'inventory', name+' group')
        check(len(saved) == len(paths) and {fact['source_path'] for fact in saved} == paths, 'inventory', name+' compound constituents')
        for fact in saved:
            check(set(fact) == {'source_path', 'value'} and pointer(source, fact['source_path']) == fact['value'],
                  'inventory', packet['packet_id']+' '+fact['source_path'])
    check(facts['facts'] == [fact for unit in units for fact in unit['source_facts']], 'inventory', 'flattened inventory')
    groups = {group: {name for name, (registered, _) in expected.items() if registered == group} for group in GROUPS}
    check(set(facts['source_groups']) == set(GROUPS), 'inventory', 'thematic groups')
    for group in GROUPS:
        check(set(facts['source_groups'][group]) == groups[group], 'inventory', group+' membership')
    absent = {group for group, members in groups.items() if not members}
    check(facts['query'] == query, 'inventory', 'indexed query')
    check(facts['source_absence_unit_count'] == len(absent) == len(facts['source_absence_units']), 'inventory', 'absence count')
    check({unit['source_group'] for unit in facts['source_absence_units']} == absent and
          all(unit == {'source_group': unit['source_group'], 'status': 'not_supplied'} for unit in facts['source_absence_units']),
          'inventory', 'typed absence distinctions')
    expected_count, expected_absent = MASKS.get(audit['arm'], (21, set()))
    check(len(expected) == expected_count and absent == expected_absent, 'ablation_mask', audit['arm'])
    OBSERVATIONS['supplied_required_units'] += len(expected)
    OBSERVATIONS['required_source_absence_units'] += len(absent)


def review_source(source, label):
    if 'locations' in source:
        nodes = indexed(source['locations'], 'location')
        check(set(nodes) == set(LOCS), 'locations', label)
        for loc, node in nodes.items():
            observed = node['observed']; evidence = node['stored_evidence']
            check(type(observed) is bool, 'content_semantics', label+' observed')
            if observed:
                raw = evidence['raw_orientation_logit']
                check(type(raw) in (float, int) and math.isfinite(raw), 'content_semantics', label+' logit')
                check(evidence['status'] == 'sample_stored' and evidence['evidence_item'] in {'e0','e1','e2','e3'}
                      and evidence['decision_if_no_sample'] is None, 'content_semantics', label+' stored status')
                check(evidence['logit_sign'] == ('positive' if raw > 0 else 'negative' if raw < 0 else 'zero') and
                      evidence['stored_decision'] == ('vertical' if raw > 0 else 'horizontal'), 'content_semantics', label+' decision')
            else:
                check(evidence['status'] == 'no_sample_stored' and all(evidence[key] is None for key in
                      ('evidence_item','raw_orientation_logit','logit_sign','stored_decision')) and
                      evidence['decision_if_no_sample'] == {'value':'horizontal','basis':'fixed default guess'},
                      'content_semantics', label+' unseen null/default')
                OBSERVATIONS['unseen_location_rows'] += 1
            if 'external_image_statistic_readouts' in node:
                external = node['external_image_statistic_readouts']
                check(external['status'] == ('estimated_for_stored_sample' if observed else 'undefined_without_stored_sample'),
                      'external_semantics', label+' status')
                for key in ('image_value_std_estimate','nearest_neighbor_rms_estimate'):
                    value = external[key]
                    check((type(value) in (float,int) and math.isfinite(value)) if observed else value is None,
                          'external_semantics', label+' '+key)
                    if value is not None and value < 0: OBSERVATIONS['negative_external_estimates'] += 1
    if 'native_route_history' in source:
        history = indexed(source['native_route_history'], 'command')
        check(set(history) == set(CMDS), 'history', label+' addresses')
        for row in history.values():
            delivery = row['last_observed_delivery_one_hot']
            check(set(delivery) == set(LOCS) and all(type(value) is int and value in (0,1) for value in delivery.values())
                  and sum(delivery.values()) in (0,1), 'history', label+' onehot/unknown')
            check(row['status'] == ('observed_delivery' if sum(delivery.values()) else 'no_delivery_observed'), 'history', label+' status')
            if not sum(delivery.values()): OBSERVATIONS['unknown_history_rows'] += 1
    if 'command_forecasts' in source:
        commands = indexed(source['command_forecasts'], 'command')
        check(set(commands) == set(CMDS), 'forecasts', label+' addresses')
        for command, row in commands.items():
            delivery = row['predicted_delivery_distribution']
            check(set(delivery) == set(LOCS) and all(type(value) in (float,int) and 0 <= value <= 1 for value in delivery.values())
                  and abs(sum(delivery.values())-1) <= 1e-12, 'forecasts', label+' probabilities')
            if 'estimated_decision_success_after_command' in row:
                after = row['estimated_decision_success_after_command']
                check(set(after) == set(LOCS), 'composition', label+' post addresses')
                for loc in LOCS:
                    current = nodes[loc]['estimated_current_decision_success']
                    fresh = nodes[loc]['estimated_fresh_glimpse_decision_success']
                    expected = delivery[loc]*fresh+(1-delivery[loc])*current
                    check(after[loc] == expected and 0 <= current <= 1 and 0 <= fresh <= 1, 'composition', label+' '+command+'/'+loc)


def main():
    registration = read(BASE/'registration.json')
    audit = read(TARGETS/'audit.json')
    completion = read(TARGETS/'completion.json')
    verification = read(TARGETS/'verification.json')
    started = read(TARGETS/'attempt.json')
    execution = read(BASE/'execution_receipt.json')
    regpath = str((BASE/'registration.json').relative_to(ROOT))
    check((BASE/'registration.json').read_bytes() == blob(FREEZE, regpath), 'frozen_registration', FREEZE)
    check(registration['arms'] == list(ARMS) and registration['max_target_packets'] == 624 and
          registration['opportunity'] == {'visual_seeds':list(SEEDS), 'world_indices':list(range(16)),
          'locations':list(LOCS), 'snapshot_after_commands':8}, 'registration_rules', 'fixed full opportunity')
    check(registration['query_rule'] == {'location':'world_index % 4','command':'(world_index // 4) % 4'},
          'registration_rules', 'indexed query')
    check(registration['assignment_seed'] == 20261010 and registration['identifier_seed'] == 20261011 and
          registration['export_attempts'] == 1 and registration['reporter_calls_during_export'] == 0 and
          registration['factual_qualification_required_before_reporter'] is True,
          'registration_rules', 'fixed seeds/attempt/reporting boundary')
    check(TARGETS.resolve() == (ROOT/registration['output_root']).resolve(), 'output_binding', 'registered root')
    for label, value in [('completion', completion),('verification', verification),('attempt',started),('execution',execution)]:
        check(value['freeze_commit'] == FREEZE, 'receipts', label+' freeze commit')
    check(completion['status'] == 'single_registered_export_completed' and completion['packet_count'] == 624 and
          completion['factual_qualification_required'] is True and not completion['reporter_run'], 'receipts', 'completion')
    check(verification['status'] == 'exact_packets_sources_inverse_controls_verified' and verification['packet_count'] == 624
          and not verification['reporter_run'], 'receipts', 'saved source replay receipt')
    check(started['attempt'] == 1 and started['registration_path'] == regpath and not started['reporter_run'], 'receipts', 'sole attempt')
    check(execution['exit_code'] == 0 and execution['reporter_calls'] == 0 and execution['new_model_inference'] is False
          and execution['new_fits'] is False and execution['retry'] is False, 'receipts', 'execution declaration')
    for value in (completion,verification):
        check(value['registration_sha256'] == HASHES[regpath], 'hashes', 'registration receipt')
    check(completion['audit_sha256'] == HASHES[str((TARGETS/'audit.json').relative_to(ROOT))], 'hashes', 'audit')
    check(completion['verification_sha256'] == HASHES[str((TARGETS/'verification.json').relative_to(ROOT))], 'hashes', 'verification')
    check(not (TARGETS/'failure.json').exists(), 'receipts', 'no failed attempt receipt')
    paths = list((TARGETS/'reporter_packets').iterdir())
    check(len(paths) == 624 and {path.name for path in paths} == {name+'.json' for name in completion['packet_sha256']},
          'budget', 'exact packet file inventory')
    packets = {}
    for path in paths:
        packet = read(path); packet_id = packet['packet_id']
        check(packet_id not in packets and path.name == packet_id+'.json' and re.fullmatch(r'P[0-9a-f]{32}',packet_id),
              'identifiers', packet_id)
        packets[packet_id] = packet
        check(HASHES[str(path.relative_to(ROOT))] == completion['packet_sha256'][packet_id], 'hashes', packet_id+' file')
        check(set(packet) == {'packet_id','instruction','indexed_query','source_record'}, 'packet_schema', packet_id)
    check(len(audit) == len(packets) == 624 and len({row['packet_id'] for row in audit}) == 624, 'budget', 'unique audit associations')
    ordered_packets = [packets[row['packet_id']] for row in audit]
    check(sha(serialized({'packets':ordered_packets,'audit':audit})) == verification['prepared_sha256'], 'hashes', 'combined prepared serialization')
    rows = {(row['visual_seed'], row['world_index'], row['arm']): row for row in audit}
    check(set(rows) == {(seed,world,arm) for seed in SEEDS for world in range(16) for arm in ARMS}, 'budget', '48 opportunities × 13 arms')
    schedule = [(seed,world,arm) for seed in SEEDS for world in range(16) for arm in ARMS]
    random.Random(20261010).shuffle(schedule)
    check([(row['visual_seed'],row['world_index'],row['arm']) for row in audit] == schedule, 'randomization', 'registered shuffled schedule')
    normalized = {}
    banned = {'chronological_events','command_mapping','world_truth','physical_targets','pixels','future_images','labels',
              'correctness','identifier_map','world_index','source_variant','audit_provenance','required_facts','goal_achieved'}
    def packet_keys(value):
        if isinstance(value,dict):
            for key,item in value.items():
                yield key; yield from packet_keys(item)
        elif isinstance(value,list):
            for item in value: yield from packet_keys(item)
    for position, row in enumerate(audit):
        packet = packets[row['packet_id']]; label = row['packet_id']; arm = row['arm']; seed = row['visual_seed']; world = row['world_index']
        check(label == 'P'+sha(f'grounded_neutral_report_pilot_v1:20261010:{position}'.encode())[:32], 'randomization', label+' opaque schedule ID')
        check(row['packet_sha256'] == sha(serialized(packet)), 'hashes', label+' audit association')
        check(row['canonical_query'] == {'location':f'l{world%4}', 'command':f'k{(world//4)%4}'}, 'query', label)
        mapping = row['identifier_map']; inverse = {value:key for key,value in mapping.items()}
        canonical_ids = set(LOCS+CMDS)|{'e0','e1','e2','e3'}|{name for model in SEEDS for name in
                         (f'visual-{model}',f'structured-{model}',f'risk-hidden-ridge-{model}')}
        check(set(mapping) == canonical_ids and len(inverse) == 21 and all(re.fullmatch(r'x[0-9]{4}',value) for value in mapping.values()),
              'aliasing', label+' bijection coverage')
        check(walk(packet['indexed_query'],inverse) == row['canonical_query'], 'aliasing', label+' query inverse')
        source = walk(packet['source_record'],inverse)
        check(source == row['source_view_before_aliases'], 'aliasing', label+' whole source inverse')
        check(not (set(packet_keys(packet)) & banned), 'packet_boundary', label+' excluded namespaces')
        text = json.dumps(packet['source_record'])
        check(all('"'+token+'"' not in text for token in canonical_ids), 'packet_boundary', label+' all ID references aliased')
        normalized[seed,world,arm] = join_tables(source) if arm == 'tables' else source
        attempt('source_semantics',label,lambda source=normalized[seed,world,arm],label=label:review_source(source,label))
        attempt('inventory',label,lambda packet=packet,row=row:review_inventory(packet,row))
        sa = row['source_audit']; provenance = sa['frozen_provenance']
        check(sa['world_index'] == world and sa['source_variant'] == row['source_variant'] and
              sa['risk_hidden_tensor_pointer'] == {'variant':row['source_variant'],'row':world,'snapshot_axis':0},
              'audit_association', label+' canonical source pointers')
        check(sa['history_retains_actual_past'] and sa['current_memory_precedence'] and not sa['reporter_run'] and
              not sa['appearance_qualified'] and not sa['goal_achieved'] and not row['derived_joint_network_inference'],
              'audit_association', label+' interpretation flags')
        check(provenance['manifest_sha256'] == registration['source_sha256'][registration['physical_root']+'/manifest.json'],
              'source_hash_association', label+' physical manifest')
        expected_archive_names = {f"{registration['physical_root']}/model-{seed}/{name}.pt.gz" for name in
                                 ('split-2/snapshot','split-2/decoded','split-2/forecasts','evaluation_predictions','interventions','decoders')}
        check(set(provenance['archive_sha256']) == expected_archive_names and all(registration['source_sha256'].get(name) == value
              for name,value in provenance['archive_sha256'].items()), 'source_hash_association', label+' archive inventory')
        model_path = f'audits/grounded_attention_outcomes_v1/model-{seed}/instrument-structured.pt.gz'
        check(provenance['grounded_model_archive'] == model_path and provenance['grounded_model_sha256'] ==
              registration['source_sha256'][model_path], 'source_hash_association', label+' model association')
        check(row['unmatched_ablation'] == (arm in MASKS) and row['matched_information_control'] ==
              (arm in ('id_remap','apparatus','tables')), 'arm_classification', label)
    max_readout_difference = 0.0
    max_current_fresh_difference = 0.0
    natural_seen = natural_unseen = natural_unknown = 0
    for seed in SEEDS:
        for world in range(16):
            natural = normalized[seed,world,'natural']; memory = normalized[seed,world,'memory']
            natural_packet = packets[rows[seed,world,'natural']['packet_id']]
            for arm in ('sham','restore_memory','restore_route','id_remap','apparatus','tables'):
                check(normalized[seed,world,arm] == natural,'paired_controls',f'{seed}/{world}/{arm} source equality')
            for arm in ('sham','restore_memory','restore_route'):
                packet = packets[rows[seed,world,arm]['packet_id']]
                check({key:value for key,value in packet.items() if key!='packet_id'} ==
                      {key:value for key,value in natural_packet.items() if key!='packet_id'},'paired_controls',arm+' identical substantive input')
            for arm in ARMS:
                packet = packets[rows[seed,world,arm]['packet_id']]
                if arm == 'apparatus':
                    check(packet['instruction'].replace('ordinary apparatus','system') == natural_packet['instruction'],
                          'paired_controls','apparatus exact referent replacement')
                else:
                    check(packet['instruction'] == natural_packet['instruction'],'paired_controls',arm+' same common prompt')
                if arm != 'id_remap':
                    check(rows[seed,world,arm]['identifier_map'] == rows[seed,world,'natural']['identifier_map'],
                          'paired_controls',arm+' common aliases')
            check(rows[seed,world,'id_remap']['identifier_map'] != rows[seed,world,'natural']['identifier_map'],
                  'paired_controls','second remap bijection')
            for arm in MASKS:
                check(normalized[seed,world,arm] == remove_groups(natural,arm),'ablation_mask',f'{seed}/{world}/{arm} exact deletion')
            nloc = indexed(natural['locations'],'location'); mloc = indexed(memory['locations'],'location')
            for loc in LOCS:
                if nloc[loc]['observed']: natural_seen += 1
                else: natural_unseen += 1
            natural_unknown += sum(row['status']=='no_delivery_observed' for row in natural['native_route_history'])
            for arm,base in (('route',natural),('memory_route',memory)):
                source = normalized[seed,world,arm]
                check(source['locations'] == base['locations'] and source['native_route_history'] == natural['native_route_history'],
                      'factorial',f'{seed}/{world}/{arm} memory/history invariants')
                commands = indexed(source['command_forecasts'],'command'); originals = indexed(base['command_forecasts'],'command')
                for index,command in enumerate(CMDS):
                    check(commands[command]['predicted_delivery_distribution'] ==
                          originals[CMDS[(index-1)%4]]['predicted_delivery_distribution'],'factorial',arm+' delivery +1 row roll')
            check(memory['native_route_history'] == natural['native_route_history'],'memory_donor','fixed native history')
            check(indexed(memory['command_forecasts'],'command').keys() == set(CMDS),'memory_donor','four commands')
            for command in CMDS:
                check(indexed(memory['command_forecasts'],'command')[command]['predicted_delivery_distribution'] ==
                      indexed(natural['command_forecasts'],'command')[command]['predicted_delivery_distribution'],
                      'memory_donor','unchanged delivery forecast')
            for loc_index,loc in enumerate(LOCS):
                donor = LOCS[(loc_index-1)%4]; current = mloc[loc]; original = nloc[donor]
                check(current['observed'] == original['observed'] and current['stored_evidence'] == original['stored_evidence'],
                      'memory_donor',f'{seed}/{world}/{loc} evidence donor {donor}')
                for key in ('estimated_current_decision_success','estimated_fresh_glimpse_decision_success'):
                    delta = abs(current[key]-original[key]); max_current_fresh_difference=max(max_current_fresh_difference,delta)
                    check(delta <= 1e-6+1e-5*abs(original[key]),'memory_donor',key+' registered scalar source tolerance')
                for key in ('image_value_std_estimate','nearest_neighbor_rms_estimate'):
                    a=current['external_image_statistic_readouts'][key]; b=original['external_image_statistic_readouts'][key]
                    check((a is None)==(b is None),'memory_donor','readout donor definedness')
                    if a is not None: max_readout_difference=max(max_readout_difference,abs(a-b))
            history = indexed(natural['native_route_history'],'command')
            for arm in ARMS:
                row=rows[seed,world,arm]; sa=row['source_audit']; events=sa['external_observed_history_instrumentation']['chronological_events']
                check(events == rows[seed,world,'natural']['source_audit']['external_observed_history_instrumentation']['chronological_events']
                      and len(events)==8 and [event['event'] for event in events]==list(range(1,9)),
                      'audit_history',f'{seed}/{world}/{arm} shared eight-event audit chronology')
                expected_donors=[(index-1)%4 for index in range(4)] if arm in ('memory','memory_route') else list(range(4))
                check(sa['evidence_original_location_by_current_slot']==expected_donors,'audit_history',arm+' donor pointers')
                acquisition=[]
                for donor in expected_donors:
                    matches=[event['event'] for event in events if event['delivered_location']==LOCS[donor]]
                    acquisition.append(matches[-1] if matches else None)
                check(sa['evidence_acquisition_event_by_current_slot']==acquisition,'audit_history',arm+' acquisition pointers')
            for command in CMDS:
                events=rows[seed,world,'natural']['source_audit']['external_observed_history_instrumentation']['chronological_events']
                deliveries=[event['delivered_location'] for event in events if event['command']==command]
                expected={loc:int(bool(deliveries) and deliveries[-1]==loc) for loc in LOCS}
                check(history[command]['last_observed_delivery_one_hot']==expected,'audit_history','latest-only native history association')
    for relative in HASHES:
        if relative.startswith('audits/grounded_neutral_report_pilot_v1/'):
            check((ROOT/relative).read_bytes() == blob(SAVED_EXPORT,relative), 'saved_export_blobs', relative)
    summary={'packet_count':len(packets),'model_world_opportunities':len({(row['visual_seed'],row['world_index']) for row in audit}),
             'arms':dict(collections.Counter(row['arm'] for row in audit)),
             'per_model':dict(collections.Counter(str(row['visual_seed']) for row in audit)),
             'natural_location_rows':natural_seen+natural_unseen,'natural_observed_location_rows':natural_seen,
             'natural_unseen_location_rows':natural_unseen,'natural_unknown_history_rows':natural_unknown,
             'max_memory_donor_current_fresh_absolute_difference':max_current_fresh_difference,
             'max_memory_donor_external_readout_absolute_difference_observed_only':max_readout_difference,
             **dict(OBSERVATIONS)}
    result={'schema':'independent_saved_json_export_review_v1','freeze_commit':FREEZE,'saved_export_commit':SAVED_EXPORT,
            'status':'passed_bounded_json_checks' if not ANOMALIES else 'first_pass_anomalies',
            'checks':dict(CHECK_COUNTS),'total_checks':sum(CHECK_COUNTS.values()),'anomaly_count':len(ANOMALIES),
            'anomalies':ANOMALIES,'counts':summary,'input_sha256':HASHES,
            'limitations':['JSON/source association arithmetic only; no independent binary or neural reconstruction',
                           'binary source hashes and decoder propagation envelope not recomputed',
                           'no target reports or factual/character qualification assessed',
                           'no appearance counterpart or source-of-qualia claim qualified']}
    (OUT/'checks.json').write_bytes(serialized(result))
    print(json.dumps({key:result[key] for key in ('status','total_checks','anomaly_count','anomalies','counts')},indent=2))
    return 0 if not ANOMALIES else 1


if __name__ == '__main__':
    sys.exit(main())
