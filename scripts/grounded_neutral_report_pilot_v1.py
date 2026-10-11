"""Reviewed, single-attempt preparation of the auxiliary grounded report pilot.

Importing this module reads no archive. Registration is a pure description of a
fixed opportunity budget; the export entry point requires that exact description
in a Git commit. This module neither runs a Reporter nor calls a learned model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess

import grounded_report_interface_v1 as interface


SCHEMA = 'grounded_neutral_report_pilot_v1'
ARMS = ('natural', 'memory', 'route', 'memory_route', 'sham',
        'restore_memory', 'restore_route', 'id_remap', 'apparatus', 'tables',
        'sensory_only', 'route_only', 'no_external')
VARIANT_FOR_ARM = {'natural': 'natural', 'memory': 'memory_readdressed',
                   'route': 'delivery_patched', 'memory_route': 'memory_readdressed',
                   'sham': 'sham', 'restore_memory': 'memory_restored',
                   'restore_route': 'delivery_restored'}
UNMATCHED_ARMS = ('sensory_only', 'route_only', 'no_external')
ASSIGNMENT_SEED = 20261010
IDENTIFIER_SEED = 20261011
QUERY_RULE = {'location': 'world_index % 4', 'command': '(world_index // 4) % 4'}
MAX_TARGETS = len(interface.SEEDS)*16*len(ARMS)
DEFAULT_OUTPUT_ROOT = 'audits/grounded_neutral_report_pilot_v1/targets'
INSTRUCTION_TEMPLATE = (
    'Write a factual account of the supplied record in at most 180 words; compact '
    'numeric tables are welcome. [REFERENT] Answer the indexed location and command '
    'identified in the request, using only supplied fields. For the indexed location, '
    'give stored-sample status, evidence identifier, raw orientation decision coordinate '
    'and sign, stored orientation decision or explicitly labelled default guess, '
    'current and fresh-glimpse decision-success estimates, and the two external image '
    'statistic estimates in their stated units, wherever supplied. For the indexed '
    'command, state its latest observed delivery or unknown status and give the complete '
    'delivery-probability and post-command decision-success vectors with addresses. '
    'Distinguish stored decisions/defaults, represented forecasts and external objective '
    'estimates. If a requested source group is absent, say it is not supplied; keep '
    'this distinct from a supplied null or unknown. Round numbers to six decimal places, '
    'using scientific notation if needed to preserve a small nonzero sign. State only '
    'uncertainty and information limits supported by the record. Do not invent '
    'unobserved events or results.'
)
REFERENT = 'Describe the system recorded in the payload as an external diagnostic observer.'
APPARATUS_REFERENT = 'Describe the ordinary apparatus recorded in the payload as an external diagnostic observer.'
INSTRUCTION = INSTRUCTION_TEMPLATE.replace('[REFERENT]', REFERENT)



def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    return sha256(Path(path).read_bytes())


def expected_source_paths(physical_root=interface.PHYSICAL_ROOT):
    """Every file read by the frozen loader, including receipt inputs."""
    root = Path(physical_root)
    names = [root/name for name in ('manifest.json', 'completion.json',
                                   'verification.json', 'results.json')]
    for seed in interface.SEEDS:
        names.extend(root/f'model-{seed}'/(name+'.pt.gz') for name in
                     ('split-2/snapshot', 'split-2/decoded', 'split-2/forecasts',
                      'evaluation_predictions', 'interventions', 'decoders'))
        names.append(Path(f'audits/grounded_attention_outcomes_v1/model-{seed}/instrument-structured.pt.gz'))
    return tuple(str(path) for path in names)


def _hash_map(value, name):
    require(isinstance(value, dict) and bool(value), name+' hashes required')
    for path, hash_value in value.items():
        require(isinstance(path, str) and path and not Path(path).is_absolute()
                and '..' not in Path(path).parts, 'registered paths must be relative')
        require(isinstance(hash_value, str) and len(hash_value) == 64
                and all(c in '0123456789abcdef' for c in hash_value), 'invalid '+name+' hash')


def make_registration(*, source_sha256, code_sha256, protocol_sha256,
                      review_receipts, physical_root=str(interface.PHYSICAL_ROOT),
                      output_root=DEFAULT_OUTPUT_ROOT):
    """Pure registration builder; caller supplies inventory/review evidence."""
    value = {'schema': SCHEMA, 'status': 'reviewed_registration',
             'opportunity': {'visual_seeds': list(interface.SEEDS),
                             'world_indices': list(range(16)),
                             'locations': list(interface.LOCATION_NAMES),
                             'snapshot_after_commands': 8},
             'arms': list(ARMS), 'max_target_packets': MAX_TARGETS,
             'assignment_seed': ASSIGNMENT_SEED, 'identifier_seed': IDENTIFIER_SEED,
             'query_rule': copy.deepcopy(QUERY_RULE),
             'physical_root': physical_root, 'output_root': output_root,
             'source_sha256': copy.deepcopy(source_sha256),
             'code_sha256': copy.deepcopy(code_sha256),
             'protocol_sha256': copy.deepcopy(protocol_sha256),
             'review_receipts': copy.deepcopy(review_receipts),
             'reporter_calls_during_export': 0, 'export_attempts': 1,
             'matched_information_controls': ['id_remap', 'apparatus', 'tables'],
             'unmatched_ablations': list(UNMATCHED_ARMS),
             'factual_qualification_required_before_reporter': True}
    validate_registration(value)
    return value


def validate_registration(value):
    expected = {'schema', 'status', 'opportunity', 'arms', 'max_target_packets',
                'assignment_seed', 'identifier_seed', 'query_rule', 'physical_root', 'output_root', 'source_sha256',
                'code_sha256', 'protocol_sha256', 'review_receipts',
                'reporter_calls_during_export', 'export_attempts',
                'matched_information_controls', 'unmatched_ablations',
                'factual_qualification_required_before_reporter'}
    require(isinstance(value, dict) and set(value) == expected, 'registration fields differ')
    require(value['schema'] == SCHEMA and value['status'] == 'reviewed_registration', 'unreviewed registration')
    require(value['opportunity'] == {'visual_seeds': list(interface.SEEDS),
                'world_indices': list(range(16)), 'locations': list(interface.LOCATION_NAMES),
                'snapshot_after_commands': 8}, 'opportunity rule changed')
    require(value['arms'] == list(ARMS) and value['max_target_packets'] == MAX_TARGETS,
            'arm budget changed')
    require(value['assignment_seed'] == ASSIGNMENT_SEED and value['identifier_seed'] == IDENTIFIER_SEED
            and value['query_rule'] == QUERY_RULE,
            'assignment/query seeds changed')
    for key in ('source_sha256', 'code_sha256', 'protocol_sha256'):
        _hash_map(value[key], key)
    require(set(expected_source_paths(value['physical_root'])) <= set(value['source_sha256']),
            'source inventory must cover all loader reads and frozen upstream inputs')
    for name in ('physical_root', 'output_root'):
        path = Path(value[name])
        require(not path.is_absolute() and '..' not in path.parts and str(path) not in ('', '.'),
                'registered roots must be relative nonempty paths')
    require({'scripts/grounded_report_interface_v1.py',
             'scripts/grounded_neutral_report_pilot_v1.py'} <= set(value['code_sha256']),
            'frozen interface/exporter hashes required')
    receipts = value['review_receipts']
    require(isinstance(receipts, dict) and set(receipts) == {
            'inventory', 'independent_correction_review', 'exporter_independent_review'},
            'inventory, correction review, and independent exporter design review receipts required')
    for receipt in receipts.values():
        require(isinstance(receipt, dict) and set(receipt) == {'id', 'path', 'sha256'}
                and isinstance(receipt['id'], str) and bool(receipt['id']), 'invalid review receipt')
        _hash_map({receipt['path']: receipt['sha256']}, 'review receipt')
    require(value['reporter_calls_during_export'] == 0 and value['export_attempts'] == 1
            and value['factual_qualification_required_before_reporter'] is True,
            'single-attempt/factual gate changed')
    require(value['matched_information_controls'] == ['id_remap', 'apparatus', 'tables']
            and value['unmatched_ablations'] == list(UNMATCHED_ARMS), 'arm classification changed')


def derive_memory_route(canonical_memory):
    """Readdressed native risk/readout plus represented delivery-row patch.

    This is an export composition of existing values, not joint network inference.
    The frozen schema validator runs before and after this pure intervention.
    """
    interface.validate_reporter_input(canonical_memory)
    value = copy.deepcopy(canonical_memory)
    original = canonical_memory['command_forecasts']
    for index, row in enumerate(value['command_forecasts']):
        row['predicted_delivery_distribution'] = copy.deepcopy(
            original[(index-1) % 4]['predicted_delivery_distribution'])
        for node in value['locations']:
            loc = node['location']
            d = row['predicted_delivery_distribution'][loc]
            row['estimated_decision_success_after_command'][loc] = (
                d*node['estimated_fresh_glimpse_decision_success']
                +(1-d)*node['estimated_current_decision_success'])
    interface.validate_reporter_input(value)
    return value


def identifier_map(seed):
    """Injective mapping covers addresses, evidence and the nine source IDs."""
    rng = random.Random(seed)
    canonical = list(interface.LOCATION_NAMES)+list(interface.COMMAND_NAMES)
    canonical += [f'e{i}' for i in range(4)]
    canonical += [name for model in interface.SEEDS for name in
                  (f'visual-{model}', f'structured-{model}', f'risk-hidden-ridge-{model}')]
    labels = [f'x{number:04d}' for number in rng.sample(range(1000, 10000), len(canonical))]
    return dict(zip(canonical, labels))


def relabel(value, mapping):
    """Generic exact-token/key relabel only after canonical validation."""
    require(len(set(mapping.values())) == len(mapping) and not (set(mapping) & set(mapping.values())),
            'identifier mapping must be injective with disjoint alphabets')
    def walk(node):
        if isinstance(node, dict):
            return {mapping.get(key, key): walk(item) for key, item in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, str):
            return mapping.get(node, node)
        return node
    return walk(value)


def undo_relabel(value, mapping):
    return relabel(value, {new: old for old, new in mapping.items()})


def remapped_record(canonical, mapping):
    interface.validate_reporter_input(canonical)
    result = relabel(canonical, mapping)
    inverse = undo_relabel(result, mapping)
    interface.validate_reporter_input(inverse)
    require(inverse == canonical, 'identifier inverse changed source information')
    return result


def remapped_tables(canonical, mapping):
    tables = interface.address_aware_tables(canonical)
    require(interface.join_address_aware_tables(tables) == canonical, 'table roundtrip changed source')
    result = relabel(tables, mapping)
    inverse = undo_relabel(result, mapping)
    require(interface.join_address_aware_tables(inverse) == canonical, 'remapped table inverse changed source')
    return result


def ablate(canonical, arm):
    """Explicit information-removal arms, never claimed information-matched."""
    interface.validate_reporter_input(canonical)
    require(arm in UNMATCHED_ARMS, 'unknown ablation')
    value = copy.deepcopy(canonical)
    if arm == 'sensory_only':
        value.pop('native_route_history')
        value.pop('command_forecasts')
        for node in value['locations']:
            node.pop('estimated_current_decision_success')
            node.pop('estimated_fresh_glimpse_decision_success')
        for key in ('history', 'forecasts', 'success', 'after_composition'):
            value['source_semantics'].pop(key)
    elif arm == 'route_only':
        value.pop('locations')
        for row in value['command_forecasts']:
            row.pop('estimated_decision_success_after_command')
        for key in ('stored_content', 'logits', 'success', 'after_composition', 'external_readout'):
            value['source_semantics'].pop(key)
        value['frozen_sources'].pop('external_readout')
        value.pop('external_readout_evaluation_summary')
    else:
        for node in value['locations']:
            node.pop('external_image_statistic_readouts')
        value['source_semantics'].pop('external_readout')
        value['frozen_sources'].pop('external_readout')
        value.pop('external_readout_evaluation_summary')
    return value


def _atomic_facts(value, prefix=''):
    """Machine-readable source paths/values; never provided to the Reporter."""
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = key.replace('~', '~0').replace('/', '~1')
            yield from _atomic_facts(item, prefix+'/'+escaped)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _atomic_facts(item, prefix+'/'+str(index))
    else:
        yield {'source_path': prefix, 'value': value}


def required_facts(source_record, query):
    """Bounded 21-unit inventory and fixed source-absence masks, audit-only.

    Compound units include all constituents. Metadata/error coordinates remain
    available for auditing volunteered claims but are not required coverage.
    """
    groups = {name: [] for name in ('stored_evidence', 'native_history',
                                   'delivery_forecasts', 'outcome_forecasts', 'external_readouts')}
    units = []

    def fact(path):
        node = source_record
        for part in path.split('/')[1:]:
            part = part.replace('~1', '/').replace('~0', '~')
            node = node[int(part)] if isinstance(node, list) else node[part]
        return {'source_path': path, 'value': copy.deepcopy(node)}

    def unit(name, group, paths):
        result = {'unit_id': name, 'source_group': group,
                  'source_facts': [fact(path) for path in paths]}
        groups[group].append(name)
        units.append(result)

    def row_prefix(collection, address, target):
        for index, row in enumerate(collection):
            if row[address] == target:
                return index
        raise ValueError('indexed query address missing')

    if 'tables' in source_record:
        tables = source_record['tables']
        location_prefix = {}
        command_prefix = {}
        for name in ('stored_evidence', 'risk_forecasts', 'external_readouts'):
            index = row_prefix(tables[name], 'location', query['location'])
            location_prefix[name] = '/tables/'+name+'/'+str(index)
        for name in ('delivery_forecasts', 'post_command_forecasts'):
            index = row_prefix(tables[name], 'command', query['command'])
            command_prefix[name] = '/tables/'+name+'/'+str(index)
    else:
        location_prefix = {}
        command_prefix = {}
        if 'locations' in source_record:
            index = row_prefix(source_record['locations'], 'location', query['location'])
            prefix = '/locations/'+str(index)
            node = source_record['locations'][index]
            if 'stored_evidence' in node:
                location_prefix['stored_evidence'] = prefix
            if 'estimated_current_decision_success' in node:
                location_prefix['risk_forecasts'] = prefix
            if 'external_image_statistic_readouts' in node:
                location_prefix['external_readouts'] = prefix
        if 'command_forecasts' in source_record:
            index = row_prefix(source_record['command_forecasts'], 'command', query['command'])
            prefix = '/command_forecasts/'+str(index)
            command_prefix['delivery_forecasts'] = prefix
            if 'estimated_decision_success_after_command' in source_record['command_forecasts'][index]:
                command_prefix['post_command_forecasts'] = prefix

    if 'stored_evidence' in location_prefix:
        prefix = location_prefix['stored_evidence']
        evidence = prefix+'/stored_evidence'
        unit('sample_status', 'stored_evidence', [prefix+'/observed', evidence+'/status'])
        for name in ('evidence_item', 'raw_orientation_logit', 'logit_sign'):
            unit(name, 'stored_evidence', [evidence+'/'+name])
        unit('decision_or_default', 'stored_evidence', [evidence+'/stored_decision', evidence+'/decision_if_no_sample'])
        unit('stored_decision_meaning', 'stored_evidence', ['/source_semantics/stored_content', '/source_semantics/logits'])
    if 'native_route_history' in source_record:
        index = row_prefix(source_record['native_route_history'], 'command', query['command'])
        unit('latest_observed_delivery', 'native_history', ['/native_route_history/'+str(index)])
    if 'delivery_forecasts' in command_prefix:
        prefix = command_prefix['delivery_forecasts']+'/predicted_delivery_distribution'
        for address in fact(prefix)['value']:
            unit('delivery_'+address, 'delivery_forecasts', [prefix+'/'+address])
        unit('forecast_meaning', 'delivery_forecasts', ['/source_semantics/forecasts'])
    if 'post_command_forecasts' in command_prefix:
        prefix = command_prefix['post_command_forecasts']+'/estimated_decision_success_after_command'
        for address in fact(prefix)['value']:
            unit('post_command_'+address, 'outcome_forecasts', [prefix+'/'+address])
    if 'risk_forecasts' in location_prefix:
        prefix = location_prefix['risk_forecasts']
        for name in ('estimated_current_decision_success', 'estimated_fresh_glimpse_decision_success'):
            unit(name, 'outcome_forecasts', [prefix+'/'+name])
    if 'external_readouts' in location_prefix:
        prefix = location_prefix['external_readouts']+'/external_image_statistic_readouts'
        for name in ('image_value_std_estimate', 'nearest_neighbor_rms_estimate'):
            unit(name, 'external_readouts', [prefix+'/'+name])
        unit('external_estimate_meaning', 'external_readouts',
             [prefix+'/source', prefix+'/units', prefix+'/interpretation', '/source_semantics/external_readout'])
    absence = [{'source_group': name, 'status': 'not_supplied'}
               for name, members in groups.items() if not members]
    return {'query': copy.deepcopy(query), 'required_units': units,
            'required_unit_count': len(units), 'source_absence_units': absence,
            'source_absence_unit_count': len(absence), 'source_groups': groups,
            'facts': [item for entry in units for item in entry['source_facts']]}


def build_prepared(bundles, registration):
    """Pure fake-data-testable preparation. Caller authorizes production loading."""
    validate_registration(registration)
    require(set(bundles) == set(interface.SEEDS), 'exactly three registered models required')
    entries = []
    for seed in interface.SEEDS:
        bundle = bundles[seed]
        require(bundle.visual_seed == seed, 'bundle/model association mismatch')
        interface.validate_bundle(bundle)
        for world in interface.selection_indices(bundle):
            query = {'location': interface.LOCATION_NAMES[world % 4],
                     'command': interface.COMMAND_NAMES[(world // 4) % 4]}
            records = {variant: interface.record(bundle, world, variant)
                       for variant in interface.VARIANTS}
            natural = records['natural']['reporter_input']
            for restored in ('sham', 'memory_restored', 'delivery_restored'):
                require(records[restored]['reporter_input'] == natural, 'sham/inverse is not exact natural input')
            for arm in ARMS:
                variant = VARIANT_FOR_ARM.get(arm, 'natural')
                canonical = records[variant]['reporter_input']
                interface.validate_reporter_input(canonical)
                mapping = identifier_map(IDENTIFIER_SEED+seed*100+world)
                # Every packet uses opaque aliases. The remap arm uses a second
                # registered bijection, with the same exact canonical inverse.
                if arm == 'id_remap':
                    mapping = identifier_map(ASSIGNMENT_SEED+seed*100+world+1000000)
                role = 'external diagnostic'
                if arm == 'memory_route':
                    source = derive_memory_route(canonical)
                elif arm == 'id_remap':
                    source = copy.deepcopy(canonical)
                elif arm == 'tables':
                    source = interface.address_aware_tables(canonical)
                    require(interface.join_address_aware_tables(source) == canonical, 'table input not exactly joinable')
                elif arm == 'apparatus':
                    source = copy.deepcopy(canonical)
                    role = 'ordinary apparatus'
                    require(source == natural, 'apparatus control changes source information')
                elif arm in UNMATCHED_ARMS:
                    source = ablate(canonical, arm)
                else:
                    source = copy.deepcopy(canonical)
                source_before_aliases = copy.deepcopy(source)
                source = relabel(source, mapping)
                require(undo_relabel(source, mapping) == source_before_aliases,
                        'packet identifier inverse changed source information')
                if arm == 'tables':
                    require(interface.join_address_aware_tables(undo_relabel(source, mapping)) == natural,
                            'opaque tables do not exactly join to natural input')
                elif arm not in UNMATCHED_ARMS:
                    interface.validate_reporter_input(undo_relabel(source, mapping))
                packet_query = relabel(query, mapping)
                instruction = INSTRUCTION_TEMPLATE.replace('[REFERENT]',
                    APPARATUS_REFERENT if arm == 'apparatus' else REFERENT)
                packet = {'instruction': instruction,
                          'indexed_query': packet_query, 'source_record': source}
                audit = {'visual_seed': seed, 'world_index': world, 'arm': arm,
                         'source_variant': variant, 'matched_information_control': arm in ('id_remap', 'apparatus', 'tables'),
                         'unmatched_ablation': arm in UNMATCHED_ARMS,
                         'source_audit': copy.deepcopy(records[variant]['audit_provenance']),
                         'canonical_source_record': copy.deepcopy(canonical),
                         'source_view_before_aliases': source_before_aliases,
                         'canonical_query': copy.deepcopy(query),
                         'identifier_map': mapping,
                         'derived_joint_network_inference': False,
                         'delivery_roll': 1 if arm == 'memory_route' else 0,
                         'required_facts': required_facts(source, packet_query)}
                entries.append((packet, audit))
    rng = random.Random(ASSIGNMENT_SEED)
    rng.shuffle(entries)
    packets, audit = [], []
    for index, (packet, association) in enumerate(entries):
        # IDs encode order only; model, arm, and world remain in the audit namespace.
        packet_id = 'P'+sha256(f'{SCHEMA}:{ASSIGNMENT_SEED}:{index}'.encode())[:32]
        packet = {'packet_id': packet_id, **packet}
        association = {'packet_id': packet_id, 'packet_sha256': sha256(encoded(packet)), **association}
        packets.append(packet)
        audit.append(association)
    require(len(packets) == MAX_TARGETS and len({p['packet_id'] for p in packets}) == MAX_TARGETS,
            'target opportunity budget/IDs differ')
    return {'packets': packets, 'audit': audit}


def verify_prepared(prepared, bundles, registration):
    """Exact packet, full-source, inverse-control and association replay."""
    rebuilt = build_prepared(bundles, registration)
    require(prepared == rebuilt, 'prepared packets/full-source associations differ from deterministic replay')
    return {'schema': SCHEMA, 'status': 'exact_packets_sources_inverse_controls_verified',
            'packet_count': MAX_TARGETS, 'prepared_sha256': sha256(encoded(prepared)),
            'reporter_run': False}


def _git_bytes(checkout, commit, relative):
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit), 'full freeze commit required')
    location = ['--git-dir', str(checkout)] if Path(checkout).name == '.git' else ['-C', str(checkout)]
    result = subprocess.run(['git', *location, 'show', commit+':'+relative],
                            check=True, capture_output=True)
    return result.stdout


def check_frozen_registration(registration_path, *, repo_root, freeze_checkout, freeze_commit):
    repo = Path(repo_root).resolve()
    path = Path(registration_path).resolve()
    relative = str(path.relative_to(repo))
    data = path.read_bytes()
    require(data == _git_bytes(freeze_checkout, freeze_commit, relative),
            'registration differs from reviewed freeze commit')
    value = json.loads(data)
    validate_registration(value)
    # Code/protocol/review receipts must also be in that same frozen tree.
    frozen = {**value['code_sha256'], **value['protocol_sha256']}
    for receipt in value['review_receipts'].values():
        require(receipt['path'] not in frozen or frozen[receipt['path']] == receipt['sha256'],
                'receipt path has inconsistent registered hashes')
        frozen[receipt['path']] = receipt['sha256']
    for name, expected in frozen.items():
        require(digest(repo/name) == expected, 'registered code/protocol/review changed: '+name)
        require(sha256(_git_bytes(freeze_checkout, freeze_commit, name)) == expected,
                'code/protocol/review not frozen: '+name)
    # Bind the code actually imported, not only a file at a claimed repository path.
    require(digest(__file__) == value['code_sha256']['scripts/grounded_neutral_report_pilot_v1.py']
            and digest(interface.__file__) == value['code_sha256']['scripts/grounded_report_interface_v1.py'],
            'runtime code differs from registered code')
    return value, sha256(data)


def _check_sources(registration, repo_root):
    for name, expected in registration['source_sha256'].items():
        require(digest(Path(repo_root)/name) == expected, 'registered source hash changed: '+name)


def _load_registered_bundles(registration, repo_root):
    require(Path.cwd().resolve() == Path(repo_root).resolve(), 'frozen loader requires repository cwd')
    _check_sources(registration, repo_root)
    manifest = json.loads((Path(repo_root)/registration['physical_root']/'manifest.json').read_bytes())
    for name, expected in manifest['frozen_input_sha256'].items():
        require(registration['source_sha256'].get(name) == expected,
                'upstream frozen input absent/different in registration: '+name)
    bundles = {seed: interface.load_bundle(seed, root=registration['physical_root'])
               for seed in interface.SEEDS}
    for bundle in bundles.values():
        provenance = bundle.provenance
        for name, expected in provenance['archive_sha256'].items():
            require(registration['source_sha256'].get(name) == expected,
                    'loader source differs from registered inventory')
        require(registration['source_sha256'].get(provenance['grounded_model_archive'])
                == provenance['grounded_model_sha256'], 'loader model association differs')
        require(registration['source_sha256'][str(Path(registration['physical_root'])/'manifest.json')]
                == provenance['manifest_sha256'], 'loader receipt association differs')
    _check_sources(registration, repo_root)
    return bundles


def _exclusive_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def export_once(registration_path, output_dir, *, repo_root, freeze_checkout, freeze_commit):
    """Consume the sole attempt even on failure; no overwrite, repair or retry."""
    registration = json.loads(Path(registration_path).read_bytes())
    validate_registration(registration)
    output = Path(output_dir)
    require(output.resolve() == (Path(repo_root)/registration['output_root']).resolve(),
            'output directory differs from sole registered attempt location')
    output.mkdir(parents=True, exist_ok=False)
    _exclusive_json(output/'attempt.json', {'schema': SCHEMA, 'attempt': 1,
                    'registration_path': str(registration_path), 'freeze_commit': freeze_commit,
                    'status': 'started', 'reporter_run': False})
    try:
        registration, registration_hash = check_frozen_registration(
            registration_path, repo_root=repo_root, freeze_checkout=freeze_checkout,
            freeze_commit=freeze_commit)
        bundles = _load_registered_bundles(registration, repo_root)
        prepared = build_prepared(bundles, registration)
        verification = verify_prepared(prepared, bundles, registration)
        # Keep packet-only files separate from every source-aware audit artifact.
        (output/'reporter_packets').mkdir()
        for packet in prepared['packets']:
            _exclusive_json(output/'reporter_packets'/(packet['packet_id']+'.json'), packet)
        _exclusive_json(output/'audit.json', prepared['audit'])
        _exclusive_json(output/'verification.json', {**verification, 'registration_sha256': registration_hash,
                                                    'freeze_commit': freeze_commit})
        _exclusive_json(output/'completion.json', {'schema': SCHEMA,
            'status': 'single_registered_export_completed', 'registration_sha256': registration_hash,
            'freeze_commit': freeze_commit, 'packet_count': MAX_TARGETS,
            'audit_sha256': digest(output/'audit.json'),
            'verification_sha256': digest(output/'verification.json'),
            'packet_sha256': {packet['packet_id']: digest(output/'reporter_packets'/(packet['packet_id']+'.json'))
                              for packet in prepared['packets']},
            'reporter_run': False, 'factual_qualification_required': True})
        return verification
    except Exception as error:
        _exclusive_json(output/'failure.json', {'schema': SCHEMA, 'status': 'single_attempt_failed',
                                              'error_type': type(error).__name__, 'message': str(error),
                                              'retry_allowed': False, 'reporter_run': False})
        raise


def verify_export(registration_path, output_dir, *, repo_root, freeze_checkout, freeze_commit):
    registration, registration_hash = check_frozen_registration(
        registration_path, repo_root=repo_root, freeze_checkout=freeze_checkout, freeze_commit=freeze_commit)
    output = Path(output_dir)
    require(output.resolve() == (Path(repo_root)/registration['output_root']).resolve(),
            'output directory differs from sole registered attempt location')
    require(not (output/'failure.json').exists(), 'failed attempt cannot be released')
    completion = json.loads((output/'completion.json').read_bytes())
    require(completion['status'] == 'single_registered_export_completed'
            and completion['registration_sha256'] == registration_hash
            and completion['freeze_commit'] == freeze_commit, 'completion identity differs')
    require(digest(output/'audit.json') == completion['audit_sha256']
            and digest(output/'verification.json') == completion['verification_sha256'], 'audit/verification hash differs')
    files = list((output/'reporter_packets').iterdir())
    require({path.name for path in files} == {packet_id+'.json' for packet_id in completion['packet_sha256']},
            'extra/missing packet files')
    packets = []
    for packet_id, expected in completion['packet_sha256'].items():
        path = output/'reporter_packets'/(packet_id+'.json')
        require(digest(path) == expected, 'packet hash differs')
        packets.append(json.loads(path.read_bytes()))
    # File ordering is immaterial; the registered assignment remains in the audit.
    audit = json.loads((output/'audit.json').read_bytes())
    by_id = {packet['packet_id']: packet for packet in packets}
    prepared = {'packets': [by_id[row['packet_id']] for row in audit], 'audit': audit}
    receipt = verify_prepared(prepared, _load_registered_bundles(registration, repo_root), registration)
    saved = json.loads((output/'verification.json').read_bytes())
    require(saved == {**receipt, 'registration_sha256': registration_hash, 'freeze_commit': freeze_commit},
            'saved verification differs from exact replay')
    return receipt


def reporter_release(registration_path, output_dir, qualification_path, *, repo_root,
                     freeze_checkout, freeze_commit):
    """Fail closed until the separate reporting registration/verifier is frozen.

    Export readiness is not reporting readiness. A hand-entered perfect-count
    dictionary cannot authorize a Reporter. The reviewed reporting stage must bind
    saved fresh first passes, frozen fixtures/key/scorer and independent review.
    """
    raise ValueError('Reporter release is unavailable: independently reviewed reporting '
                     'registration and source-linked factual qualification verifier are required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('export', 'verify'))
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--freeze-checkout', '--git-dir', dest='freeze_checkout', type=Path, required=True)
    parser.add_argument('--freeze-commit', '--registration-commit', dest='freeze_commit', required=True)
    args = parser.parse_args()
    action = export_once if args.operation == 'export' else verify_export
    result = action(args.registration, args.output, repo_root=args.repo_root,
                    freeze_checkout=args.freeze_checkout, freeze_commit=args.freeze_commit)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
