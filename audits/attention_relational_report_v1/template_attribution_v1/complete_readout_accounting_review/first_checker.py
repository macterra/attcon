"""Exclusive independent COMPLETE descriptive accounting check; standard library only.
Written before target artifacts were opened. No semantic endpoint is computed.
Run only on root's explicit complete-input readiness instruction.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = 'audits/attention_relational_report_v1/template_attribution_v1/'
PREP = BASE + 'semantic_review/preparation/'
MANIFEST_SHA = '0f444a55011dffb7b53e1cb18f45a8304b44282aba9b65216eda30a2e0b094fe'
BANK = BASE + 'property_pairs/first_pairs.json'
BANK_SHA = 'c944d239cd52863fefa9331fc91800d048362d38d2fd7ee2659dc3b41d5706a5'
FROZEN = {
    'scripts/attention_relational_template_review_collection_v1.py': '091a6ee6691f4cb26d7a674002fe203b5c22ab682708267e23266f523b7920fd',
    'scripts/attention_relational_template_binding_inventory_v1.py': 'ee4af6717408a27b0a1858ccbd25398bccedace92013320f308cb97794fb49b3',
    'docs/ATTENTION_RELATIONAL_TEMPLATE_SEMANTIC_REVIEW_V1.md': 'e55a74c579593711a3ef40e308b4ca0b2d7b220f1a491a52d7f2ab2e481bf7a7',
    BASE + 'semantic_review/first_review_output_schema.json': 'f7a70c0ef14b0f6279e9ae2db40f34c48442493c4b25b05e159fe915529e1a27',
}
STREAMS = ['reviewer_0', 'reviewer_1']
CV = 'attention_relational_template_review_collection_v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identical(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def at(value, steps):
    for step in steps:
        if type(value) is dict and type(step) is str:
            value = value[step]
        elif type(value) is list and type(step) is int and 0 <= step < len(value):
            value = value[step]
        else:
            raise ValueError('nonresolving typed path')
    return value


def require(condition, label):
    if not condition:
        raise ValueError(label)


def schema_ok(v, s):
    if 'anyOf' in s:
        return any(schema_ok(v, choice) for choice in s['anyOf'])
    if 'type' in s and type(v) is not {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'integer': int}[s['type']]:
        return False
    if 'enum' in s and not any(identical(v, x) for x in s['enum']):
        return False
    if 'const' in s and not identical(v, s['const']):
        return False
    if type(v) is dict:
        props = s.get('properties', {})
        if not set(s.get('required', [])).issubset(v):
            return False
        if s.get('additionalProperties') is False and not set(v).issubset(props):
            return False
        return all(schema_ok(x, props.get(k, {})) for k, x in v.items())
    if type(v) is list:
        return s.get('minItems', 0) <= len(v) <= s.get('maxItems', float('inf')) and all(schema_ok(x, s.get('items', {})) for x in v)
    return not (type(v) is int and v < s.get('minimum', v))


class Audit:
    def __init__(self):
        self.inputs = {}
        self.discrepancies = []
        self.checks = Counter()
        self.stats = Counter()

    def path(self, path):
        p = Path(path)
        return p if p.is_absolute() else ROOT / p

    def read(self, path, expected=None, parse=True):
        p = self.path(path)
        raw = p.read_bytes()
        actual = sha(raw)
        key = str(p.resolve())
        self.inputs[key] = dict(path=key, sha256=actual, bytes=len(raw), expected_sha256=expected)
        if expected is not None:
            self.eq('input hash ' + str(path), actual, expected)
            require(actual == expected, 'bound input hash mismatch: ' + str(path))
        return json.loads(raw) if parse else raw

    def eq(self, label, actual, expected):
        self.checks[label.split(':')[0]] += 1
        if not identical(actual, expected):
            ar, er = encoded(actual), encoded(expected)
            self.discrepancies.append(dict(check=label, actual_sha256=sha(ar), expected_sha256=sha(er),
                actual=actual if len(ar) < 1200 else '<large value; hash retained>',
                expected=expected if len(er) < 1200 else '<large value; hash retained>'))

    def descriptors(self, value):
        # Bind declared source files without interpreting their evidence or labels.
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                self.read(value['path'], value['sha256'], parse=False)
            for child in value.values():
                self.descriptors(child)
        elif isinstance(value, list):
            for child in value:
                self.descriptors(child)

    def source_map(self, value):
        if isinstance(value, dict):
            for path, expected in value.items():
                if isinstance(expected, str) and len(expected) == 64 and isinstance(path, str) and '/' in path:
                    self.read(path, expected, parse=False)


def inventory(items, field, originals):
    ks = [encoded(x[field]) for x in items]
    require(len(ks) == len(set(ks)) and set(ks) == {encoded(x['path']) for x in originals}, 'exact inventory differs: ' + field)
    return dict(zip(ks, items))


def record_contract(j, r):
    for name in ['id', 'unit_id', 'text_sha256', 'reference_established']:
        require(identical(j[name], r[name]), 'record identity ' + name)
    require(bool(j['notes'].strip()), 'empty record reason')
    if not r['reference_established']:
        require(not any(j[k] for k in ['component_reviews', 'evidence_reviews', 'binding_reviews']), 'unestablished reference semantic salvage')
        require('unestablished' in j['notes'].lower(), 'unestablished diagnostic absent')
        return []
    require(len(r['components']) == 12, 'twelve components required')
    comps = inventory(j['component_reviews'], 'component_path', r['components'])
    evs = inventory(j['evidence_reviews'], 'evidence_path', r['evidence'])
    bs = inventory(j['binding_reviews'], 'binding_path', r['bindings'])
    units = {x['unit_id']: x for x in r['property_table']}
    evidence_ids = {x['evidence_id']: x for x in r['evidence']}
    require(len(units) == len(r['property_table']) and len(evidence_ids) == len(r['evidence']), 'nonunique construction/evidence identities')
    for c in r['components']:
        review = comps[encoded(c['path'])]
        require(review['original_status'] == c['recorded_status'] and bool(review['notes'].strip()), 'component status/reason')
    derived_refs = []
    for e in r['evidence']:
        review = evs[encoded(e['path'])]
        require(bool(review['notes'].strip()), 'empty evidence reason')
        for number, ref in enumerate(review['properties']):
            unit = units[ref['unit_id']]
            value = at(unit['value'], ref['value_path'])
            require(identical(value, ref['value']) and bool(ref['reason'].strip()), 'exact property value/reason')
            packet = unit.get('packet_path')
            derived_refs.append(dict(evidence_path=e['path'], property_position=number, unit_id=ref['unit_id'],
                value_path=ref['value_path'], value=value, semantic_role=ref['semantic_role'], reason=ref['reason'],
                packet_value_path=None if packet is None else packet + ref['value_path'], unit_kind=unit['kind'],
                unit_start=unit['start'], unit_end=unit['end'],
                overlaps_quoted_evidence=any(x['unit_id'] == ref['unit_id'] for x in e['geometry']['intersections']),
                evidence_geometry=e['geometry'], context=unit.get('context'), source_provenance=unit.get('source_resolution'),
                semantic_relation_asserted_by_reviewer_only=True))
    for b in r['bindings']:
        review = bs[encoded(b['path'])]
        require(bool(review['notes'].strip()), 'empty binding reason')
        fs = [x['feature'] for x in review['feature_witnesses']]
        require(len(fs) == len(set(fs)) and set(fs) == set(b['required_features']), 'required features differ')
        for w in review['feature_witnesses']:
            paths = [encoded(x) for x in w['evidence_paths']]
            possible = {encoded(evidence_ids[x]['path']) for x in b['candidate_feature_evidence_ids'][w['feature']]}
            require(len(paths) == len(set(paths)) and set(paths).issubset(possible), 'invalid feature witness selection')
            require(review['assessment'] != 'supported' or bool(paths), 'supported binding missing required witness')
        own = [encoded(x) for x in review['binding_evidence_paths']]
        require(len(own) == len(set(own)) and set(own) == {encoded(evidence_ids[x]['path']) for x in b['own_evidence_ids']}, 'binding own evidence differs')
    return derived_refs


def inspect_collection(a, collection, manifest, chunks, schema):
    deps = collection['input_dependencies']
    a.descriptors(deps)
    launch = a.read(deps['launch_receipt']['path'], deps['launch_receipt']['sha256'])
    outcomes = a.read(deps['outcome_receipt']['path'], deps['outcome_receipt']['sha256'])
    a.eq('collection:launch hash', collection['launch_receipt_sha256'], deps['launch_receipt']['sha256'])
    for receipt in [launch, outcomes]:
        a.eq('receipt:manifest binding', receipt['manifest_sha256'], MANIFEST_SHA)
        a.eq('receipt:complete contexts', len(receipt['contexts']), 46)
    a.eq('receipt:launch schema', launch['schema'], CV + '_launch')
    a.eq('receipt:outcome schema', outcomes['schema'], CV + '_outcomes')
    a.eq('receipt:launch linkage', outcomes['launch_receipt_sha256'], deps['launch_receipt']['sha256'])
    a.eq('receipt:protocol linkage', launch['protocol_sha256'], manifest['review_protocol']['sha256'])
    a.eq('receipt:schema linkage', launch['output_schema_sha256'], manifest['review_output_schema']['sha256'])
    launch_by = {(x['reviewer_id'], x['chunk_number']): x for x in launch['contexts']}
    out_by = {(x['reviewer_id'], x['chunk_number']): x for x in outcomes['contexts']}
    context_by = {(x['reviewer_id'], x['chunk_number']): x for x in collection['chunks']}
    expected_keys = {(s, n) for s in STREAMS for n in range(23)}
    for name, mapping in [('launch', launch_by), ('outcome', out_by), ('retained', context_by)]:
        a.eq('contexts:' + name + ' complete', sorted(mapping), sorted(expected_keys))
    a.eq('receipt:unique context ids', len(set(x['independent_context_id'] for x in launch['contexts'])), 46)
    a.eq('receipt:unique first paths', len(set(str(a.path(x['first_artifact_path']).resolve()) for x in launch['contexts'])), 46)
    rows = []
    for stream in STREAMS:
        for number, prepared in enumerate(chunks):
            ident = (stream, number)
            saved = context_by[ident]
            start = launch_by[ident]
            outcome = out_by[ident]
            label = stream + '/' + str(number)
            for name, val in [('fresh_context', True), ('first_pass', True), ('peer_judgments_supplied', False)]:
                a.eq('receipt:' + label + '/' + name, start[name], val)
            a.eq('receipt:' + label + '/chunk', start['chunk_sha256'], manifest['chunks'][number]['sha256'])
            for name in ['reviewer_id', 'independent_context_id', 'chunk_number', 'chunk_sha256']:
                a.eq('receipt:' + label + '/' + name, outcome[name], start[name])
            a.eq('retained:' + label + '/launch', saved['launch_context'], start)
            a.eq('retained:' + label + '/outcome', saved['outcome_context'], outcome)
            expected_header = dict(reviewer_id=stream, independent_context_id=start['independent_context_id'], manifest_sha256=MANIFEST_SHA,
                chunk_sha256=start['chunk_sha256'], protocol_sha256=manifest['review_protocol']['sha256'], output_schema_sha256=manifest['review_output_schema']['sha256'])
            for name, val in expected_header.items():
                a.eq('retained:' + label + '/' + name, saved[name], val)
            raw = None if saved['first_bytes_base64'] is None else base64.b64decode(saved['first_bytes_base64'], validate=True)
            a.eq('retained:' + label + '/raw hash', saved['first_bytes_sha256'], None if raw is None else sha(raw))
            matching_first = False
            fd = outcome['first_artifact']
            if fd is not None:
                a.eq('receipt:' + label + '/first path', str(a.path(fd['path']).resolve()), str(a.path(start['first_artifact_path']).resolve()))
                try:
                    observed = a.read(fd['path'], parse=False)
                    a.eq('retained:' + label + '/actual raw hashes', None if raw is None else sha(raw), sha(observed))
                    matching_first = sha(observed) == fd['sha256']
                    if not matching_first:
                        a.eq('retained:' + label + '/hash error retained', bool(saved['contract_errors']), True)
                except OSError:
                    a.eq('retained:' + label + '/unreadable first diagnostic', bool(saved['contract_errors']), True)
            else:
                a.eq('retained:' + label + '/missing raw', raw, None)
            terminal = outcome['terminal_attestation']
            if terminal is not None:
                a.eq('receipt:' + label + '/observed terminal', terminal['observed_terminal'], True)
                a.eq('receipt:' + label + '/terminal identity', terminal['independent_context_id'], start['independent_context_id'])
                a.eq('receipt:' + label + '/terminal chunk', terminal['chunk_sha256'], start['chunk_sha256'])
                a.eq('receipt:' + label + '/terminal kind', terminal['observation_kind'] in ['completed', 'failed', 'interrupted'], True)
                observation = a.read(terminal['observation_artifact']['path'], terminal['observation_artifact']['sha256'], parse=False)
                a.eq('retained:' + label + '/terminal bytes', sha(base64.b64decode(saved['terminal_observation_bytes_base64'], validate=True)), sha(observation))
                a.eq('receipt:' + label + '/nonempty observation', bool(observation.strip()), True)
            else:
                a.eq('retained:' + label + '/no terminal bytes', saved['terminal_observation_bytes_base64'], None)
            a.eq('receipt:' + label + '/ready', matching_first or terminal is not None, True)
            parsed = None
            parse_good = False
            try:
                if raw is not None:
                    parsed = json.loads(raw)
                    parse_good = True
            except (ValueError, UnicodeError):
                pass
            a.eq('retained:' + label + '/parsed first', saved['parsed_first'], parsed)
            shape_good = parse_good and schema_ok(parsed, schema)
            contract_good = shape_good and matching_first
            judgments = {}
            references = {}
            record_validity = {}
            if shape_good:
                try:
                    for name, val in expected_header.items():
                        require(identical(parsed[name], val), 'chunk header ' + name)
                    ids = [j['id'] for j in parsed['records']]
                    require(len(ids) == len(set(ids)) and set(ids) == {r['id'] for r in prepared['records']}, 'chunk must contain exact twenty-four identities')
                    judgments = {j['id']: j for j in parsed['records']}
                    for r in prepared['records']:
                        try:
                            references[r['id']] = record_contract(judgments[r['id']], r)
                            record_validity[r['id']] = True
                        except (ValueError, KeyError, TypeError, IndexError):
                            record_validity[r['id']] = False
                    contract_good = contract_good and all(record_validity.values())
                except (ValueError, KeyError, TypeError, IndexError):
                    contract_good = False
            a.eq('chunk:' + label + '/structural validity', saved['structurally_valid'], bool(contract_good))
            a.stats['valid_chunks'] += bool(contract_good)
            a.stats['unestablished_chunks'] += not contract_good
            if record_validity:
                checks = {c['id']: c for c in saved['record_checks']}
                a.eq('chunk:' + label + '/record check inventory', list(checks), list(record_validity))
                for rid, valid in record_validity.items():
                    a.eq('chunk:' + label + '/' + rid + '/validity', checks[rid]['valid'], valid)
                    a.eq('chunk:' + label + '/' + rid + '/derived references', checks[rid]['property_references'], references.get(rid, []) if contract_good else [])
            if not contract_good:
                for check in saved['record_checks']:
                    a.eq('chunk:' + label + '/no invalid derived references', check['property_references'], [])
            for r in prepared['records']:
                rows.append(dict(reviewer_id=stream, independent_context_id=start['independent_context_id'], chunk_number=number,
                    id=r['id'], unit_id=r['unit_id'], text_sha256=r['text_sha256'], reference_established=r['reference_established'],
                    reference_measurement=r['reference_measurement'], review_state='retained_structurally_valid_judgment' if contract_good and r['reference_established'] else 'unestablished',
                    chunk_structurally_valid=bool(contract_good), first_judgment=judgments.get(r['id']) if contract_good else None,
                    semantic_correctness_established_by_code=False))
    a.eq('collection:all rows independently reconstructed', collection['rows'], rows)
    return rows


def reconstruct_bindings(a, r, j, table):
    if j is None:
        return None
    if not r['reference_established']:
        return []
    components = {encoded(x['component_path']): x for x in j['component_reviews']}
    evidences = {encoded(x['evidence_path']): x for x in j['evidence_reviews']}
    bindings = {encoded(x['binding_path']): x for x in j['binding_reviews']}
    units = {x['unit_id']: x for x in r['property_table']}
    atomics = {}
    for identity, prop in table.items():
        k = (prop['construction_unit_id'], encoded(prop['identity']['value_path']))
        require(k not in atomics, 'nonunique atomic mapping')
        atomics[k] = (identity, prop)
    result = []
    for b in r['bindings']:
        original = at(r['canonical_first_annotation'], b['path'])
        name = b['path'][1]
        composite = r['canonical_first_annotation']['composites'][name]
        cr = components[encoded(['composites', name])]
        br = bindings[encoded(b['path'])]
        features = []
        selected = []
        for w in br['feature_witnesses']:
            c = components[encoded(['features', w['feature']])]
            features.append(dict(feature=w['feature'], original_status=c['original_status'], first_component_assessment=c['assessment'],
                first_component_reason=c['notes'], selected_evidence_paths=w['evidence_paths']))
            selected.extend(w['evidence_paths'])
        selected.extend(br['binding_evidence_paths'])
        seen = set()
        grounds = []
        for p in selected:
            if encoded(p) in seen:
                continue
            seen.add(encoded(p))
            e = evidences[encoded(p)]
            operands = []
            for ref in e['properties']:
                u = units[ref['unit_id']]
                value = at(u['value'], ref['value_path'])
                require(identical(value, ref['value']), 'exact property operand mismatch')
                match = atomics.get((ref['unit_id'], encoded(ref['value_path'])))
                identity, prop = match if match is not None else (None, None)
                if prop is not None:
                    require(identical(prop['value'], value), 'bank atomic value mismatch')
                a.stats['selected_property_operands'] += 1
                a.stats['exact_atomic_operands' if match else 'unmapped_exact_operands'] += 1
                if type(value) in [dict, list]:
                    a.stats['container_operands_not_expanded'] += 1
                operands.append(dict(first_review_property=ref, construction_kind=u['kind'], construction_context=u.get('context'),
                    source_provenance=u.get('source_resolution'), atomic_identity_sha256=identity, atomic_identity=None if prop is None else prop['identity'],
                    atomic_mapping_state='exact' if match else 'no_exact_atomic_identity', missing_atomic_mapping_is_feature_absence=False))
            grounds.append(dict(first_evidence_review=e, exact_original_evidence=at(r['canonical_first_annotation'], p), operands=operands))
        facts = {
            'original_composite_nonabsent': composite['status'] != 'absent',
            'composite_application_supported': cr['assessment'] == 'supported',
            'required_features_nonabsent': all(f['original_status'] != 'absent' for f in features),
            'required_feature_applications_supported': all(f['first_component_assessment'] == 'supported' for f in features),
            'each_required_feature_has_selected_evidence': all(bool(f['selected_evidence_paths']) for f in features),
            'selected_evidence_grounding_supported': all(g['first_evidence_review']['grounding'] == 'supported' for g in grounds),
            'binding_compatibility_supported': br['assessment'] == 'supported',
        }
        a.stats['stream_binding_inventories'] += 1
        a.stats['stream_recorded_support_candidates'] += all(facts.values())
        result.append(dict(binding_path=b['path'], composite=name, original_composite_status=composite['status'], original_binding=original,
            original_scope=original['scope'], original_time=original['time'], first_composite_review=cr, first_binding_review=br,
            required_features=features, selected_evidence=grounds, recorded_support_facts=facts, all_recorded_support_facts_true=all(facts.values()),
            semantic_correctness_established_by_code=False, source_attributed_endpoint_established_by_code=False))
    return result


def run(a, args):
    for path, expected in FROZEN.items():
        a.read(path, expected, parse=False)
    schema_path = BASE + 'semantic_review/first_review_output_schema.json'
    schema = a.read(schema_path, FROZEN[schema_path])
    collection = a.read(args.collection, args.collection_sha256)
    produced = a.read(args.inventory, args.inventory_sha256)
    manifest = a.read(PREP + 'manifest.json', MANIFEST_SHA)
    bank = a.read(BANK, BANK_SHA)
    for item in [manifest, bank]:
        a.descriptors(item.get('input_dependencies', {}))
        a.source_map(item.get('source_sha256', {}))
    a.source_map(collection['source_sha256'])
    for field in ['review_protocol', 'review_output_schema', 'frozen_reference']:
        a.descriptors(manifest[field])
    a.eq('collection:schema', collection['schema'], CV)
    for field, value in [('manifest_sha256', MANIFEST_SHA), ('template_opportunities', 552), ('reviewer_opportunities', 1104),
            ('narrator_opportunities', 552), ('pair_opportunities', 684), ('whole_chunk_failure_policy', True),
            ('no_replacement_or_gap_filling', True), ('semantic_correctness_established_by_code', False), ('no_api_or_model_calls', True)]:
        a.eq('collection:' + field, collection[field], value)
    for name, size in [('chunks', 46), ('rows', 1104), ('pairs', 684), ('narrator_availability', 552)]:
        a.eq('collection:' + name + ' denominator', len(collection[name]), size)
    a.eq('collection:original pairs', collection['pairs'], manifest['pairs'])
    a.eq('collection:narrator availability', collection['narrator_availability'], manifest['narrator_availability'])
    a.eq('collection:reference lineage', collection['reference_lineage'], manifest['reference_lineage'])
    a.eq('manifest:chunks denominator', len(manifest['chunks']), 23)
    a.eq('manifest:streams', manifest['reviewer_streams'], STREAMS)
    chunks = []
    records = []
    for number, ref in enumerate(manifest['chunks']):
        c = a.read(PREP + ref['path'], ref['sha256'])
        a.eq('prepared:chunk number', c['chunk_number'], number)
        a.eq('prepared:chunk denominator', len(c['records']), 24)
        a.eq('prepared:chunk order', c['ordered_template_ids'], [r['id'] for r in c['records']])
        a.eq('prepared:manifest order', ref['ordered_template_ids'], c['ordered_template_ids'])
        for field in ['review_protocol', 'review_output_schema', 'frozen_reference']:
            a.eq('prepared:' + field, c[field], manifest[field])
        chunks.append(c)
        records.extend(c['records'])
    ids = [r['id'] for r in records]
    a.eq('prepared:complete unique identities', len(set(ids)), 552)
    a.eq('prepared:full frozen order', ids, manifest['ordered_template_ids'])
    salt = 'template_semantic_review_v1_order_20261019:'
    a.eq('prepared:order salt', manifest['order_salt'], salt)
    a.eq('prepared:salted order', ids, sorted(ids, key=lambda i: (sha((salt + i).encode()), i)))
    rows = inspect_collection(a, collection, manifest, chunks, schema)
    byrow = {(r['reviewer_id'], r['id']): r for r in rows}
    positions = []
    for record in records:
        streams = {}
        for stream in STREAMS:
            row = byrow[stream, record['id']]
            bs = reconstruct_bindings(a, record, row['first_judgment'], bank['property_tables'][record['id'].removesuffix('_template')])
            streams[stream] = dict(independent_context_id=row['independent_context_id'], chunk_number=row['chunk_number'],
                chunk_structurally_valid=row['chunk_structurally_valid'], review_state=row['review_state'], bindings=bs)
        candidate_paths = []
        for b in record['bindings']:
            path = b['path']
            selected = []
            for stream in STREAMS:
                bs = streams[stream]['bindings']
                selected.append(None if bs is None else next(x for x in bs if identical(x['binding_path'], path)))
            if all(x is not None and x['all_recorded_support_facts_true'] for x in selected):
                candidate_paths.append(path)
        positions.append(dict(id=record['id'], case=record['case'], reference_established=record['reference_established'],
            original_composite_statuses={name: c['status'] for name, c in record['canonical_first_annotation']['composites'].items()} if record['reference_established'] else None,
            first_streams=streams, both_first_streams_recorded_support_candidates=candidate_paths,
            candidates_require_semantic_and_source_closure=True, semantic_or_qualia_verdict=None))
    a.eq('inventory:every reconstructed position field', produced['positions'], positions)
    index = {p['id']: p for p in positions}
    a.eq('bank:pairs denominator', len(bank['pairs']), 684)
    require(len(bank['pairs']) == len(manifest['pairs']) == 684, 'complete pair denominator')
    pairs = []
    for original, numerical in zip(manifest['pairs'], bank['pairs']):
        a.eq('pairs:exact bank original', numerical['original_pair'], original)
        first = index[original['first_id'] + '_template']
        second = index[original['second_id'] + '_template']
        pairs.append(dict(original_pair=original, first_recorded_support_candidates=first['both_first_streams_recorded_support_candidates'],
            second_recorded_support_candidates=second['both_first_streams_recorded_support_candidates'], numerical_property_counts=numerical['counts'],
            numerical_by_actual_source=numerical['by_actual_source'], literal_text_equal=numerical['literal_text_equal'], semantic_predicate_tracking=None,
            semantic_true_null=None, semantic_exact_restoration=None, numerical_facts_do_not_establish_report_meaning=True))
    a.eq('inventory:every reconstructed pair field', produced['pairs'], pairs)
    summary = dict(template_opportunities=552, pair_opportunities=684,
        reference_established=sum(p['reference_established'] for p in positions), first_chunks_structurally_valid=a.stats['valid_chunks'],
        positions_with_both_first_streams_recorded_support_candidates=sum(bool(p['both_first_streams_recorded_support_candidates']) for p in positions),
        both_first_streams_recorded_support_candidate_bindings=sum(len(p['both_first_streams_recorded_support_candidates']) for p in positions),
        original_composite_statuses={name: dict(Counter(p['original_composite_statuses'][name] if p['reference_established'] else 'unestablished' for p in positions)) for name in ['COD', 'CA']})
    a.eq('inventory:summary', produced['summary'], summary)
    for name, expected in [('schema', 'attention_relational_template_binding_inventory_v1'),
            ('analysis_type', 'descriptive_join_of_retained_first_reviews_no_new_semantic_grader'),
            ('original_judgments_or_scopes_changed', False), ('missing_first_judgments_filled', False),
            ('semantic_endpoint_assessment_complete', False), ('goal_achieved', False)]:
        a.eq('inventory:' + name, produced[name], expected)
    a.eq('inventory:collection descriptor', produced['input_collection'], dict(path=str(args.collection), sha256=args.collection_sha256))
    a.eq('inventory:manifest descriptor', produced['prepared_manifest'], dict(path=PREP + 'manifest.json', sha256=MANIFEST_SHA))
    a.eq('inventory:bank descriptor', produced['numerical_bank'], dict(path=BANK, sha256=BANK_SHA))
    a.eq('inventory:production source', produced['source_sha256'], FROZEN['scripts/attention_relational_template_binding_inventory_v1.py'])
    return dict(recomputed_summary=summary, reconstructed_positions_sha256=sha(encoded(positions)), reconstructed_pairs_sha256=sha(encoded(pairs)),
        reconstructed_rows_sha256=sha(encoded(rows)), independently_reconstructed_rows=len(rows), independently_reconstructed_positions=len(positions),
        independently_reconstructed_pairs=len(pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readiness-confirmed', action='store_true', required=True)
    for name in ['collection', 'inventory']:
        parser.add_argument('--' + name, required=True, type=Path)
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    require(not (OUT / 'first_review.json').exists() and not (OUT / 'first_review.md').exists(), 'exclusive first output already exists')
    a = Audit()
    checker_raw = Path(__file__).read_bytes()
    result = {}
    fatal = None
    try:
        result = run(a, args)
    except Exception as exc:
        fatal = dict(type=type(exc).__name__, message=str(exc))
    limitations = [
        'Descriptive accounting only. Recorded support candidates are retained judgments, not semantic correctness, endpoint closure or qualia verdicts.',
        'Original frozen preparation, reference, numerical bank and production terminal provenance are hash-bound inputs; this check does not independently repeat the earlier numerical bank derivation or V3 runtime execution.',
        'Actual terminal receipt artifacts and first bytes can be checked for exact retention, not for truthful observation or independent training/commissioning.',
        'Reviewer reason adequacy, scope/content/time compatibility meaning, missed features and scientific interpretation remain outside this checker.',
        'Pair numerical counts/source groupings are checked as exact joins to the frozen numerical bank; numerical invariance does not establish a semantic true null.',
    ]
    report = dict(schema='attention_relational_template_complete_readout_accounting_review_v1_first',
        checker=dict(path=str(Path(__file__).resolve()), sha256=sha(checker_raw), standard_library_only=True, production_imports=False),
        root_explicit_readiness_confirmed=args.readiness_confirmed, original_denominators=dict(template=552, reviewer=1104, contexts=46, pairs=684),
        accounting_consistent=fatal is None and not a.discrepancies, fatal=fatal, discrepancies=a.discrepancies,
        input_bindings=sorted(a.inputs.values(), key=lambda x:x['path']), check_counts=dict(a.checks), descriptive_counts=dict(a.stats),
        semantic_verdict=None, qualia_verdict=None, source_or_first_evidence_edited=False, limitations=limitations, **result)
    raw = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    with (OUT / 'first_review.json').open('xb') as handle:
        handle.write(raw)
    summary = result.get('recomputed_summary', {})
    lines = ['# Complete template readout: independent descriptive accounting', '',
        'Accounting consistent: ' + str(report['accounting_consistent']) + '.',
        'Original denominators: 552 template positions; 1,104 reviewer positions; 46 contexts; 684 pairs.',
        'Discrepancies: ' + str(len(a.discrepancies)) + '. Fatal check error: ' + str(fatal) + '.', '',
        'Checker SHA-256: `' + sha(checker_raw) + '`.', 'First JSON review SHA-256: `' + sha(raw) + '`.', '',
        'Recomputed summary: `' + json.dumps(summary, sort_keys=True) + '`.', '',
        'All reconstructed row/position/pair fields are compared exactly, including original statuses/scopes/times, each selected witness and atomic property operand. Containers are not expanded. Whole-chunk failures remain unestablished.', '',
        'Limitations:', ''] + ['- ' + x for x in limitations]
    if a.discrepancies:
        lines += ['', 'Discrepancies:', ''] + ['- ' + x['check'] + ' (actual ' + x['actual_sha256'] + '; expected ' + x['expected_sha256'] + ')' for x in a.discrepancies]
    with (OUT / 'first_review.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(accounting_consistent=report['accounting_consistent'], discrepancies=len(a.discrepancies), fatal=fatal,
        json_sha256=sha(raw), markdown_sha256=sha((OUT / 'first_review.md').read_bytes()), summary=summary), sort_keys=True))


if __name__ == '__main__':
    main()
