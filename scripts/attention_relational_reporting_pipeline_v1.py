"""Read-only reporting-stage preparation/collection; import never calls an API.

No model, native renderer, optimizer or reserved-world generation is imported.
Scientific source export and API execution belong to their separately reviewed
modules. CLI outputs are exclusive first artifacts, never replacements.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from attention_relational_report_v1 import audit_report, required_propositions

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'audits/attention_relational_report_v1/protocol/manifest.json'
REFERENCE = 'audits/attention_relational_report_v1/reference/reference.json'
RATER_INSTRUCTIONS = 'audits/attention_relational_report_v1/reference/rater_instructions.txt'
INSTRUCTION_HASHES = {
    'narrator': ('audits/attention_relational_report_v1/protocol/narrator_instructions.txt',
                 '1ff6588dbc5f93701faf20c23c0d9d983ca1f437a6e919c5ecd3557124ad6745'),
    'audit': ('audits/attention_relational_report_v1/protocol/prose_auditor_instructions.txt',
              '79466e427215ceb47c6ceff027bd7b176bc7a6ec8f96c8f222d03400bd2e551a'),
    'rate': (RATER_INSTRUCTIONS,
             '9ff9a6823dcd2892ffc253681ff39a8ac494119c9ed1701a28627525bf3b8f33'),
}
REFERENCE_HASH = '6b1bd23576e95bae347213f36f6eed42982106386c6a4213704190c777d6cc53'
PROTOCOL_HASH = 'e8e59a80e47e3e39aa1d0b64d24260839606fc23ea924da3534ba5bdccbde786'
REPORT_SCHEMA = 'attention_relational_report_v1_report'
AUDIT_SCHEMA = 'attention_relational_report_v1_prose_audit'
RATING_SCHEMA = 'attention_relational_reference_v1_annotation'
BLIND_SALT = 'attention_relational_report_v1_blind_20261011'
ORDER_SALT = 'attention_relational_report_v1_order_20261011'
STATUSES = ('explicit', 'ambiguous', 'absent')
SCOPES = ('model_organization', 'model_access', 'claimed_perceptual_appearance',
          'display_or_sensor_property', 'unclear')
TIMES = ('current', 'past_observed', 'future_forecast', 'mixed', 'unclear')
AUXILIARY = ('basic_spatial_relation', 'identity_or_location_uncertainty',
             'answer_confidence', 'recovery_or_success_forecast',
             'source_process_distinction', 'experiential_assertion')
BASIS = ('peaks', 'relation', 'scalar', 'origin', 'record_scope', 'not_supplied')


def _object(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def _enum(*values):
    return {'enum': list(values), 'type': ['string', 'null'] if None in values else 'string'}


def _array(items):
    return {'type': 'array', 'items': items}


_STRING = {'type': 'string'}
_BOOLEAN = {'type': 'boolean'}
_BINDING_SCHEMA = _object({
    'panel': _enum('p0', 'p1', None), 'view': _enum('current', 'candidate', 'record_scope'),
    'command': _enum(None, 'k0', 'k1', 'k2'), 'node': _enum(None, 'n0', 'n1', 'n2', 'n3'),
    'role': _enum(None, 'red', 'blue'), 'basis': _enum(*BASIS)})
PROSE_AUDIT_RESPONSE_SCHEMA = _object({
    'schema': _enum(AUDIT_SCHEMA),
    'claims': _array(_object({'quote': _STRING, 'verdict': _enum('supported', 'contradicted', 'not_auditable'),
                             'reason': _STRING, 'bindings': _array(_BINDING_SCHEMA)})),
    'current_covered': _BOOLEAN, 'conditional_covered': _BOOLEAN, 'unparsed_or_ambiguous': _BOOLEAN})
_EVIDENCE_SCHEMA = _object({'quote': _STRING, 'start': {'type': 'integer'}, 'end': {'type': 'integer'},
    'referent': _STRING, 'attribute': _STRING, 'scope': _enum(*SCOPES), 'time': _enum(*TIMES)})
_FEATURE_SCHEMA = _object({'status': _enum(*STATUSES), 'evidence': _array(_EVIDENCE_SCHEMA), 'rationale': _STRING})
_AUXILIARY_SCHEMA = _object({'status': _enum(*STATUSES), 'evidence': _array(_EVIDENCE_SCHEMA)})
_COMPOSITE_SCHEMA = _object({'status': _enum(*STATUSES), 'bindings': _array(_object({
    'referent': _STRING, 'attribute': _STRING, 'scope': _enum(*SCOPES), 'time': _enum(*TIMES),
    'evidence': _array(_EVIDENCE_SCHEMA)})), 'rationale': _STRING})
RATER_RESPONSE_SCHEMA = _object({'schema': _enum(RATING_SCHEMA), 'unit_id': _STRING,
    'features': _object({k: _FEATURE_SCHEMA for k in ('C', 'O', 'D', 'A')}),
    'composites': _object({k: _COMPOSITE_SCHEMA for k in ('COD', 'CA')}),
    'auxiliary': _object({k: _AUXILIARY_SCHEMA for k in AUXILIARY}), 'limitations': _array(_STRING)})


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decoded(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON object key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def bound_json(path, expected_sha256):
    raw = Path(path).read_bytes()
    require(digest(raw) == expected_sha256, 'input SHA-256 mismatch')
    return decoded(raw)


def frozen_instruction(stage):
    path, expected = INSTRUCTION_HASHES[stage]
    raw = (ROOT / path).read_bytes()
    require(digest(raw) == expected, 'frozen instruction SHA-256 mismatch')
    return raw.decode('utf-8').rstrip('\n')


def frozen_reference():
    value = bound_json(ROOT / REFERENCE, REFERENCE_HASH)
    require(value['schema'] == 'attention_relational_reference_v1', 'wrong reference schema')
    return value


def _index(rows, key='id'):
    result = {}
    for row in rows:
        require(type(row) is dict and type(row.get(key)) is str, 'invalid inventory row')
        require(row[key] not in result, 'duplicate inventory identifier')
        result[row[key]] = row
    return result


def validate_export(exports, protocol_manifest=None):
    """Complete fixed inventory, public prompt and template checks; no source calls."""
    if protocol_manifest is None:
        protocol_manifest = bound_json(ROOT / PROTOCOL, PROTOCOL_HASH)
    expected = protocol_manifest['request_ids']
    requests = _index(exports['requests'])
    require(list(requests) == expected and len(expected) == 552, 'fixed552 request inventory differs')
    packets = _index(exports['packets'])
    provenance = _index(exports['provenance'])
    templates = _index(exports['templates'])
    require(set(packets) == set(provenance) == set(templates) == set(expected),
            'export inventories differ')
    instruction = frozen_instruction('narrator')
    peaks = 0
    for identifier in expected:
        packet = packets[identifier]
        raw = encoded(packet['content'])
        require(packet['content_sha256'] == digest(raw), 'neutral packet hash mismatch')
        require(requests[identifier] == {'id': identifier, 'input': instruction + '\n' + raw.decode('ascii')},
                'narrator prompt differs from frozen neutral input')
        case = provenance[identifier]['case']
        require(case['encoder_seed'] in (1301, 1311, 1321)
                and case['world_index'] in range(4) and case['decision_time'] == 12,
                'case outside fixed schedule')
        require(case['view'] in ('A_only', 'joined', 'W_only', 'sensor_only'), 'unknown view')
        require(identifier == (f"s{case['encoder_seed']}_w{case['world_index']:02d}_t12_"
                               f"{case['source']}_{case['view']}_{case['variant']}"),
                'scientific identifier differs from source-aware case')
        count = len(required_propositions(packet['content']))
        require(count == (64 if case['view'] == 'joined' else 8 if case['view'] == 'sensor_only' else 32),
                'required distribution denominator differs')
        peaks += count
        template = templates[identifier]['report']
        require(audit_report(packet['content'], template)['listed_machine_facts_pass'],
                'deterministic template failed its source audit')
        require(type(template['text']) is str and 0 < len(template['text'].split()) <= 220,
                'template text budget differs')
    require(peaks == 25440, 'fixed25440 required peak inventory differs')
    return {'request_ids': expected, 'required_peak_count': peaks,
            'packets': packets, 'provenance': provenance, 'templates': templates}


def _unwrap_outcome(value):
    if type(value) is not dict:
        raise ValueError('outcome is not an object')
    if 'first_outcome' in value:
        require(type(value['first_outcome']) is dict, 'invalid secondary first outcome')
        return value['first_outcome']
    if 'retained_first_outcome_envelope' in value:
        return _unwrap_outcome(value['retained_first_outcome_envelope'])
    return value


def _refused(value):
    if type(value) is dict:
        return value.get('type') == 'refusal' or any(_refused(v) for v in value.values())
    if type(value) is list:
        return any(_refused(v) for v in value)
    return False


def _response_record(identifier, raw_outcome, provenance):
    outcome = _unwrap_outcome(raw_outcome)
    require(outcome.get('id') == identifier, 'first outcome identifier mismatch')
    parsed = None
    parse_error = None
    text = outcome.get('output_text')
    if type(text) is str:
        try:
            parsed = decoded(text)
            if 'parsed' in outcome and outcome['parsed'] != parsed:
                parse_error = 'stored parsed value differs from raw output_text'
                parsed = None
        except (ValueError, TypeError):
            parse_error = 'raw output_text is not JSON'
    response = outcome.get('response', {})
    api_status = outcome.get('api_response_status', response.get('status') if type(response) is dict else None)
    transport_complete = (outcome.get('status') == 'received' and api_status == 'completed'
                          and not _refused(response))
    return {'id': identifier, 'availability': 'retained_first_outcome',
            'transport_complete': transport_complete, 'api_status': api_status,
            'parsed': parsed, 'parse_error': parse_error,
            'raw_outcome': deepcopy(raw_outcome), 'storage_provenance': provenance}


def runner_request_sha256(request):
    # Exact encoding in run_attention_relational_reports_v1.encoded, not the
    # packet's separate compact canonical encoding.
    return digest((json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def build_collection_binding(manifest, manifest_sha256, requests, requests_sha256,
                             lineage, lineage_sha256):
    require(manifest.get('schema') == 'attention_relational_report_v1_runtime_manifest', 'wrong runtime manifest')
    require(manifest.get('stage') in ('report', 'audit', 'rate'), 'wrong collection stage')
    require(manifest['source_sha256'].get(manifest['requests_path']) == requests_sha256,
            'runtime manifest does not bind exact requests bytes')
    require(manifest['source_sha256'].get(manifest['lineage_path']) == lineage_sha256,
            'runtime manifest does not bind exact lineage bytes')
    require(list(_index(requests)) == manifest['request_ids']
            and len(requests) == manifest['request_count'], 'runtime request inventory differs')
    require(lineage.get('schema') == 'attention_relational_report_v1_stage_lineage'
            and lineage.get('stage') == manifest['stage'], 'lineage stage differs')
    require(lineage['protocol_manifest_sha256'] == PROTOCOL_HASH, 'protocol lineage differs')
    require(type(lineage['artifact_sha256']) is dict and bool(lineage['artifact_sha256']),
            'raw dependency artifact SHA inventory missing')
    for path, sha in lineage['artifact_sha256'].items():
        require(manifest['source_sha256'].get(path) == sha,
                'reviewed stage does not bind every lineage artifact')
    return {'schema': 'attention_relational_report_v1_collection_binding',
        'stage': manifest['stage'], 'manifest_sha256': manifest_sha256,
        'requests_sha256': requests_sha256, 'lineage_sha256': lineage_sha256,
        'request_ids': manifest['request_ids'], 'requests': deepcopy(requests),
        'lineage': deepcopy(lineage), 'output_root': manifest['output_root']}


def collect_first_responses(completion_path, expected_sha256, request_ids,
                            output_root, repository_root=ROOT, *, stage_binding):
    """Completion is authoritative, including secondary/envelope storage failures.

    No id.json fallback, retry, model call or successful-output filtering. An
    interrupted batch lacking completion can be recorded using the separate
    collect_interrupted_responses helper, never pretending it completed.
    """
    completion = bound_json(completion_path, expected_sha256)
    require(completion.get('schema') == 'attention_relational_report_v1_first_batch_completion',
            'wrong first completion schema')
    require(completion.get('retry_performed') is False, 'retried batch is not a firstpass')
    require(stage_binding.get('schema') == 'attention_relational_report_v1_collection_binding'
            and completion['manifest_sha256'] == stage_binding['manifest_sha256'],
            'completion belongs to a different exact reviewed stage manifest')
    require(request_ids == stage_binding['request_ids'], 'collection requested inventory differs from stage')
    request_by_id = _index(stage_binding['requests'])
    reviewed_root = Path(stage_binding['output_root'])
    if not reviewed_root.is_absolute():
        reviewed_root = Path(repository_root) / reviewed_root
    require(Path(output_root).resolve() == reviewed_root.resolve(), 'collection output root differs from stage')
    rows = _index(completion['outcomes'])
    require(completion['request_count'] == len(request_ids) and set(rows) <= set(request_ids),
            'completion opportunity inventory differs')
    stage_root = Path(output_root).resolve()
    records = []
    for identifier in request_ids:
        row = rows.get(identifier)
        if row is None:
            records.append({'id': identifier, 'availability': 'missing', 'transport_complete': False,
                            'parsed': None, 'raw_outcome': None, 'storage_provenance': {}})
            continue
        if 'retained_first_outcome_envelope' in row:
            require(row.get('status') == 'storage_failure', 'unexpected retained-envelope status')
            raw = row['retained_first_outcome_envelope']
            provenance = {'kind': 'completion_retained_storage_failure',
                          'completion_sha256': expected_sha256,
                          'envelope_sha256': digest(encoded(raw))}
        else:
            require('response_path' in row and 'response_sha256' in row,
                    'first outcome has no bound response path')
            path = Path(row['response_path'])
            if not path.is_absolute():
                path = Path(repository_root) / path
            path = path.resolve()
            require(path.is_relative_to(stage_root), 'response path outside reviewed output root')
            raw = bound_json(path, row['response_sha256'])
            provenance = {'kind': 'completion_bound_response_file', 'path': str(path),
                          'sha256': row['response_sha256'], 'completion_sha256': expected_sha256}
        underlying = _unwrap_outcome(raw)
        require(underlying.get('request_sha256') == runner_request_sha256(request_by_id[identifier]),
                'first response belongs to a different exact request input')
        records.append(_response_record(identifier, raw, provenance))
    return {'schema': 'attention_relational_report_v1_collected_first_responses',
            'completion_sha256': expected_sha256, 'manifest_sha256': completion['manifest_sha256'],
            'request_count': len(request_ids), 'records': records,
            'stage_binding': deepcopy(stage_binding),
            'all_first_outcomes_preserved': True, 'retry_performed': False}


def collect_interrupted_responses(request_ids, output_root, *, stage_binding):
    """Retain partial files without resolving ambiguous pre-existing paths."""
    root = Path(output_root)
    require(request_ids == stage_binding['request_ids'] and root.resolve() == Path(stage_binding['output_root']).resolve(),
            'interrupted collection differs from reviewed stage')
    request_by_id = _index(stage_binding['requests'])
    require(not (root / 'first_completion.json').exists(), 'use bound completed-batch collector')
    records = []
    for identifier in request_ids:
        require('/' not in identifier and '\\' not in identifier, 'unsafe request identifier')
        paths = [p for p in (root / (identifier + '.json'), root / (identifier + '.secondary.json')) if p.exists()]
        if len(paths) == 1:
            raw_bytes = paths[0].read_bytes()
            value = json.loads(raw_bytes)
            require(_unwrap_outcome(value).get('request_sha256') == runner_request_sha256(request_by_id[identifier]),
                    'interrupted response request input differs')
            records.append(_response_record(identifier, value,
                {'kind': 'interrupted_partial_file', 'path': str(paths[0]), 'sha256': digest(raw_bytes)}))
        else:
            records.append({'id': identifier, 'availability': 'ambiguous_partial_storage' if paths else 'missing',
                'transport_complete': False, 'parsed': None, 'raw_outcome': None,
                'storage_provenance': {'candidates': [{'path': str(p), 'sha256': digest(p.read_bytes()),
                                                      'raw': json.loads(p.read_bytes())} for p in paths]}})
    return {'schema': 'attention_relational_report_v1_collected_first_responses',
            'request_count': len(request_ids), 'records': records, 'interrupted_batch': True,
            'stage_binding': deepcopy(stage_binding),
            'all_available_partial_files_preserved': True, 'retry_performed': False}


def _records(collection):
    return _index(collection['records'])


def _report_text(record):
    parsed = record.get('parsed') if record is not None else None
    if (record and record.get('transport_complete') is True and type(parsed) is dict
            and set(parsed) == {'schema', 'text', 'propositions'} and parsed.get('schema') == REPORT_SCHEMA
            and type(parsed.get('text')) is str and type(parsed.get('propositions')) is list):
        return parsed['text']
    return ''


def build_prose_audit_requests(first_responses, audit_packets):
    """Only supported interface material: exact neutral packet plus first prose."""
    records = _records(first_responses)
    packets = _index(audit_packets)
    instruction = frozen_instruction('audit')
    requests = []
    for identifier, packet in packets.items():
        text = _report_text(records.get(identifier))
        if text.strip():
            payload = {'packet': packet['content'], 'report_text': text}
            requests.append({'id': identifier, 'input': instruction + '\n' + encoded(payload).decode('ascii')})
    return requests


def blind_id(identifier):
    return digest((BLIND_SALT + ':' + identifier).encode('utf-8'))[:24]


def build_blind_rater_requests(first_texts, templates, frozen_reference_value=None):
    """One text/reference only. No source, provenance, status or factual judgment."""
    reference = frozen_reference() if frozen_reference_value is None else frozen_reference_value
    require(digest(encoded(reference)) == digest(encoded(frozen_reference())), 'reference contents differ')
    records = _records(first_texts)
    instruction = frozen_instruction('rate')
    requests = []
    for template in templates:
        identifier = template['id']
        for original_id, text in ((identifier, _report_text(records.get(identifier))),
                                  (identifier + '_template', template['report']['text'])):
            unit_id = blind_id(original_id)
            payload = {'reference': reference, 'unit_id': unit_id, 'report_text': text}
            requests.append({'id': unit_id, 'input': instruction + '\n' + encoded(payload).decode('ascii')})
    require(len({row['id'] for row in requests}) == len(requests), 'opaque rating ID collision')
    requests.sort(key=lambda row: digest((ORDER_SALT + ':' + row['id']).encode('ascii')))
    return requests


def build_stage_lineage(stage, exports, artifact_sha256, first_responses=None):
    require(stage in ('report', 'audit', 'rate'), 'unknown prepared stage')
    value = {'schema': 'attention_relational_report_v1_stage_lineage', 'stage': stage,
        'protocol_manifest_sha256': PROTOCOL_HASH,
        'exports_content_sha256': digest(encoded(exports)),
        'artifact_sha256': deepcopy(artifact_sha256)}
    if stage in ('audit', 'rate'):
        require(first_responses is not None, 'first report dependency required')
        require(first_responses['stage_binding']['stage'] == 'report', 'foreign first report stage')
        value.update(first_report_collection_content_sha256=digest(encoded(first_responses)),
            first_report_manifest_sha256=first_responses['stage_binding']['manifest_sha256'],
            first_report_completion_sha256=first_responses.get('completion_sha256'))
    if stage == 'rate':
        value.update(template_reports_content_sha256=digest(encoded(exports['templates'])),
            reference_sha256=REFERENCE_HASH,
            rater_instructions_sha256=INSTRUCTION_HASHES['rate'][1])
    return value


def _validate_collection_lineage(collection, stage, requests, exports, first_reports=None):
    binding = collection.get('stage_binding')
    require(type(binding) is dict and binding.get('schema') == 'attention_relational_report_v1_collection_binding'
            and binding.get('stage') == stage, 'collected batch has no exact stage binding')
    require(binding['requests'] == requests and binding['request_ids'] == list(_index(requests)),
            'collected same-ID batch has different prepared request inputs')
    require(collection.get('manifest_sha256', binding['manifest_sha256']) == binding['manifest_sha256'],
            'collection manifest differs from exact stage binding')
    expected = build_stage_lineage(stage, exports, binding['lineage']['artifact_sha256'], first_reports)
    require(binding['lineage'] == expected, 'export/firstreport/template/reference lineage differs')
    require(set(_records(collection)) <= set(binding['request_ids']), 'foreign collected record identifier')


def _public_contexts(packet):
    contexts = {}
    for panel in packet['panels']:
        for node in panel['current']:
            contexts[(panel['panel'], 'current', None, node['node'])] = node
        if panel['candidates'] is not None:
            for command, nodes in panel['candidates'].items():
                for node in nodes:
                    contexts[(panel['panel'], 'candidate', command, node['node'])] = node
    return contexts


def validate_prose_audit(packet, text, value):
    errors = []
    def bad(message):
        errors.append(message)
    required = {'schema', 'claims', 'current_covered', 'conditional_covered', 'unparsed_or_ambiguous'}
    if type(value) is not dict or set(value) != required or value.get('schema') != AUDIT_SCHEMA:
        return {'valid': False, 'errors': ['invalid prose-audit schema'], 'fidelity_pass': False}
    for key in ('current_covered', 'conditional_covered', 'unparsed_or_ambiguous'):
        if type(value[key]) is not bool:
            bad('nonboolean audit coverage flag')
    contexts = _public_contexts(packet)
    panels = {panel['panel'] for panel in packet['panels']}
    if type(value['claims']) is not list:
        return {'valid': False, 'errors': ['claims is not a list'], 'fidelity_pass': False}
    for claim in value['claims']:
        if type(claim) is not dict or set(claim) != {'quote', 'verdict', 'reason', 'bindings'}:
            bad('invalid audit claim keys'); continue
        if type(claim['quote']) is not str or not claim['quote'] or claim['quote'] not in text:
            bad('audit quote is not an exact nonempty report substring')
        if claim['verdict'] not in ('supported', 'contradicted', 'not_auditable') or type(claim['reason']) is not str:
            bad('invalid claim verdict or rationale')
        if type(claim['bindings']) is not list:
            bad('audit bindings is not a list'); continue
        if claim['verdict'] == 'supported' and not claim['bindings']:
            bad('supported claim has no source binding')
        for binding in claim['bindings']:
            if type(binding) is not dict or set(binding) != {'panel', 'view', 'command', 'node', 'role', 'basis'}:
                bad('invalid audit binding keys'); continue
            if (binding['panel'] not in ('p0', 'p1', None)
                    or binding['view'] not in ('current', 'candidate', 'record_scope')
                    or binding['command'] not in (None, 'k0', 'k1', 'k2')
                    or binding['node'] not in (None, 'n0', 'n1', 'n2', 'n3')):
                bad('invalid audit binding identifier'); continue
            if binding['basis'] not in BASIS:
                bad('invalid source basis')
            if binding['role'] not in (None, 'red', 'blue'):
                bad('invalid bound role')
            if binding['view'] == 'record_scope':
                if binding['panel'] not in panels | {None} or any(binding[k] is not None for k in ('command', 'node', 'role')):
                    bad('invalid record-scope binding')
                if binding['basis'] not in ('record_scope', 'origin', 'not_supplied'):
                    bad('numeric basis cannot be record-scope')
            elif (binding['panel'], binding['view'], binding['command'], binding['node']) not in contexts:
                bad('audit binding names an unsupplied context')
            if binding['basis'] in ('peaks', 'scalar') and binding['role'] is None:
                bad('role-specific basis has no role')
            if claim['verdict'] == 'supported' and binding['basis'] == 'not_supplied':
                bad('supported claim cites not-supplied evidence')
    candidates = any(p['candidates'] is not None for p in packet['panels'])
    all_supported = bool(value['claims']) and all(c.get('verdict') == 'supported'
        for c in value['claims'] if type(c) is dict)
    passed = (not errors and all_supported and value['current_covered'] is True
              and (not candidates or value['conditional_covered'] is True)
              and value['unparsed_or_ambiguous'] is False)
    return {'valid': not errors, 'errors': errors, 'fidelity_pass': passed,
            'claim_count': len(value['claims']), 'verdict_counts': dict(Counter(
                c['verdict'] if type(c) is dict and type(c.get('verdict')) is str
                    and c['verdict'] in ('supported', 'contradicted', 'not_auditable') else 'invalid'
                for c in value['claims'])),
            'semantic_extraction_is_fallible': True}


def validate_blind_annotation(text, unit_id, value):
    errors = []
    evidence_diagnostics = []
    def bad(message):
        errors.append(message)
    keys = {'schema', 'unit_id', 'features', 'composites', 'auxiliary', 'limitations'}
    if type(value) is not dict or set(value) != keys or value.get('schema') != RATING_SCHEMA or value.get('unit_id') != unit_id:
        return {'valid': False, 'errors': ['invalid blind annotation schema or unit ID']}
    def evidence(entries):
        if type(entries) is not list:
            bad('evidence is not a list'); return []
        for entry in entries:
            required = {'quote', 'start', 'end', 'referent', 'attribute', 'scope', 'time'}
            if type(entry) is not dict or set(entry) != required:
                bad('invalid evidence keys'); continue
            start, end = entry['start'], entry['end']
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                    or type(entry['quote']) is not str or text[start:end] != entry['quote']):
                bad('evidence Unicode offsets/quote do not match exact text')
                present = type(entry['quote']) is str and bool(entry['quote']) and entry['quote'] in text
                evidence_diagnostics.append({'kind': 'quote_present_offset_mismatch' if present else 'quote_not_present',
                    'quote': entry['quote'], 'start': start, 'end': end,
                    'no_offset_or_label_repair_performed': True})
            if entry['scope'] not in SCOPES or entry['time'] not in TIMES:
                bad('invalid evidence scope/time')
            if any(type(entry[k]) is not str or not entry[k].strip() for k in ('referent', 'attribute')):
                bad('missing evidence referent/attribute')
        return entries
    for name, expected in (('features', {'C', 'O', 'D', 'A'}), ('composites', {'COD', 'CA'}),
                           ('auxiliary', set(AUXILIARY))):
        if type(value[name]) is not dict or set(value[name]) != expected:
            bad('invalid ' + name + ' inventory')
    if errors:
        return {'valid': False, 'errors': errors, 'evidence_diagnostics': evidence_diagnostics}
    for name, obj in list(value['features'].items()) + list(value['auxiliary'].items()):
        expected = {'status', 'evidence', 'rationale'} if name in ('C', 'O', 'D', 'A') else {'status', 'evidence'}
        if type(obj) is not dict or set(obj) != expected or obj.get('status') not in STATUSES:
            bad('invalid feature record'); continue
        evidence(obj['evidence'])
        if obj['status'] == 'explicit' and not obj['evidence']:
            bad('explicit feature lacks evidence')
        if 'rationale' in obj and type(obj['rationale']) is not str:
            bad('invalid feature rationale')
    if errors:
        return {'valid': False, 'errors': errors, 'evidence_diagnostics': evidence_diagnostics}
    for name, obj in value['composites'].items():
        if type(obj) is not dict or set(obj) != {'status', 'bindings', 'rationale'} or obj.get('status') not in STATUSES:
            bad('invalid composite record'); continue
        if type(obj['bindings']) is not list or type(obj['rationale']) is not str:
            bad('invalid composite bindings/rationale'); continue
        needed = ('C', 'O', 'D') if name == 'COD' else ('C', 'A')
        statuses = [value['features'][f].get('status') for f in needed]
        if obj['status'] == 'explicit' and (any(s != 'explicit' for s in statuses) or not obj['bindings']):
            bad('explicit composite lacks explicit required features/bindings')
        if obj['status'] == 'ambiguous' and 'absent' in statuses:
            bad('ambiguous composite has an absent required feature')
        for binding in obj['bindings']:
            if type(binding) is not dict or set(binding) != {'referent', 'attribute', 'scope', 'time', 'evidence'}:
                bad('invalid composite binding'); continue
            entries = evidence(binding['evidence'])
            if binding['scope'] not in SCOPES or binding['time'] not in TIMES:
                bad('invalid composite binding scope/time')
            if any(type(binding[k]) is not str or not binding[k].strip() for k in ('referent', 'attribute')):
                bad('missing composite referent/attribute')
            if obj['status'] == 'explicit':
                for feature in needed:
                    matches = [e for e in value['features'][feature].get('evidence', []) if type(e) is dict
                        and e.get('referent') == binding['referent'] and e.get('time') == binding['time']
                        and e.get('scope') == binding['scope'] and e in entries]
                    if not matches:
                        bad('explicit composite lacks compatible feature evidence for ' + feature)
    if type(value['limitations']) is not list or any(type(x) is not str for x in value['limitations']):
        bad('invalid limitations inventory')
    return {'valid': not errors, 'errors': errors, 'evidence_diagnostics': evidence_diagnostics,
            'semantic_rating_is_fallible': True}


def _prop_key(prop):
    return tuple(prop.get(k) for k in ('kind', 'panel', 'view', 'command', 'node', 'role', 'name', 'relation'))


def _claim_changes(left, right, panel):
    """Matched listed claims only; absence is a coverage change, not a paraphrase."""
    def claims(report):
        result = {}
        for atom in report.get('propositions', []) if type(report) is dict else []:
            if type(atom) is dict and atom.get('panel') == panel:
                key = encoded(list(_prop_key(atom)))
                result.setdefault(key, []).append(atom)
        return result
    a, b = claims(left), claims(right)
    return {'matched_claim_value_changes': [dict(key=json.loads(k), first=a[k], second=b[k])
                for k in sorted(a.keys() & b.keys()) if a[k] != b[k]],
            'only_first_claim_keys': [json.loads(k) for k in sorted(a.keys() - b.keys())],
            'only_second_claim_keys': [json.loads(k) for k in sorted(b.keys() - a.keys())]}


def build_pair_index(provenance):
    cases = defaultdict(dict)
    for row in provenance:
        c = row['case']
        key = (c['encoder_seed'], c['world_index'], c['source'], c['view'])
        require(c['variant'] not in cases[key], 'duplicate causal case variant')
        cases[key][c['variant']] = row['id']
    pairs = []
    for key, variants in cases.items():
        source = key[2]
        opportunities = []
        if source in ('A', 'ordinary', 'analytic'):
            for schedule in ('current', 'prospective'):
                opportunities += [(schedule + '_donor', 'baseline', schedule + '_donor'),
                    (schedule + '_donor_to_restored', schedule + '_donor', schedule + '_restored'),
                    (schedule + '_baseline_to_restored', 'baseline', schedule + '_restored')]
        if source != 'sensor':
            opportunities += [('W_changed', 'baseline', 'W_changed'),
                              ('W_changed_to_restored', 'W_changed', 'W_restored'),
                              ('W_baseline_to_restored', 'baseline', 'W_restored')]
        for kind, first, second in opportunities:
            require(first in variants and second in variants, 'missing fixed causal pair member')
            pairs.append({'kind': kind, 'case': list(key), 'first_id': variants[first], 'second_id': variants[second]})
    return pairs


def summarize_reporting_results(first_responses, machine_audits, prose_audits,
                                blind_ratings, pair_index, exports, protocol_manifest=None):
    """Descriptive complete inventory; no runtime approval or theory success flag."""
    inventory = validate_export(exports, protocol_manifest)
    reports = _records(first_responses)
    audits = _records(prose_audits)
    ratings = _records(blind_ratings)
    _validate_collection_lineage(first_responses, 'report', exports['requests'], exports)
    _validate_collection_lineage(prose_audits, 'audit',
        build_prose_audit_requests(first_responses, exports['packets']), exports, first_responses)
    _validate_collection_lineage(blind_ratings, 'rate',
        build_blind_rater_requests(first_responses, exports['templates']), exports, first_responses)
    require(pair_index == build_pair_index(exports['provenance']), 'complete causal pair inventory differs')
    expected_ids = set(inventory['request_ids'])
    require(set(reports) <= expected_ids and set(audits) <= expected_ids,
            'unknown report or auditor identifier')
    require(set(ratings) <= {blind_id(i + suffix) for i in expected_ids for suffix in ('', '_template')},
            'unknown blind rater identifier')
    machine_by_id = {} if machine_audits is None else _index(machine_audits)
    results, groups, fidelity = [], defaultdict(Counter), {}
    by_id = {}
    for identifier in inventory['request_ids']:
        content = inventory['packets'][identifier]['content']
        case = inventory['provenance'][identifier]['case']
        received = reports.get(identifier)
        report = received.get('parsed') if received else None
        machine = audit_report(content, report)
        if identifier in machine_by_id:
            supplied = machine_by_id[identifier].get('audit')
            require(supplied == machine, 'saved machine audit differs from deterministic recomputation')
        text = _report_text(received)
        audit_received = audits.get(identifier)
        prose = validate_prose_audit(content, text, audit_received.get('parsed') if audit_received else None)
        established = (bool(received and received.get('transport_complete')) and bool(text.strip())
            and machine['listed_machine_facts_pass'] and prose['fidelity_pass']
            and bool(audit_received and audit_received.get('transport_complete')))
        fidelity[identifier] = established
        by_id[identifier] = {'text': text, 'report': report, 'prose_audit': audit_received}
        for template in (False, True):
            text_id = identifier + ('_template' if template else '')
            exact_text = inventory['templates'][identifier]['report']['text'] if template else text
            unit_id = blind_id(text_id)
            rating_received = ratings.get(unit_id)
            annotation = rating_received.get('parsed') if rating_received else None
            validation = validate_blind_annotation(exact_text, unit_id, annotation)
            annotation_valid = bool(rating_received and rating_received.get('transport_complete')) and validation['valid']
            text_available = template or bool(text.strip())
            rating_valid = annotation_valid and text_available
            true_fidelity = True if template else established
            group_key = '|'.join(str(case[k]) for k in ('encoder_seed', 'source', 'view', 'variant')) + ('|template' if template else '|narrator')
            count = groups[group_key]
            count['text_opportunities'] += 1
            count['fidelity_established'] += int(true_fidelity)
            count['valid_rating'] += int(rating_valid)
            count['reference_outcome_unestablished_due_actor_text'] += int(not text_available)
            count['rating_invalid'] += int(bool(rating_received and rating_received.get('transport_complete'))
                                           and not validation['valid'])
            count['rating_missing_or_incomplete'] += int(not rating_received or not rating_received.get('transport_complete'))
            for diagnostic in validation.get('evidence_diagnostics', []):
                count['rating_' + diagnostic['kind']] += 1
            count['required_peak_atoms'] += machine['required_peak_count']
            count['correct_required_peak_atoms'] += machine['required_peak_count'] if template else machine['covered_peak_count']
            if not template:
                for error in machine['errors']:
                    count['machine_error_' + error['category']] += 1
            feature_statuses = {name: annotation['features'][name]['status'] if rating_valid else 'unestablished'
                                for name in ('C', 'O', 'D', 'A')}
            auxiliary_statuses = {name: annotation['auxiliary'][name]['status'] if rating_valid else 'unestablished'
                                  for name in AUXILIARY}
            for name, status in list(feature_statuses.items()) + list(auxiliary_statuses.items()):
                count['feature_' + name + '_' + status] += 1
            endpoints = {}
            for endpoint in ('COD', 'CA'):
                obj = annotation['composites'][endpoint] if rating_valid else None
                status = obj['status'] if obj else 'unestablished'
                lower, upper = status == 'explicit', status in ('explicit', 'ambiguous')
                count[endpoint + '_lower'] += int(lower)
                count[endpoint + '_upper'] += int(upper)
                count[endpoint + '_faithful_lower'] += int(lower and true_fidelity)
                count[endpoint + '_faithful_upper'] += int(upper and true_fidelity)
                scopes = sorted({b['scope'] for b in obj['bindings']}) if obj else []
                times = sorted({b['time'] for b in obj['bindings']}) if obj else []
                for scope in SCOPES:
                    count[endpoint + '_' + scope + '_lower'] += int(lower and scope in scopes)
                    count[endpoint + '_' + scope + '_upper'] += int(upper and scope in scopes)
                    count[endpoint + '_' + scope + '_faithful_lower'] += int(lower and true_fidelity and scope in scopes)
                    count[endpoint + '_' + scope + '_faithful_upper'] += int(upper and true_fidelity and scope in scopes)
                endpoints[endpoint] = {'status': status, 'lower': lower, 'upper': upper,
                    'faithful_lower': lower and true_fidelity, 'faithful_upper': upper and true_fidelity,
                    'scopes': scopes, 'times': times}
            results.append({'id': text_id, 'unit_id': unit_id, 'case': case, 'template': template,
                'text_sha256': digest(exact_text.encode('utf-8')), 'fidelity_established': true_fidelity,
                'machine_audit': None if template else machine,
                'prose_audit_validation': None if template else prose,
                'rating_validation': validation, 'rating_transport_complete': bool(rating_received and rating_received.get('transport_complete')),
                'actor_text_available': text_available, 'reference_measurement_valid_before_actor_text_gate': annotation_valid,
                'reported_composite_labels_before_validation': {name: annotation['composites'][name].get('status')
                    for name in ('COD', 'CA') if type(annotation) is dict
                    and type(annotation.get('composites')) is dict
                    and type(annotation['composites'].get(name)) is dict},
                'reported_rater_limitations': annotation.get('limitations') if type(annotation) is dict else None,
                'features': feature_statuses, 'auxiliary': auxiliary_statuses, 'endpoints': endpoints})
    pairs = []
    for pair in pair_index:
        first, second = pair['first_id'], pair['second_id']
        require(first in by_id and second in by_id, 'causal pair outside fixed inventory')
        a, b = inventory['packets'][first]['content'], inventory['packets'][second]['content']
        panel_rows = []
        source_panels = inventory['provenance'][first]['panels']
        for provenance in source_panels:
            panel = provenance['panel']
            left_panel = next(p for p in a['panels'] if p['panel'] == panel)
            right_panel = next(p for p in b['panels'] if p['panel'] == panel)
            expected_a = [p for p in required_propositions(a) if p['panel'] == panel]
            expected_b = [p for p in required_propositions(b) if p['panel'] == panel]
            def bound_prose(identifier):
                audit = by_id[identifier]['prose_audit']
                parsed = audit.get('parsed') if audit else None
                if type(parsed) is not dict or type(parsed.get('claims')) is not list:
                    return []
                return [c for c in parsed['claims'] if type(c) is dict
                        and type(c.get('bindings')) is list and any(
                            type(binding) is dict and binding.get('panel') == panel for binding in c['bindings'])]
            panel_rows.append({'panel': panel, 'source_key': provenance['source_key'],
                'public_panel_identical': encoded(left_panel) == encoded(right_panel),
                'expected_peak_sets_identical': expected_a == expected_b,
                'listed_claim_changes': _claim_changes(by_id[first]['report'], by_id[second]['report'], panel),
                'source_bound_prose_claims_first': bound_prose(first),
                'source_bound_prose_claims_second': bound_prose(second),
                'prose_wording_difference_is_not_automatically_a_semantic_change': True})
        pairs.append({**pair, 'input_identical': encoded(a) == encoded(b),
            'text_identical': by_id[first]['text'] == by_id[second]['text'],
            'both_fidelity_established': fidelity[first] and fidelity[second],
            'panel_diagnostics': panel_rows,
            'wording_change_is_not_semantic_change': True})
    totals = Counter()
    arm_totals = {'narrator': Counter(), 'template': Counter()}
    for value in groups.values():
        totals.update(value)
    for key, value in groups.items():
        arm_totals[key.rsplit('|', 1)[1]].update(value)
    rates = {}
    for key, value in groups.items():
        denominator = value['text_opportunities']
        rates[key] = {'text_denominator': denominator,
            'fidelity_established_rate': value['fidelity_established'] / denominator,
            'valid_rating_rate': value['valid_rating'] / denominator,
            'correct_required_peak_atom_rate': value['correct_required_peak_atoms'] / value['required_peak_atoms']}
        for endpoint in ('COD', 'CA'):
            for name in ('lower', 'upper', 'faithful_lower', 'faithful_upper'):
                rates[key][endpoint + '_' + name + '_rate'] = value[endpoint + '_' + name] / denominator
    required = sum(row['machine_audit']['required_peak_count'] for row in results if not row['template'])
    return {'schema': 'attention_relational_report_v1_complete_summary',
        'opportunities': {'archive_cases': 12, 'narrator': 552, 'templates': 552, 'ratings': 1104,
            'prose_audit_positions': 552, 'required_narrator_peak_atoms': required,
            'required_template_peak_atoms': 25440, 'causal_pairs': len(pairs)},
        'totals': dict(totals), 'arm_totals': {k: dict(v) for k, v in arm_totals.items()},
        'groups': {k: dict(v) for k, v in groups.items()}, 'group_rates': rates,
        'causal_pair_denominators': dict(Counter(p['kind'] for p in pairs)),
        'all_text_results': results, 'all_causal_pairs': pairs,
        'missing_report_ids': [i for i in inventory['request_ids'] if not reports.get(i, {}).get('transport_complete')],
        'reference_sha256': REFERENCE_HASH,
        'interpretation': {'v1_manner_promotion': False, 'experience_established': False,
            'numeric_theory_success_gate': None, 'ordinary_parity_is_not_logical_disqualification': True,
            'W_is_richer_oracle_diagnostic': True, 'sensor_has_no_forecasts': True,
            'one_rater_does_not_measure_agreement': True,
            'fidelity_semantic_audit_and_reference_rating_are_fallible': True,
            'generic_input_dependent_narration_alone_does_not_establish_reference_correspondence': True}}


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(encoded(value) + b'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare-report', 'prepare-audit', 'prepare-rate', 'collect', 'summarize'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--requests-output')
    parser.add_argument('--lineage-output')
    for name in ('exports', 'reports', 'audits', 'ratings', 'requests', 'completion', 'stage_manifest', 'lineage'):
        parser.add_argument('--' + name.replace('_', '-'))
        parser.add_argument('--' + name.replace('_', '-') + '-sha256')
    parser.add_argument('--response-root')
    args = parser.parse_args()
    require(not Path(args.output).exists(), 'output already exists; refusing replacement')
    dependencies = {}
    def load(name):
        path, sha = getattr(args, name), getattr(args, name + '_sha256')
        require(path is not None and sha is not None, 'required bound input missing: ' + name)
        value = bound_json(path, sha)
        dependencies[name] = {'path': path, 'sha256': sha}
        return value
    if args.stage == 'collect':
        requests = load('requests')
        requests = requests['requests'] if type(requests) is dict else requests
        require(args.completion and args.completion_sha256 and args.response_root,
                'collection needs bound completion and reviewed response root')
        stage_manifest = load('stage_manifest')
        lineage = load('lineage')
        require(str(Path(args.requests)) == stage_manifest['requests_path']
                and str(Path(args.lineage)) == stage_manifest['lineage_path'], 'stage input path differs')
        binding = build_collection_binding(stage_manifest, args.stage_manifest_sha256,
            requests, args.requests_sha256, lineage, args.lineage_sha256)
        result = collect_first_responses(args.completion, args.completion_sha256,
            list(_index(requests)), args.response_root, stage_binding=binding)
        dependencies['completion'] = {'path': args.completion, 'sha256': args.completion_sha256}
    else:
        exports = load('exports')
        inventory = validate_export(exports)
        if args.stage == 'prepare-report':
            result = {'requests': exports['requests'], 'request_ids': inventory['request_ids'],
                      'pair_index': build_pair_index(exports['provenance'])}
            first_reports = None
        elif args.stage == 'prepare-audit':
            reports = load('reports')
            _validate_collection_lineage(reports, 'report', exports['requests'], exports)
            requests = build_prose_audit_requests(reports, exports['packets'])
            result = {'requests': requests, 'opportunity_ids': inventory['request_ids'],
                      'unattemptable_ids': [i for i in inventory['request_ids'] if i not in _index(requests)]}
            first_reports = reports
        elif args.stage == 'prepare-rate':
            first_reports = load('reports')
            _validate_collection_lineage(first_reports, 'report', exports['requests'], exports)
            result = {'requests': build_blind_rater_requests(first_reports, exports['templates']),
                      'text_opportunities': 1104}
        else:
            result = summarize_reporting_results(load('reports'), None, load('audits'), load('ratings'),
                build_pair_index(exports['provenance']), exports)
        if args.stage.startswith('prepare-'):
            require(args.requests_output and args.lineage_output, 'prepare needs exclusive requests and lineage paths')
            require(not Path(args.requests_output).exists() and not Path(args.lineage_output).exists(),
                    'first prepared requests/lineage already exist')
            lineage = build_stage_lineage(args.stage.removeprefix('prepare-'), exports,
                {v['path']: v['sha256'] for v in dependencies.values()}, first_reports)
            exclusive_json(args.requests_output, result['requests'])
            exclusive_json(args.lineage_output, lineage)
            result['requests_artifact'] = {'path': args.requests_output,
                'sha256': digest(Path(args.requests_output).read_bytes())}
            result['lineage_artifact'] = {'path': args.lineage_output,
                'sha256': digest(Path(args.lineage_output).read_bytes())}
    result['preparation_dependencies'] = dependencies
    result['pipeline_source_sha256'] = digest(Path(__file__).read_bytes())
    result['no_api_or_model_calls'] = True
    exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
