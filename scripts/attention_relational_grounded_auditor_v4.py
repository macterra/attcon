"""Inert V4 addressing/coverage mechanics, not semantic qualification or a runner.

Exact evidence locations and legal contexts do not establish assertion discovery,
correct support judgments, actor availability, or report fidelity. No old result
is repaired or rescored. Qualification and actual runtime lineage remain separate.
"""
from copy import deepcopy
from pathlib import Path
import re

import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline


RAW_SCHEMA = 'attention_relational_grounded_auditor_v4_annotation'
INTERFACE_SCHEMA = 'attention_relational_grounded_auditor_v4_interface'
INSTRUCTIONS = Path(__file__).resolve().parents[1] / 'audits/attention_relational_report_v1/grounded_auditor_v4/instructions.txt'
SPAN_RULE = 'Split immediately after CRLF/CR/LF or ASCII .!? followed by Python Unicode whitespace; retain every character.'
_BOUNDARY = re.compile(r'\r\n|[\r\n]|[.!?](?=\s)')


def _hash(value):
    return pipeline.digest(pipeline.encoded(value))


def partition_spans(text):
    pipeline.require(type(text) is str, 'report text must be an exact string')
    ends = [match.end() for match in _BOUNDARY.finditer(text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    spans, start = [], 0
    for end in ends:
        if start < end:
            spans.append(dict(span_id=f's{len(spans):04d}', start=start, end=end, text=text[start:end]))
        start = end
    pipeline.require(''.join(span['text'] for span in spans) == text, 'span partition lost text')
    return spans


def source_contexts(packet):
    """Legal actual metadata, origins, roles and relations, in a fixed traversal.

Relation IDs explicitly carry both role operands. Cross-context assertions must
cite each applicable actual ID; no record-level citation expands to all nodes.
"""
    pipeline.require(packet.get('schema') == exporter.PACK_SCHEMA, 'neutral numerical packet required')
    contexts, addresses = [], set()

    def add(value):
        identifier = f'c{len(contexts):04d}'
        contexts.append(dict(context_id=identifier, **value))
        return identifier

    add(dict(kind='record_metadata', packet_path=[],
             metadata=deepcopy({k: v for k, v in packet.items() if k != 'panels'})))
    panels = set()
    for pi, panel in enumerate(packet['panels']):
        name = panel['panel']
        pipeline.require(name in ('p0', 'p1') and name not in panels, 'invalid/duplicate supplied panel')
        panels.add(name)
        candidates = panel['candidates']
        commands = sorted(candidates) if candidates is not None else []
        pipeline.require(all(c in exporter.COMMANDS for c in commands), 'invalid supplied command')
        add(dict(kind='record_metadata', packet_path=['panels', pi],
                 metadata=dict(panel=name, candidate_commands=commands if candidates is not None else None)))
        views = [('current', None, panel['current'], ['panels', pi, 'current'])]
        views += [('candidate', command, candidates[command], ['panels', pi, 'candidates', command]) for command in commands]
        for view, command, nodes, path in views:
            for ni, node in enumerate(nodes):
                scope = dict(panel=name, view=view, command=command, node=node['node'])
                address = (name, view, command, node['node'])
                pipeline.require(node['node'] in exporter.NODES and address not in addresses, 'invalid/duplicate supplied node')
                addresses.add(address)
                pipeline.require(set(node['distributions']) == set(exporter.ROLES), 'both supplied roles required')
                origin = deepcopy(node['origin'])
                node_path = path + [ni]
                add(dict(kind='node_origin', **scope, packet_path=node_path + ['origin'], origin=origin))
                operands, summaries = [], {}
                for role in exporter.ROLES:
                    summary = exporter.localization_summary(node['distributions'][role])
                    summaries[role] = summary
                    operands.append(add(dict(kind='role_distribution', **scope, role=role,
                        packet_path=node_path + ['distributions', role], origin=deepcopy(origin),
                        calculations=summary,
                        supplied_derived=deepcopy(node.get('derived', {}).get(role)))))
                add(dict(kind='red_blue_peak_relation', **scope, origin=deepcopy(origin),
                    operands=operands, calculations=exporter.peak_relation(summaries['red']['peaks'], summaries['blue']['peaks']),
                    supplied_relation=deepcopy(node.get('red_blue_peak_relation'))))
    return contexts


def response_schema(context_ids=None, span_ids=None):
    """Static schema by default, compatible with a shared generic-runner schema.

Optional ID enums are a prospective per-request alternative; using different
schemas at runtime requires an honestly versioned runtime. Code always checks
the actual legal inventory, including when an empty enum uses a string type.
"""
    def ids(values):
        if values is None:
            return {'type': 'string'}
        pipeline.require(type(values) in (list, tuple) and all(type(v) is str and v for v in values)
                         and len(set(values)) == len(values), 'invalid schema ID inventory')
        return {'type': 'string', 'enum': list(values)} if values else {'type': 'string'}
    obj, array = pipeline._object, pipeline._array
    string = {'type': 'string'}
    return obj(dict(schema=pipeline._enum(RAW_SCHEMA), unit_id=string,
        assertions=array(obj(dict(assertion_id=string, paraphrase=string, span_ids=array(ids(span_ids)),
            source_context_ids=array(ids(context_ids)), verdict=pipeline._enum('supported', 'contradicted', 'not_auditable'), reason=string))),
        coverage=array(obj(dict(span_id=ids(span_ids), assertion_ids=array(string),
            classification=pipeline._enum('assertive', 'nonassertive'), reason=string))),
        complete_assertion_inventory={'type': 'boolean'}, unresolved_ambiguity={'type': 'boolean'}))


def interface(unit_id, text, packet):
    pipeline.require(type(unit_id) is str and bool(unit_id), 'unit ID required')
    pipeline.require(type(text) is str, 'report text must be an exact string')
    return dict(schema=INTERFACE_SCHEMA, unit_id=unit_id, report_text=text,
        report_text_sha256=pipeline.digest(text.encode('utf-8')), packet=deepcopy(packet),
        packet_content_sha256=_hash(packet), span_rule=SPAN_RULE,
        spans=partition_spans(text), source_contexts=source_contexts(packet))


def build_request(identifier, text, packet):
    pipeline.require(type(identifier) is str and re.fullmatch(r'[A-Za-z0-9_-]+', identifier), 'safe request ID required')
    unit_id = pipeline.digest(('attention_relational_grounded_auditor_v4_unit:' + identifier).encode('utf-8'))[:24]
    value = interface(unit_id, text, packet)
    instruction = INSTRUCTIONS.read_text().rstrip('\n')
    return dict(id=identifier, input=instruction + '\n' + pipeline.encoded(value).decode('ascii'))


def derive_annotation(request, raw, *, expected_request_sha256):
    """Verify mechanical references and derive separate exact spans; no quotes guessed."""
    actual_request_sha = pipeline.runner_request_sha256(request)
    prefix, payload = request['input'].rsplit('\n', 1)
    value = pipeline.decoded(payload)
    rebuilt = build_request(request['id'], value['report_text'], value['packet'])
    pipeline.require(request == rebuilt and prefix == INSTRUCTIONS.read_text().rstrip('\n'), 'request/interface content differs')
    pipeline.require(actual_request_sha == expected_request_sha256, 'actual request SHA mismatch')
    expected = interface(value['unit_id'], value['report_text'], value['packet'])
    pipeline.require(value == expected, 'span/context/hash inventory differs')
    span_map = {s['span_id']: s for s in value['spans']}
    context_map = {c['context_id']: c for c in value['source_contexts']}
    errors, parsed_assertions, parsed_coverage = [], {}, {}

    def bad(message):
        errors.append(message)

    def nonempty(value):
        return type(value) is str and bool(value.strip())

    def references(values, legal, label, required=False):
        if type(values) is not list or any(type(v) is not str for v in values):
            bad(label + ': references must be a string list')
            return []
        if len(set(values)) != len(values) or any(v not in legal for v in values) or (required and not values):
            bad(label + ': illegal, duplicate, or missing reference')
        return values

    keys = {'schema', 'unit_id', 'assertions', 'coverage', 'complete_assertion_inventory', 'unresolved_ambiguity'}
    if type(raw) is not dict or set(raw) != keys or raw.get('schema') != RAW_SCHEMA or raw.get('unit_id') != value['unit_id']:
        bad('raw V4 schema/unit/keys differ')
        raw_value = {}
    else:
        raw_value = raw
    for key in ('complete_assertion_inventory', 'unresolved_ambiguity'):
        if type(raw_value.get(key)) is not bool:
            bad('invalid boolean: ' + key)
    assertions = raw_value.get('assertions')
    if type(assertions) is not list:
        bad('assertions must be a list')
        assertions = []
    for assertion in assertions:
        required_keys = {'assertion_id', 'paraphrase', 'span_ids', 'source_context_ids', 'verdict', 'reason'}
        if type(assertion) is not dict or set(assertion) != required_keys:
            bad('malformed assertion'); continue
        identifier = assertion['assertion_id']
        if not nonempty(identifier) or identifier in parsed_assertions:
            bad('invalid/duplicate assertion ID'); continue
        if not nonempty(assertion['paraphrase']) or not nonempty(assertion['reason']):
            bad('empty assertion paraphrase/reason')
        if assertion['verdict'] not in ('supported', 'contradicted', 'not_auditable'):
            bad('invalid support verdict')
        spans = references(assertion['span_ids'], span_map, identifier + ' spans', required=True)
        contexts = references(assertion['source_context_ids'], context_map, identifier + ' contexts',
                              required=assertion['verdict'] == 'supported')
        parsed_assertions[identifier] = dict(raw=assertion, spans=spans, contexts=contexts)
    coverage = raw_value.get('coverage')
    if type(coverage) is not list:
        bad('coverage must be a list')
        coverage = []
    for entry in coverage:
        if type(entry) is not dict or set(entry) != {'span_id', 'assertion_ids', 'classification', 'reason'}:
            bad('malformed coverage entry'); continue
        span_id = entry['span_id']
        if type(span_id) is not str or span_id not in span_map or span_id in parsed_coverage:
            bad('illegal/duplicate coverage span'); continue
        ids = references(entry['assertion_ids'], parsed_assertions, span_id + ' assertions')
        if (entry['classification'] not in ('assertive', 'nonassertive') or not nonempty(entry['reason'])
                or (entry['classification'] == 'assertive' and not ids)
                or (entry['classification'] == 'nonassertive' and ids)):
            bad('coverage classification/reason differs')
        parsed_coverage[span_id] = ids
    if set(parsed_coverage) != set(span_map):
        bad('coverage does not contain every span exactly once')
    for span_id, ids in parsed_coverage.items():
        cited = {identifier for identifier, assertion in parsed_assertions.items() if span_id in assertion['spans']}
        if set(ids) != cited:
            bad('assertion/span coverage links are not bidirectional: ' + span_id)
    derived = []
    if not errors:
        for assertion in parsed_assertions.values():
            selected = [context_map[c] for c in assertion['contexts']]
            operands = []
            for context in selected:
                if context['kind'] == 'role_distribution':
                    operands.append(context['context_id'])
                operands.extend(context.get('operands', []))
            operands = list(dict.fromkeys(operands))
            derived.append(dict(assertion=deepcopy(assertion['raw']),
                exact_evidence=[deepcopy(span_map[s]) for s in assertion['spans']],
                resolved_contexts=deepcopy(selected), operand_contexts=[deepcopy(context_map[c]) for c in operands]))
    return dict(schema='attention_relational_grounded_auditor_v4_derived_evidence',
        request_sha256=actual_request_sha, raw_annotation_content_sha256=_hash(raw),
        report_text_sha256=value['report_text_sha256'], packet_content_sha256=value['packet_content_sha256'],
        instructions_sha256=pipeline.digest(INSTRUCTIONS.read_bytes()), span_rule=SPAN_RULE,
        mechanical_valid=not errors, errors=errors, derived_assertions=derived if not errors else None,
        coverage=deepcopy(raw_value.get('coverage')), complete_assertion_inventory=raw_value.get('complete_assertion_inventory'),
        unresolved_ambiguity=raw_value.get('unresolved_ambiguity'),
        semantic_completeness_established=False, support_judgments_verified=False,
        actor_availability_assessed=False, actual_runtime_lineage_verified=False)
