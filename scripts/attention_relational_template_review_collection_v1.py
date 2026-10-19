"""Inert retention of the frozen template review's exclusive first judgments.

This module neither commissions reviewers nor computes a semantic endpoint.
Structural validity is not semantic correctness. Reviewer assessments, source
meanings, compatible scope/time claims and reasons remain asserted judgments.

Root receipt contracts (JSON, hash-bound CLI inputs):
* launch: schema VERSION+'_launch', manifest_sha256, protocol_sha256,
  output_schema_sha256, contexts. Each of the 46 contexts has reviewer_id,
  independent_context_id, chunk_number, chunk_sha256, first_artifact_path,
  fresh_context:true, first_pass:true, peer_judgments_supplied:false. Context IDs
  and first paths are globally unique; both named streams cover all 23 chunks.
* outcomes: schema VERSION+'_outcomes', launch_receipt_sha256,
  manifest_sha256, contexts. Each repeats reviewer/context/chunk identity and
  chunk hash and has first_artifact (null or {path,sha256}) and
  terminal_attestation (null or {independent_context_id,chunk_sha256,
  observed_terminal:true,observation_kind:'completed'|'failed'|'interrupted',
  observation_artifact:{path,sha256}}). Observation artifacts are nonempty exact
  retained root observations, not a descriptor of an unfinished handle.

A matching, actually retained first file is sufficient to collect that context,
including malformed/incomplete first files. Without it a bound actual terminal
observation is necessary to retain missing first output. Unsaved nonterminal
contexts make collection not ready and produce no output. These receipts assert
actual root observations; this code checks their bindings, not tool liveness or
reviewer independence. No invented provider identity or approval is accepted.

Production re-runs the existing preparer's actual terminal/lineage verification
before loading prepared chunks or any review output. It recomputes preparation
and checks all packet/source bindings. Pure helpers accept fictional inputs and
never constitute production terminal verification. First bytes (base64), parsed
values, parse/schema/contract errors and every planned position survive intact.
No replacement, gap filling, valid-row salvage or semantic normalization occurs.
If a terminal context's first file differs from its receipt or cannot be read,
retain the discrepancy and any readable observed bytes; its entire chunk stays
unestablished. A mismatched/unreadable file never establishes nonterminal readiness.
"""
import argparse
import base64
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import attention_relational_template_review_v1 as preparation
import attention_relational_reporting_pipeline_v1 as pipeline

VERSION = 'attention_relational_template_review_collection_v1'


def _check(condition, message):
    pipeline.require(condition, message)


def _same(left, right):
    """JSON identity, including bool != number and integer != float."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _key(path):
    return pipeline.encoded(path)


def _resolve(value, path):
    for part in path:
        if type(value) is dict and type(part) is str and part in value:
            value = value[part]
        elif type(value) is list and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError('value_path does not resolve exactly')
    return value


def schema_errors(value, schema, path=()):
    """Validate the exact frozen schema vocabulary, with strict JSON types."""
    errors = []
    def issue(message):
        errors.append(dict(path=list(path), message=message))
    if not schema:
        return errors
    if 'anyOf' in schema:
        if not any(not schema_errors(value, option, path) for option in schema['anyOf']):
            issue('no anyOf branch matches')
        return errors
    kinds = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'integer': int}
    if 'type' in schema and type(value) is not kinds[schema['type']]:
        issue('expected ' + schema['type'])
        return errors
    if 'enum' in schema and not any(_same(value, v) for v in schema['enum']):
        issue('value outside frozen enum')
    if 'const' in schema and not _same(value, schema['const']):
        issue('value differs from frozen const')
    if type(value) is dict:
        properties = schema.get('properties', {})
        for name in schema.get('required', []):
            if name not in value:
                errors.append(dict(path=list(path) + [name], message='required field missing'))
        for name, entry in value.items():
            if name in properties:
                errors.extend(schema_errors(entry, properties[name], path + (name,)))
            elif schema.get('additionalProperties') is False:
                errors.append(dict(path=list(path) + [name], message='additional field forbidden'))
    if type(value) is list:
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', float('inf')):
            issue('array length outside frozen limits')
        for i, entry in enumerate(value):
            errors.extend(schema_errors(entry, schema.get('items', {}), path + (i,)))
    if type(value) is int and value < schema.get('minimum', value):
        issue('integer below frozen minimum')
    return errors


def frozen_output_schema():
    return pipeline.bound_json(pipeline.ROOT / preparation.OUTPUT_SCHEMA, preparation.OUTPUT_SCHEMA_SHA)


def _inventory(items, field, expected, label):
    paths = [_key(item[field]) for item in items]
    _check(len(paths) == len(set(paths)) and set(paths) == {_key(p) for p in expected},
           label + ' must inventory every original path exactly once')
    return {_key(item[field]): item for item in items}


def validate_record(judgment, record, schema):
    """Pure contract check after schema validation; never interprets semantics.

    Returns precise property provenance without expanding containing origin
    objects into claims about their other fields. Paths outside quote overlaps
    are legal with a reviewer reason; that reason's adequacy remains semantic.
    """
    shape_errors = schema_errors(judgment, schema)
    if shape_errors:
        return dict(valid=False, errors=shape_errors, property_references=[])
    errors, properties = [], []
    try:
        for name in ('id', 'unit_id', 'text_sha256', 'reference_established'):
            _check(_same(judgment[name], record[name]), 'original ' + name + ' differs')
        _check(bool(judgment['notes'].strip()), 'record requires a reason/diagnostic')
        if not record['reference_established']:
            _check(not judgment['component_reviews'] and not judgment['evidence_reviews'] and not judgment['binding_reviews'],
                   'unestablished reference requires empty semantic inventories; no label salvage')
            _check('unestablished' in judgment['notes'].lower(), 'unestablished reference must be recorded as unestablished')
            return dict(valid=True, errors=[], property_references=[])
        _check(len(record['components']) == 12, 'prepared record lacks all12 original components')
        components = _inventory(judgment['component_reviews'], 'component_path', [c['path'] for c in record['components']], 'components')
        evidence = _inventory(judgment['evidence_reviews'], 'evidence_path', [e['path'] for e in record['evidence']], 'evidence')
        bindings = _inventory(judgment['binding_reviews'], 'binding_path', [b['path'] for b in record['bindings']], 'bindings')
        units = {u['unit_id']: u for u in record['property_table']}
        _check(len(units) == len(record['property_table']), 'duplicate prepared property unit')
        by_evidence_id = {e['evidence_id']: e for e in record['evidence']}
        _check(len(by_evidence_id) == len(record['evidence']), 'duplicate prepared evidence identity')
        for component in record['components']:
            item = components[_key(component['path'])]
            _check(item['original_status'] == component['recorded_status'], 'original component status changed')
            _check(bool(item['notes'].strip()), 'component requires reviewer reasoning')
        for original in record['evidence']:
            item = evidence[_key(original['path'])]
            _check(bool(item['notes'].strip()), 'evidence requires reviewer reasoning')
            overlap_ids = {g['unit_id'] for g in original['geometry']['intersections']}
            for position, ref in enumerate(item['properties']):
                _check(ref['unit_id'] in units, 'property must name supplied construction unit')
                unit = units[ref['unit_id']]
                actual = _resolve(unit['value'], ref['value_path'])
                _check(_same(actual, ref['value']), 'property value/type differs from exact relative value_path')
                _check(bool(ref['reason'].strip()), 'property requires explicit relational/binding reason, including outside overlap')
                packet_path = unit.get('packet_path')
                properties.append(dict(evidence_path=deepcopy(original['path']), property_position=position,
                    unit_id=ref['unit_id'], value_path=deepcopy(ref['value_path']), value=deepcopy(actual),
                    semantic_role=ref['semantic_role'], reason=ref['reason'],
                    packet_value_path=None if packet_path is None else deepcopy(packet_path + ref['value_path']),
                    unit_kind=unit['kind'], unit_start=unit['start'], unit_end=unit['end'],
                    overlaps_quoted_evidence=ref['unit_id'] in overlap_ids,
                    evidence_geometry=deepcopy(original['geometry']),
                    context=deepcopy(unit.get('context')), source_provenance=deepcopy(unit.get('source_resolution')),
                    semantic_relation_asserted_by_reviewer_only=True))
        for original in record['bindings']:
            item = bindings[_key(original['path'])]
            _check(bool(item['notes'].strip()), 'binding requires scope/time/content compatibility reasoning')
            required = original['required_features']
            features = [w['feature'] for w in item['feature_witnesses']]
            _check(len(features) == len(set(features)) and set(features) == set(required),
                   'binding must name each required original feature exactly once')
            for witness in item['feature_witnesses']:
                candidate_ids = original['candidate_feature_evidence_ids'][witness['feature']]
                candidates = {_key(by_evidence_id[e]['path']) for e in candidate_ids}
                paths = [_key(p) for p in witness['evidence_paths']]
                _check(len(paths) == len(set(paths)) and set(paths) <= candidates,
                       'witness evidence does not belong to required original feature')
                if item['assessment'] == 'supported':
                    _check(bool(paths), 'supported binding requires recorded feature witnesses')
            own = [by_evidence_id[e]['path'] for e in original['own_evidence_ids']]
            paths = [_key(p) for p in item['binding_evidence_paths']]
            _check(len(paths) == len(set(paths)) and set(paths) == {_key(p) for p in own},
                   'binding evidence must name its own recorded evidence paths exactly once')
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc))
    return dict(valid=not errors, errors=errors, property_references=properties if not errors else [])


def validate_chunk(raw, chunk, expected, schema, first_artifact_errors=None):
    """Pure whole-chunk check; retains exact first bytes and parsed content."""
    retained = dict(first_bytes_base64=None if raw is None else base64.b64encode(raw).decode('ascii'),
        first_bytes_sha256=None if raw is None else pipeline.digest(raw), parsed_first=None,
        parse_errors=[], schema_errors=[], contract_errors=deepcopy(first_artifact_errors or []),
        record_checks=[], structurally_valid=False)
    if raw is None:
        retained['parse_errors'].append('missing exclusive first artifact')
        return retained
    try:
        value = pipeline.decoded(raw)
        retained['parsed_first'] = value
    except (ValueError, UnicodeError) as exc:
        retained['parse_errors'].append(str(exc))
        return retained
    retained['schema_errors'] = schema_errors(value, schema)
    if retained['schema_errors']:
        return retained
    try:
        for name in ('reviewer_id', 'independent_context_id', 'manifest_sha256', 'chunk_sha256', 'protocol_sha256', 'output_schema_sha256'):
            _check(_same(value[name], expected[name]), 'first chunk ' + name + ' binding differs')
        expected_ids = [r['id'] for r in chunk['records']]
        ids = [r['id'] for r in value['records']]
        _check(len(set(ids)) == len(ids) and set(ids) == set(expected_ids), 'first chunk must retain all24 expected identities once')
        judgments = {r['id']: r for r in value['records']}
        for original in chunk['records']:
            check = validate_record(judgments[original['id']], original, schema['properties']['records']['items'])
            retained['record_checks'].append(dict(id=original['id'], **check))
        _check(all(c['valid'] for c in retained['record_checks']), 'record contract failed; entire24-position chunk unestablished')
        retained['structurally_valid'] = not retained['contract_errors']
    except (ValueError, KeyError, TypeError) as exc:
        retained['contract_errors'].append(str(exc))
    if not retained['structurally_valid']:
        # Parsed originals survive above; derived property inventories cannot
        # be released from individually well-formed rows in a failed chunk.
        for check in retained['record_checks']:
            check['property_references'] = []
            check['diagnostic_only_whole_chunk_unestablished'] = True
    return retained


def _closed(value, fields, name):
    _check(type(value) is dict and set(value) == set(fields), name + ' receipt fields differ')


def validate_manifest(manifest, chunks):
    """Pure complete opportunity/binding check (production also recomputes it)."""
    _check(manifest['schema'] == preparation.VERSION + '_manifest'
        and manifest['reviewer_streams'] == ['reviewer_0', 'reviewer_1']
        and manifest['template_opportunities'] == manifest['narrator_opportunities'] == 552
        and manifest['pair_opportunities'] == len(manifest['pairs']) == 684
        and len(manifest['narrator_availability']) == 552
        and len(chunks) == len(manifest['chunks']) == 23, 'complete552x2/684 manifest required')
    ids = [r['id'] for c in chunks for r in c['records']]
    _check(len(ids) == len(set(ids)) == 552 and ids == manifest['ordered_template_ids']
        and manifest['order_salt'] == preparation.ORDER_SALT
        and ids == sorted(ids, key=lambda i: (pipeline.digest((preparation.ORDER_SALT + i).encode('utf-8')), i)),
        'complete frozen ordered552 template identities required')
    _check(len({r['id'] for r in manifest['narrator_availability']}) == 552, 'complete distinct552 narrator availability required')
    for number, (ref, chunk) in enumerate(zip(manifest['chunks'], chunks)):
        _check(chunk['schema'] == preparation.VERSION + '_chunk' and type(chunk['chunk_number']) is int
            and chunk['chunk_number'] == number and ref['positions'] == len(chunk['records']) == 24
            and ref['path'] == f'chunk_{number:02d}.json'
            and ref['ordered_template_ids'] == chunk['ordered_template_ids'] == [r['id'] for r in chunk['records']]
            and ref['sha256'] == pipeline.digest(pipeline.encoded(chunk) + b'\n'), 'exact prepared chunk binding differs')
        for name in ('review_protocol', 'frozen_reference', 'review_output_schema'):
            _check(chunk[name] == manifest[name], 'prepared chunk frozen dependency differs')
    _check(manifest['review_protocol'] == preparation.descriptor(preparation.PROTOCOL, preparation.PROTOCOL_SHA)
        and manifest['review_output_schema'] == preparation.descriptor(preparation.OUTPUT_SCHEMA, preparation.OUTPUT_SCHEMA_SHA)
        and manifest['frozen_reference'] == preparation.descriptor(pipeline.REFERENCE, pipeline.REFERENCE_HASH), 'frozen review dependencies differ')


def validate_receipts(manifest, manifest_sha, launch, launch_sha, outcomes):
    """Check 46 bound actual root receipt entries, without pretending liveness."""
    _closed(launch, ('schema', 'manifest_sha256', 'protocol_sha256', 'output_schema_sha256', 'contexts'), 'launch')
    _closed(outcomes, ('schema', 'launch_receipt_sha256', 'manifest_sha256', 'contexts'), 'outcomes')
    _check(launch['schema'] == VERSION + '_launch' and outcomes['schema'] == VERSION + '_outcomes'
        and launch['manifest_sha256'] == outcomes['manifest_sha256'] == manifest_sha
        and outcomes['launch_receipt_sha256'] == launch_sha
        and launch['protocol_sha256'] == preparation.PROTOCOL_SHA
        and launch['output_schema_sha256'] == preparation.OUTPUT_SCHEMA_SHA, 'root receipt dependency binding differs')
    _check(type(launch['contexts']) is list and type(outcomes['contexts']) is list
        and len(launch['contexts']) == len(outcomes['contexts']) == 46, 'all46 root launch/outcome contexts required')
    planned, seen_contexts, seen_paths = {}, set(), set()
    for entry in launch['contexts']:
        _closed(entry, ('reviewer_id', 'independent_context_id', 'chunk_number', 'chunk_sha256', 'first_artifact_path',
                        'fresh_context', 'first_pass', 'peer_judgments_supplied'), 'launch context')
        stream, number, context = entry['reviewer_id'], entry['chunk_number'], entry['independent_context_id']
        _check(stream in manifest['reviewer_streams'] and type(number) is int and 0 <= number < 23
            and type(context) is str and bool(context.strip()) and context not in seen_contexts
            and type(entry['first_artifact_path']) is str and bool(entry['first_artifact_path'])
            and entry['fresh_context'] is True and entry['first_pass'] is True and entry['peer_judgments_supplied'] is False
            and entry['chunk_sha256'] == manifest['chunks'][number]['sha256'], 'fresh exclusive launch/chunk contract differs')
        path = str(Path(entry['first_artifact_path']).resolve())
        _check(path not in seen_paths and (stream, number) not in planned, 'duplicate first artifact/launch position')
        seen_contexts.add(context)
        seen_paths.add(path)
        planned[stream, number] = entry
    _check(set(planned) == {(s, n) for s in manifest['reviewer_streams'] for n in range(23)}, 'both streams need all23 chunks')
    retained = {}
    for entry in outcomes['contexts']:
        _closed(entry, ('reviewer_id', 'independent_context_id', 'chunk_number', 'chunk_sha256', 'first_artifact', 'terminal_attestation'), 'outcome context')
        key = entry['reviewer_id'], entry['chunk_number']
        _check(key in planned and key not in retained, 'duplicate/unplanned outcome context')
        first = planned[key]
        _check(all(_same(entry[k], first[k]) for k in ('reviewer_id', 'independent_context_id', 'chunk_number', 'chunk_sha256')), 'outcome does not bind actual launched context')
        if entry['first_artifact'] is not None:
            _closed(entry['first_artifact'], ('path', 'sha256'), 'first artifact')
            _check(Path(entry['first_artifact']['path']).resolve() == Path(first['first_artifact_path']).resolve(), 'first artifact path differs from launch reservation')
        terminal = entry['terminal_attestation']
        if terminal is not None:
            _closed(terminal, ('independent_context_id', 'chunk_sha256', 'observed_terminal', 'observation_kind', 'observation_artifact'), 'terminal attestation')
            _check(terminal['independent_context_id'] == first['independent_context_id']
                and terminal['chunk_sha256'] == first['chunk_sha256'] and terminal['observed_terminal'] is True
                and terminal['observation_kind'] in ('completed', 'failed', 'interrupted'), 'actual observed terminal context required')
            _closed(terminal['observation_artifact'], ('path', 'sha256'), 'terminal observation')
        retained[key] = entry
    return planned, retained


class ArtifactBindingError(ValueError):
    """A readable first artifact differs from its unchanged declared binding."""

    def __init__(self, message, observed_bytes):
        super().__init__(message)
        self.observed_bytes = observed_bytes


def _bound_bytes(ref):
    _check(type(ref['path']) is str and type(ref['sha256']) is str and len(ref['sha256']) == 64, 'artifact descriptor differs')
    raw = Path(ref['path']).read_bytes()
    if pipeline.digest(raw) != ref['sha256']:
        raise ArtifactBindingError('retained first/observation SHA-256 mismatch', raw)
    return raw


def collect(manifest, chunks, manifest_sha, launch, launch_sha, outcomes, schema, read_bound=_bound_bytes):
    """Retain all first contexts; IO reader is injectable for fictional tests."""
    validate_manifest(manifest, chunks)
    planned, receipts = validate_receipts(manifest, manifest_sha, launch, launch_sha, outcomes)
    collected, rows = [], []
    for stream in manifest['reviewer_streams']:
        for number, chunk in enumerate(chunks):
            first, outcome = planned[stream, number], receipts[stream, number]
            raw, observation, artifact_errors = None, None, []
            if outcome['first_artifact'] is not None:
                try:
                    raw = read_bound(outcome['first_artifact'])
                    # Recheck injected readers as well as the production reader.
                    if pipeline.digest(raw) != outcome['first_artifact']['sha256']:
                        artifact_errors.append('first artifact SHA-256 differs from unchanged outcome descriptor')
                except ArtifactBindingError as exc:
                    raw = exc.observed_bytes
                    artifact_errors.append(type(exc).__name__ + ': ' + str(exc))
                except (OSError, ValueError) as exc:
                    artifact_errors.append(type(exc).__name__ + ': ' + str(exc))
            if outcome['terminal_attestation'] is not None:
                observation = read_bound(outcome['terminal_attestation']['observation_artifact'])
                _check(pipeline.digest(observation) == outcome['terminal_attestation']['observation_artifact']['sha256'],
                       'actual terminal observation SHA-256 mismatch')
                _check(bool(observation.strip()), 'terminal observation artifact must retain actual root observation')
            _check((raw is not None and not artifact_errors) or observation is not None,
                   'collection not ready: no matching retained first file or actual terminal observation for ' + first['independent_context_id'])
            expected = dict(reviewer_id=stream, independent_context_id=first['independent_context_id'],
                manifest_sha256=manifest_sha, chunk_sha256=first['chunk_sha256'],
                protocol_sha256=preparation.PROTOCOL_SHA, output_schema_sha256=preparation.OUTPUT_SCHEMA_SHA)
            result = validate_chunk(raw, chunk, expected, schema, first_artifact_errors=artifact_errors)
            result.update(expected, chunk_number=number, launch_context=deepcopy(first), outcome_context=deepcopy(outcome),
                terminal_observation_bytes_base64=None if observation is None else base64.b64encode(observation).decode('ascii'))
            collected.append(result)
            judgments = {j['id']: j for j in result['parsed_first']['records']} if result['structurally_valid'] else {}
            for record in chunk['records']:
                valid = result['structurally_valid']
                rows.append(dict(reviewer_id=stream, independent_context_id=first['independent_context_id'], chunk_number=number,
                    id=record['id'], unit_id=record['unit_id'], text_sha256=record['text_sha256'],
                    reference_established=record['reference_established'], reference_measurement=deepcopy(record['reference_measurement']),
                    review_state='retained_structurally_valid_judgment' if valid and record['reference_established'] else 'unestablished',
                    chunk_structurally_valid=valid, first_judgment=deepcopy(judgments.get(record['id'])),
                    semantic_correctness_established_by_code=False))
    return dict(schema=VERSION, manifest_sha256=manifest_sha, launch_receipt_sha256=launch_sha,
        template_opportunities=552, reviewer_opportunities=1104, narrator_opportunities=552,
        narrator_availability=deepcopy(manifest['narrator_availability']), pair_opportunities=684, pairs=deepcopy(manifest['pairs']),
        reference_lineage=deepcopy(manifest['reference_lineage']), chunks=collected, rows=rows,
        semantic_correctness_established_by_code=False, no_api_or_model_calls=True,
        no_replacement_or_gap_filling=True, whole_chunk_failure_policy=True)


def load_prepared_actual(manifest_path, manifest_sha):
    """Verify original V3 terminal before reading any prepared packet chunks."""
    # Only preparation metadata is opened first; no annotation/packet files yet.
    manifest = pipeline.bound_json(manifest_path, manifest_sha)
    dependencies = manifest.get('input_dependencies', {})
    names = ('index', 'preparation', 'collection', 'derived', 'execution_receipt', 'manifest')
    args = {}
    for name in names:
        ref = dependencies[name]
        args[name] = Path(ref['path'])
        args[name + '_sha256'] = ref['sha256']
    terminal = dependencies.get('terminal_receipt')
    args.update(terminal_receipt=None if terminal is None else Path(terminal['path']),
                terminal_receipt_sha256=None if terminal is None else terminal['sha256'])
    index, prepared, derived, verified = preparation.load_actual(SimpleNamespace(**args))
    expected, recomputed_chunks = preparation.assemble(index, prepared, derived)
    for name, value in expected.items():
        _check(_same(manifest.get(name), value), 'prepared manifest differs from exact actual recomputation: ' + name)
    _check(manifest.get('terminal_verification') == verified and manifest.get('no_api_or_model_calls') is True,
           'prepared manifest terminal verification differs')
    _check(manifest.get('source_sha256') == preparation.source_hashes(), 'prepared source hashes/runtime/preparation lineage changed')
    chunks = []
    root = Path(manifest_path).parent
    for ref, recomputed in zip(manifest['chunks'], recomputed_chunks):
        path = root / ref['path']
        _check(path.resolve().parent == root.resolve(), 'prepared chunk escapes manifest directory')
        chunk = pipeline.bound_json(path, ref['sha256'])
        _check(_same(chunk, recomputed), 'prepared chunk differs from exact recomputation')
        chunks.append(chunk)
    validate_manifest(manifest, chunks)
    return manifest, chunks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-manifest', 'launch-receipt', 'outcome-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    _check(not args.output.exists(), 'refusing replacement of exclusive first collection')
    manifest, chunks = load_prepared_actual(args.prepared_manifest, args.prepared_manifest_sha256)
    schema = frozen_output_schema()
    launch = pipeline.bound_json(args.launch_receipt, args.launch_receipt_sha256)
    outcomes = pipeline.bound_json(args.outcome_receipt, args.outcome_receipt_sha256)
    result = collect(manifest, chunks, args.prepared_manifest_sha256, launch, args.launch_receipt_sha256, outcomes, schema)
    result['input_dependencies'] = {name: preparation.descriptor(getattr(args, name), getattr(args, name + '_sha256'))
        for name in ('prepared_manifest', 'launch_receipt', 'outcome_receipt')}
    result['source_sha256'] = {str(Path(__file__).resolve().relative_to(pipeline.ROOT)): pipeline.digest(Path(__file__).read_bytes()),
        'scripts/attention_relational_template_review_v1.py': pipeline.digest(Path(preparation.__file__).read_bytes())}
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
