"""Retain exclusive first supplementary narrator/source reviews; no semantic grading.

Receipt contract is the template collector's closed launch/outcome contract with
this VERSION, the narrator plan/schema hashes, and 2 * len(manifest.chunks)
contexts instead of 46. Both streams cover every candidate chunk (0..138); every
context has a unique ID and reserved first-artifact path. Launch is frozen before
review. Outcomes bind that launch and retain either the matching first artifact
or an actual terminal observation, or both. Terminal observations bind a nonempty
root observation artifact and the exact context/chunk; an observation timeout is
not terminal. Empty candidate inventory has empty launch/outcome lists and makes
no reviewer calls. Metadata checks cannot observe liveness or independence.

Malformed or incomplete first outputs invalidate their entire chunk, without
row salvage/replacement. Declared SHA mismatches/unreadable first files require
an actual bound terminal observation; retain readable observed bytes and the
unchanged declaration/error. Shared retention/type/path primitives are pinned.

The production loader recomputes the preparer's actual terminal/all47-source
verification and complete preparation before opening chunks or first reviews.
Schema/path/index checks do not establish semantic correctness, exhaustive raw
operands, ownership, arithmetic truth, property tracking, appearance or qualia.
Expert assertions and calculation flags remain assertions. All552 availability
rows and684 pending pair dispositions survive; narrator fidelity stays false.
"""
import argparse
import base64
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import attention_relational_narrator_source_closure_v1 as preparation
import attention_relational_reporting_pipeline_v1 as pipeline
import attention_relational_template_review_collection_v1 as shared

VERSION = 'attention_relational_narrator_source_closure_collection_v1'
SHARED_SHA = '091a6ee6691f4cb26d7a674002fe203b5c22ab682708267e23266f523b7920fd'
_check, _same, _resolve, _key, _closed = shared._check, shared._same, shared._resolve, shared._key, shared._closed
ArtifactBindingError, _bound_bytes = shared.ArtifactBindingError, shared._bound_bytes


def schema_errors(value, schema, path=()):
    """Closed schema vocabulary with strict JSON identity and explicit null."""
    errors = []
    def issue(message):
        errors.append(dict(path=list(path), message=message))
    if not schema:
        return errors
    if 'anyOf' in schema:
        if not any(not schema_errors(value, option, path) for option in schema['anyOf']):
            issue('no anyOf branch matches')
        return errors
    kinds = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'integer': int, 'null': type(None)}
    if 'type' in schema and type(value) is not kinds[schema['type']]:
        issue('expected ' + schema['type'])
        return errors
    if 'enum' in schema and not any(_same(value, v) for v in schema['enum']):
        issue('value outside frozen enum')
    if 'const' in schema and not _same(value, schema['const']):
        issue('value differs from frozen const')
    if type(value) is dict:
        for name in schema.get('required', []):
            if name not in value:
                errors.append(dict(path=list(path) + [name], message='required field missing'))
        for name, entry in value.items():
            if name in schema.get('properties', {}):
                errors.extend(schema_errors(entry, schema['properties'][name], path + (name,)))
            elif schema.get('additionalProperties') is False:
                errors.append(dict(path=list(path) + [name], message='additional field forbidden'))
    if type(value) is list:
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', float('inf')):
            issue('array length outside frozen limits')
        for i, entry in enumerate(value):
            errors.extend(schema_errors(entry, schema.get('items', {}), path + (i,)))
    if type(value) is int and value < schema.get('minimum', value):
        issue('integer below frozen minimum')
    return errors


def frozen_output_schema():
    return pipeline.bound_json(pipeline.ROOT / preparation.OUTPUT_SCHEMA, preparation.schema_sha())


def _paths(actual, expected, label):
    keys = [_key(p) for p in actual]
    _check(len(keys) == len(set(keys)) and set(keys) == {_key(p) for p in expected},
           label + ' must retain every original path once, including duplicate quotes at distinct positions')


def _indices(values, size, label):
    _check(len(values) == len(set(values)) and all(type(v) is int and 0 <= v < size for v in values),
           label + ' indices must name supplied operands exactly')


def validate_record(judgment, record, schema):
    errors = schema_errors(judgment, schema)
    if errors:
        return dict(valid=False, errors=errors, operand_references=[])
    refs = []
    try:
        _check(all(_same(judgment[n], record[n]) for n in ('id', 'unit_id', 'text_sha256')), 'original narrator identity differs')
        _check(bool(judgment['notes'].strip()), 'record requires diagnostic/reason')
        originals = record['candidate_bindings']
        _paths([b['binding_path'] for b in judgment['candidate_binding_reviews']], [b['binding_path'] for b in originals], 'candidate bindings')
        by_path = {_key(b['binding_path']): b for b in originals}
        for item in judgment['candidate_binding_reviews']:
            original = by_path[_key(item['binding_path'])]
            bound = original['original_binding']
            _check(item['original_status'] == original['original_status'], 'original composite status changed')
            _check(bool(item['notes'].strip()), 'binding requires explicit reason')
            features = item['feature_reviews']
            names = [f['feature'] for f in features]
            _check(len(names) == len(set(names)) and set(names) == set(original['required_features']), 'each required feature must be reviewed once')
            for feature in features:
                _check(feature['original_status'] == record['canonical_first_annotation']['features'][feature['feature']]['status'], 'original feature status changed')
                _paths(feature['evidence_paths'], original['feature_evidence_paths'][feature['feature']], 'required feature evidence')
                _check(bool(feature['notes'].strip()), 'feature requires reason')
            meaning = item['source_meaning']
            _paths(meaning['binding_evidence_paths'], original['binding_evidence_paths'], 'own binding evidence')
            for field, key in (('content_referent', 'referent'), ('attribute', 'attribute'), ('scope', 'scope'), ('time', 'time')):
                _check(meaning[field] == bound[key], 'original binding meaning identity/' + field + ' changed')
            _check(bool(meaning['notes'].strip()) and bool(item['source_ownership']['notes'].strip())
                and bool(item['tracking_support']['notes'].strip()), 'meaning/ownership/tracking require reasons')
            operands = item['operands']
            for i, operand in enumerate(operands):
                value = _resolve(record['complete_source_packet'], operand['packet_path'])
                _check(_same(value, operand['value']), 'exact source operand value/type differs')
                _check(bool(operand['reason'].strip()), 'source operand requires reason')
                resolved = {}
                for name, source in (('provenance_paths', record['complete_source_provenance']),
                                     ('origin_field_paths', record['complete_source_packet']),
                                     ('represented_time_paths', record['complete_source_packet'])):
                    paths = operand[name]
                    _check(len(paths) == len({_key(p) for p in paths}), 'duplicate provenance/origin/time path')
                    values = []
                    for path in paths:
                        field_value = _resolve(source, path)
                        _check(not isinstance(field_value, (dict, list)), 'provenance/origin/time must name individual fields, not containers')
                        if name == 'origin_field_paths':
                            _check('origin' in path, 'origin field must belong to a supplied origin')
                        values.append(dict(path=deepcopy(path), value=deepcopy(field_value)))
                    resolved[name] = values
                refs.append(dict(binding_path=deepcopy(item['binding_path']), operand_index=i,
                    packet_path=deepcopy(operand['packet_path']), value=deepcopy(value),
                    exact_resolved_metadata=resolved, semantic_role=operand['semantic_role'],
                    source_category_asserted_by_reviewer=operand['source_category'], semantic_correctness_established_by_code=False))
            for calc in item['calculations']:
                _indices(calc['operand_indices'], len(operands), 'calculation')
                _check(bool(calc['operator'].strip()) and bool(calc['expression'].strip()) and bool(calc['notes'].strip()),
                       'calculation requires named operator, expression and reason')
            ownership = item['source_ownership']
            for name in ('native_estimated_operand_indices', 'exporter_derived_operand_indices'):
                _indices(ownership[name], len(operands), 'ownership')
            identity = item['tracking_support']['property_identity']
            if identity is not None:
                _indices(identity['source_operand_indices'], len(operands), 'property identity')
                for field, key in (('content_referent', 'referent'), ('attribute', 'attribute'), ('scope', 'scope'), ('time', 'time')):
                    _check(identity[field] == bound[key], 'property identity changed original ' + field)
                _check(bool(identity['operator'].strip()), 'property identity requires an operator')
            if item['tracking_support']['eligible'] == 'yes':
                _check(identity is not None, 'tracking eligibility requires explicit property identity')
            for path in item['tracking_support']['unaffected_property_paths']:
                _resolve(record['complete_source_packet'], path)
            # Supporting the correct recorded absence/ambiguity is not an
            # explicit-positive witness. No nonempty-evidence fiction is added.
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc))
    return dict(valid=not errors, errors=errors, operand_references=refs if not errors else [])


def validate_chunk(raw, chunk, expected, schema, artifact_errors=None):
    retained = dict(first_bytes_base64=None if raw is None else base64.b64encode(raw).decode('ascii'),
        first_bytes_sha256=None if raw is None else pipeline.digest(raw), parsed_first=None,
        parse_errors=[], schema_errors=[], contract_errors=deepcopy(artifact_errors or []), record_checks=[], structurally_valid=False)
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
        for name in expected:
            _check(_same(value[name], expected[name]), 'first chunk ' + name + ' differs')
        ids = [r['id'] for r in value['records']]
        _check(len(ids) == len(set(ids)) and set(ids) == {r['id'] for r in chunk['records']}, 'whole first chunk must retain every narrator once')
        judgments = {r['id']: r for r in value['records']}
        for record in chunk['records']:
            retained['record_checks'].append(dict(id=record['id'], **validate_record(judgments[record['id']], record, schema['properties']['records']['items'])))
        _check(all(r['valid'] for r in retained['record_checks']), 'record contract failed; whole chunk unestablished')
        retained['structurally_valid'] = not retained['contract_errors']
    except (ValueError, KeyError, TypeError) as exc:
        retained['contract_errors'].append(str(exc))
    if not retained['structurally_valid']:
        for row in retained['record_checks']:
            row['operand_references'] = []
            row['diagnostic_only_whole_chunk_unestablished'] = True
    return retained


def validate_manifest(manifest, chunks):
    _check(manifest['schema'] == preparation.VERSION + '_manifest' and manifest['reviewer_streams'] == preparation.STREAMS
        and manifest['narrator_opportunities'] == len(manifest['narrator_availability']) == 552
        and manifest['pair_opportunities'] == len(manifest['pairs']) == 684
        and len(chunks) == len(manifest['chunks']) <= 138, 'complete552/684 candidate manifest required')
    rows = manifest['narrator_availability']
    ids = [r['id'] for r in rows]
    _check(len(set(ids)) == 552 and sum(r['actor_text_available'] for r in rows) == 537 and ids == manifest['ordered_narrator_ids']
        and manifest['order_salt'] == preparation.ORDER_SALT
        and ids == sorted(ids, key=lambda i: (pipeline.digest((preparation.ORDER_SALT + i).encode()), i)), 'all552 frozen ordered identities required')
    _check(all(r['narrator_fidelity_established'] is False and r['fidelity_conjoined_witness_established'] is False for r in rows)
        and manifest['narrator_fidelity_established'] is False and manifest['fidelity_conjoined_witness_established'] is False
        and manifest['preselected_factual_summary_sha256'] == preparation.FACTUAL_SHA, 'whole preselected fidelity remains unestablished')
    candidates = [r for r in rows if r['candidate_bindings']]
    ordered = [r['id'] for r in candidates]
    _check(manifest['candidate_narrator_positions'] == len(candidates)
        and manifest['ordered_candidate_narrator_ids'] == ordered
        and [r['id'] for c in chunks for r in c['records']] == ordered, 'all candidate narrators once per stream required')
    binding_inventory = [dict(id=r['id'], binding_path=b['binding_path'], original_status=b['original_status']) for r in candidates for b in r['candidate_bindings']]
    _check(_same(manifest['candidate_binding_inventory'], binding_inventory)
        and manifest['candidate_binding_positions'] == len(binding_inventory), 'every original candidate binding required')
    by_id = {r['id']: r for r in rows}
    for n, (ref, chunk) in enumerate(zip(manifest['chunks'], chunks)):
        _check(chunk['schema'] == preparation.VERSION + '_chunk' and type(chunk['chunk_number']) is int and chunk['chunk_number'] == n
            and 1 <= len(chunk['records']) <= 4 and ref['narrator_positions'] == len(chunk['records'])
            and ref['path'] == f'chunk_{n:03d}.json' and ref['sha256'] == pipeline.digest(pipeline.encoded(chunk) + b'\n')
            and ref['narrator_ids'] == [r['id'] for r in chunk['records']]
            and all(_same(r, by_id[r['id']]) for r in chunk['records']), 'exact bounded first candidate chunk differs')
        _check(_same(ref['binding_ids'], [dict(id=r['id'], binding_path=b['binding_path']) for r in chunk['records'] for b in r['candidate_bindings']]), 'chunk binding inventory differs')
        for name in ('review_protocol', 'review_output_schema', 'frozen_reference'):
            _check(_same(chunk[name], manifest[name]), 'chunk frozen dependency differs')
    _check(manifest['review_protocol'] == preparation.descriptor(preparation.PROTOCOL, preparation.PROTOCOL_SHA)
        and manifest['review_output_schema'] == preparation.descriptor(preparation.OUTPUT_SCHEMA, preparation.schema_sha())
        and manifest['frozen_reference'] == preparation.descriptor(pipeline.REFERENCE, pipeline.REFERENCE_HASH), 'frozen dependency differs')
    assignments = [dict(reviewer_id=s, chunk_number=n, chunk_path=r['path'], chunk_sha256=r['sha256']) for s in preparation.STREAMS for n, r in enumerate(manifest['chunks'])]
    _check(_same(manifest['assignments'], assignments), 'both streams require complete identical assignments')


def validate_receipts(manifest, manifest_sha, launch, launch_sha, outcomes):
    _closed(launch, ('schema', 'manifest_sha256', 'protocol_sha256', 'output_schema_sha256', 'contexts'), 'launch')
    _closed(outcomes, ('schema', 'launch_receipt_sha256', 'manifest_sha256', 'contexts'), 'outcomes')
    count = len(manifest['chunks'])
    _check(launch['schema'] == VERSION + '_launch' and outcomes['schema'] == VERSION + '_outcomes'
        and launch['manifest_sha256'] == outcomes['manifest_sha256'] == manifest_sha
        and outcomes['launch_receipt_sha256'] == launch_sha and launch['protocol_sha256'] == preparation.PROTOCOL_SHA
        and launch['output_schema_sha256'] == preparation.schema_sha(), 'root receipt dependency differs')
    _check(type(launch['contexts']) is list and type(outcomes['contexts']) is list
        and len(launch['contexts']) == len(outcomes['contexts']) == 2 * count, 'both streams require all candidate contexts, zero if none')
    planned, retained, contexts, paths = {}, {}, set(), set()
    for item in launch['contexts']:
        _closed(item, ('reviewer_id', 'independent_context_id', 'chunk_number', 'chunk_sha256', 'first_artifact_path', 'fresh_context', 'first_pass', 'peer_judgments_supplied'), 'launch context')
        stream, n, context = item['reviewer_id'], item['chunk_number'], item['independent_context_id']
        _check(stream in preparation.STREAMS and type(n) is int and 0 <= n < count
            and type(context) is str and bool(context.strip()) and context not in contexts
            and type(item['first_artifact_path']) is str and bool(item['first_artifact_path'])
            and item['fresh_context'] is True and item['first_pass'] is True and item['peer_judgments_supplied'] is False
            and item['chunk_sha256'] == manifest['chunks'][n]['sha256'], 'exclusive fresh launch/chunk differs')
        path = str(Path(item['first_artifact_path']).resolve())
        _check(path not in paths and (stream, n) not in planned, 'duplicate first path/context position')
        contexts.add(context)
        paths.add(path)
        planned[stream, n] = item
    _check(set(planned) == {(s, n) for s in preparation.STREAMS for n in range(count)}, 'both streams must cover every candidate chunk')
    for item in outcomes['contexts']:
        _closed(item, ('reviewer_id', 'independent_context_id', 'chunk_number', 'chunk_sha256', 'first_artifact', 'terminal_attestation'), 'outcome context')
        key = item['reviewer_id'], item['chunk_number']
        _check(key in planned and key not in retained, 'duplicate/unplanned outcome')
        first = planned[key]
        _check(all(_same(item[n], first[n]) for n in ('reviewer_id', 'independent_context_id', 'chunk_number', 'chunk_sha256')), 'outcome launched identity differs')
        if item['first_artifact'] is not None:
            _closed(item['first_artifact'], ('path', 'sha256'), 'first artifact')
            _check(Path(item['first_artifact']['path']).resolve() == Path(first['first_artifact_path']).resolve(), 'reserved first path differs')
        terminal = item['terminal_attestation']
        if terminal is not None:
            _closed(terminal, ('independent_context_id', 'chunk_sha256', 'observed_terminal', 'observation_kind', 'observation_artifact'), 'terminal')
            _check(terminal['independent_context_id'] == first['independent_context_id'] and terminal['chunk_sha256'] == first['chunk_sha256']
                and terminal['observed_terminal'] is True and terminal['observation_kind'] in ('completed', 'failed', 'interrupted'), 'actual terminal observation required')
            _closed(terminal['observation_artifact'], ('path', 'sha256'), 'terminal observation artifact')
        retained[key] = item
    return planned, retained


def collect(manifest, chunks, manifest_sha, launch, launch_sha, outcomes, schema, read_bound=_bound_bytes):
    validate_manifest(manifest, chunks)
    planned, receipts = validate_receipts(manifest, manifest_sha, launch, launch_sha, outcomes)
    collected, rows = [], []
    for stream in preparation.STREAMS:
        for n, chunk in enumerate(chunks):
            first, outcome = planned[stream, n], receipts[stream, n]
            raw, observation, artifact_errors = None, None, []
            if outcome['first_artifact'] is not None:
                try:
                    raw = read_bound(outcome['first_artifact'])
                    if pipeline.digest(raw) != outcome['first_artifact']['sha256']:
                        artifact_errors.append('first artifact SHA-256 differs from unchanged outcome descriptor')
                except ArtifactBindingError as exc:
                    raw = exc.observed_bytes
                    artifact_errors.append(type(exc).__name__ + ': ' + str(exc))
                except (OSError, ValueError) as exc:
                    artifact_errors.append(type(exc).__name__ + ': ' + str(exc))
            if outcome['terminal_attestation'] is not None:
                ref = outcome['terminal_attestation']['observation_artifact']
                observation = read_bound(ref)
                _check(pipeline.digest(observation) == ref['sha256'] and bool(observation.strip()), 'actual nonempty terminal observation binding differs')
            _check((raw is not None and not artifact_errors) or observation is not None,
                   'collection not ready: no matching first file or actual terminal observation for ' + first['independent_context_id'])
            expected = dict(reviewer_id=stream, independent_context_id=first['independent_context_id'], manifest_sha256=manifest_sha,
                chunk_sha256=first['chunk_sha256'], protocol_sha256=preparation.PROTOCOL_SHA, output_schema_sha256=preparation.schema_sha())
            result = validate_chunk(raw, chunk, expected, schema, artifact_errors)
            result.update(expected, chunk_number=n, launch_context=deepcopy(first), outcome_context=deepcopy(outcome),
                terminal_observation_bytes_base64=None if observation is None else base64.b64encode(observation).decode('ascii'))
            collected.append(result)
            judgments = {r['id']: r for r in result['parsed_first']['records']} if result['structurally_valid'] else {}
            for record in chunk['records']:
                rows.append(dict(reviewer_id=stream, independent_context_id=first['independent_context_id'], chunk_number=n,
                    id=record['id'], unit_id=record['unit_id'], text_sha256=record['text_sha256'],
                    original_binding_inventory=deepcopy(record['candidate_bindings']), chunk_structurally_valid=result['structurally_valid'],
                    review_state='retained_structurally_valid_judgment' if result['structurally_valid'] else 'unestablished',
                    first_judgment=deepcopy(judgments.get(record['id'])), semantic_correctness_established_by_code=False))
    return dict(schema=VERSION, manifest_sha256=manifest_sha, launch_receipt_sha256=launch_sha,
        narrator_opportunities=552, narrator_availability=deepcopy(manifest['narrator_availability']), pair_opportunities=684,
        pairs=deepcopy(manifest['pairs']), candidate_narrator_positions=manifest['candidate_narrator_positions'],
        candidate_binding_inventory=deepcopy(manifest['candidate_binding_inventory']), reviewer_opportunities=2 * manifest['candidate_narrator_positions'],
        chunks=collected, rows=rows, narrator_fidelity_established=False, fidelity_conjoined_witness_established=False,
        preselected_factual_summary_sha256=preparation.FACTUAL_SHA, semantic_correctness_established_by_code=False,
        property_tracking_performed=False, no_api_or_model_calls=True, no_replacement_or_gap_filling=True, whole_chunk_failure_policy=True)


def load_prepared_actual(path, sha):
    manifest = pipeline.bound_json(path, sha)
    names = ('index', 'preparation', 'collection', 'derived', 'execution_receipt', 'manifest', 'factual_summary', 'root_terminal_attestation')
    deps, args = manifest['input_dependencies'], {}
    for name in names:
        args[name], args[name + '_sha256'] = Path(deps[name]['path']), deps[name]['sha256']
    terminal = deps.get('terminal_receipt')
    args.update(terminal_receipt=None if terminal is None else Path(terminal['path']), terminal_receipt_sha256=None if terminal is None else terminal['sha256'])
    index, prepared, derived, exports, factual, verified = preparation.load_actual(SimpleNamespace(**args))
    expected, recomputed = preparation.assemble(index, prepared, derived, exports, factual)
    for name, value in expected.items():
        _check(_same(manifest.get(name), value), 'prepared manifest differs from complete actual recomputation: ' + name)
    _check(_same(manifest.get('terminal_verification'), verified) and _same(manifest.get('source_sha256'), preparation.source_hashes()),
           'prepared terminal/source binding differs')
    _check(pipeline.digest(Path(shared.__file__).read_bytes()) == SHARED_SHA, 'pinned shared retention primitives changed')
    root, chunks = Path(path).parent, []
    for ref, original in zip(manifest['chunks'], recomputed):
        chunk_path = root / ref['path']
        _check(chunk_path.resolve().parent == root.resolve(), 'chunk escapes prepared directory')
        chunk = pipeline.bound_json(chunk_path, ref['sha256'])
        _check(_same(chunk, original), 'actual chunk differs from complete recomputation')
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
    _check(not args.output.exists(), 'refusing replacement of first source-review collection')
    manifest, chunks = load_prepared_actual(args.prepared_manifest, args.prepared_manifest_sha256)
    launch = pipeline.bound_json(args.launch_receipt, args.launch_receipt_sha256)
    outcomes = pipeline.bound_json(args.outcome_receipt, args.outcome_receipt_sha256)
    result = collect(manifest, chunks, args.prepared_manifest_sha256, launch, args.launch_receipt_sha256, outcomes, frozen_output_schema())
    result['input_dependencies'] = {n: preparation.descriptor(getattr(args, n), getattr(args, n + '_sha256')) for n in ('prepared_manifest', 'launch_receipt', 'outcome_receipt')}
    result['source_sha256'] = {str(Path(p).resolve().relative_to(pipeline.ROOT)): pipeline.digest(Path(p).read_bytes())
        for p in (__file__, preparation.__file__, shared.__file__, pipeline.ROOT / 'tests/test_attention_relational_narrator_source_closure_collection_v1.py')}
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
