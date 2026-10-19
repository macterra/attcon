"""Prepare complete source-aware semantic-review packets; no grading/API execution.

Terminal execution is checked before loading reference outputs. Frozen V3
derivation is recomputed without label repair. Geometry never establishes meaning.
Receipt checks bind preserved attestations; they cannot observe a live tool handle.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import attention_relational_quote_measurement_v3 as measurement
import attention_relational_quote_reference_v3 as quotes
import attention_relational_template_index_v1 as construction
import attention_relational_reporting_pipeline_v1 as pipeline
import collect_interrupted_attention_relational_quote_v3 as interrupted


VERSION = 'attention_relational_template_review_v1'
INDEX_SHA = 'c7b36e2275099ddf5470fb7f5063b85f51d8de401e874f93d65799ec66a747d3'
PROTOCOL = 'docs/ATTENTION_RELATIONAL_TEMPLATE_SEMANTIC_REVIEW_V1.md'
PROTOCOL_SHA = 'e55a74c579593711a3ef40e308b4ca0b2d7b220f1a491a52d7f2ab2e481bf7a7'
OUTPUT_SCHEMA = 'audits/attention_relational_report_v1/template_attribution_v1/semantic_review/first_review_output_schema.json'
OUTPUT_SCHEMA_SHA = 'f7a70c0ef14b0f6279e9ae2db40f34c48442493c4b25b05e159fe915529e1a27'
MANIFEST = 'audits/attention_relational_report_v1/quote_reference_v3/preparation/runtime_manifest.json'
MANIFEST_SHA = '89b124d9b5cb3ff72892f3e08380126320584346126bbf543e8fcca5d34cf043'
RELEASE = '5e5857fa1ac0c0affeeb64873e142a3f847f990e'
ORDER_SALT = 'template_semantic_review_v1_order_20261019:'
COMPONENTS = [('features', n) for n in ('C', 'O', 'D', 'A')] + [('auxiliary', n) for n in pipeline.AUXILIARY] + [('composites', n) for n in ('COD', 'CA')]


def digest(value):
    return pipeline.digest(pipeline.encoded(value))


def descriptor(path, sha):
    return dict(path=str(path), sha256=sha)


def path_id(kind, path):
    return kind + ':' + ''.join('/' + str(p).replace('~', '~0').replace('/', '~1') for p in path)


def verify_terminal(execution, log_sha, completion_exists, terminal=None, execution_sha=None):
    """Pure metadata contract, not evidence that a supplied fictional file is real."""
    expected = ['.venv/bin/python', 'scripts/run_attention_relational_reports_v1.py', '--manifest', MANIFEST,
                '--commit', RELEASE, '--output-root', measurement.RATE_ROOT]
    pipeline.require(execution.get('command') == expected and execution.get('preservation_commit') == RELEASE
        and execution.get('stage') == 'rate' and execution.get('retry_performed') is False
        and type(execution.get('exit_code')) is int and execution.get('log_sha256') == log_sha,
        'actual first V3 execution/command/release/log differs')
    if execution['exit_code'] == 0:
        pipeline.require(completion_exists and terminal is None, 'normal terminal requires actual first completion')
        return 'normal'
    pipeline.require(not completion_exists and type(terminal) is dict
        and terminal.get('schema') == 'attention_relational_quote_v3_terminal_receipt'
        and terminal.get('terminal_verified') is True and terminal.get('stage') == 'rate'
        and type(terminal.get('tool_session_id')) is int and terminal['tool_session_id'] > 0
        and terminal.get('manifest_sha256') == MANIFEST_SHA and terminal.get('output_root') == measurement.RATE_ROOT
        and terminal.get('execution_receipt_sha256') == execution_sha
        and type(terminal.get('child_exit_code')) is int
        and terminal.get('child_exit_code') == execution['exit_code'], 'actual interrupted V3 terminal attestation differs')
    return 'interrupted'


def validate_index(index):
    rows = index['templates']
    pipeline.require(index.get('schema') == construction.VERSION and len(rows) == 552
        and index.get('template_opportunities') == 552 and index.get('pair_opportunities') == len(index['pairs']) == 684
        and index.get('counts') == dict(header=552, origin=12720, role_peaks=25440, separator=38160)
        and index.get('partition_units') == 76872 and index.get('semantic_source_attribution_established') is False,
        'complete first construction index metadata differs')
    pipeline.require(len(pipeline._index(rows)) == 552 and index['ordered_source_ids'] == [r['source_id'] for r in rows]
        and index['pairs'] == pipeline.build_pair_index([dict(id=r['source_id'], case=r['case']) for r in rows]),
        'complete source/pair identity inventory differs')
    totals, cases = Counter(), Counter()
    for row in rows:
        item = row['index']
        units, text = item['units'], item['text']
        pipeline.require(row['id'] == row['source_id'] + '_template' and row['unit_id'] == pipeline.blind_id(row['id'])
            and item['schema'] == construction.VERSION + '_template'
            and item['text_sha256'] == pipeline.digest(text.encode('utf-8'))
            and all(item.get(k) is True for k in ('exact_text_verified', 'exact_propositions_verified',
                'independent_map_maxima_verified', 'complete_character_partition_verified'))
            and item.get('semantic_source_attribution_established') is False
            and len(pipeline._index(units, 'unit_id')) == len(units)
            and ''.join(u['text'] for u in units) == text
            and all(u['start'] == (units[i - 1]['end'] if i else 0) and u['end'] - u['start'] == len(u['text'])
                    and text[u['start']:u['end']] == u['text'] for i, u in enumerate(units))
            and Counter(u['kind'] for u in units) == Counter(item['counts'])
            and item['required_peak_count'] == item['counts']['role_peaks'], 'indexed property table/text metadata differs')
        for unit in units:
            if unit['kind'] in ('origin', 'role_peaks'):
                resolved = unit.get('source_resolution', {})
                pipeline.require(type(unit.get('packet_path')) is list and type(unit.get('context')) is dict
                    and resolved.get('state') == 'resolved' and bool(resolved.get('origin_chain'))
                    and all(resolved['binding'][k] == unit['context'][k] for k in ('panel', 'view', 'command', 'node', 'role'))
                    and (unit['kind'] != 'origin' or unit['value'] == resolved['origin_chain'][0]),
                    'indexed source property/origin metadata differs')
            else:
                pipeline.require(unit['kind'] in ('header', 'separator') and unit.get('packet_path') is None
                    and unit.get('source_resolution') is None, 'global/separator convention acquired a source')
        totals.update(item['counts'])
        cases[(row['case']['encoder_seed'], row['case']['world_index'])] += 1
    pipeline.require(totals == Counter(index['counts']) and len(cases) == 12 and set(cases.values()) == {46},
                     'all12x46 source cases/property counts required')


def evidence_geometry(item, entry):
    start, end = entry['start'], entry['end']
    text = item['text']
    pipeline.require(type(start) is int and type(end) is int and 0 <= start < end <= len(text)
        and text[start:end] == entry['quote'], 'exact evidence quote/offset differs')
    intersections = []
    for unit in item['units']:
        left, right = max(start, unit['start']), min(end, unit['end'])
        if left < right:
            intersections.append(dict(unit_id=unit['unit_id'], overlap_start=left, overlap_end=right,
                unit_contains_evidence=unit['start'] <= start and end <= unit['end'],
                evidence_contains_unit=start <= unit['start'] and unit['end'] <= end,
                partial_unit=left != unit['start'] or right != unit['end']))
    return dict(intersections=intersections, all_exact_quote_occurrences=quotes.exact_occurrences(text, entry['quote']),
                semantic_attribution='unresolved_geometry_only')


def build_record(template_indexrow, established_derivedrow):
    row, measured = template_indexrow, established_derivedrow
    item = row['index']
    pipeline.require(all(measured[k] == row[k] for k in ('id', 'unit_id', 'case')) and measured['template'] is True
        and row['id'] == row['source_id'] + '_template' and row['unit_id'] == pipeline.blind_id(row['id'])
        and measured['text_sha256'] == item['text_sha256'] == pipeline.digest(item['text'].encode('utf-8')),
        'template/reference exact ID/text/case binding differs')
    annotation = measured['canonical_annotation']
    established = measured['established'] is True
    if established:
        pipeline.require(measured['transport_complete'] is True and measured['actor_text_available'] is True
            and measured['derivation']['valid'] is True
            and pipeline.validate_blind_annotation(item['text'], row['unit_id'], annotation)['valid'],
            'established reference contract differs; no label salvage')
    else:
        pipeline.require(measured['established'] is False and annotation is None, 'unestablished reference must have no canonical annotation')
    components, evidence, bindings = [], [], []
    if established:
        for section, name in COMPONENTS:
            path = [section, name]
            component = annotation[section][name]
            cid = path_id('component', path)
            component_evidence, component_bindings = [], []
            entries = [([], component['evidence'])] if section != 'composites' else [
                (['bindings', i], b['evidence']) for i, b in enumerate(component['bindings'])]
            for suffix, entries_list in entries:
                bid = path_id('binding', path + suffix) if suffix else None
                ids = []
                for i, entry in enumerate(entries_list):
                    ep = path + suffix + ['evidence', i]
                    eid = path_id('evidence', ep)
                    ids.append(eid)
                    evidence.append(dict(evidence_id=eid, path=ep, component_id=cid, binding_id=bid,
                        recorded_evidence=deepcopy(entry), geometry=evidence_geometry(item, entry),
                        review_state='awaiting_independent_review'))
                component_evidence.extend(ids)
                if bid:
                    required = ('C', 'O', 'D') if name == 'COD' else ('C', 'A')
                    bindings.append(dict(binding_id=bid, path=path + suffix, component_id=cid,
                        own_evidence_ids=ids, required_features=list(required),
                        candidate_feature_evidence_ids={f: [path_id('evidence', ['features', f, 'evidence', j])
                            for j in range(len(annotation['features'][f]['evidence']))] for f in required},
                        compatibility_established=False, review_state='awaiting_independent_review'))
                    component_bindings.append(bid)
            components.append(dict(component_id=cid, path=path, recorded_status=component['status'],
                evidence_ids=component_evidence, binding_ids=component_bindings, review_state='awaiting_independent_review'))
    measurement_record = {k: deepcopy(v) for k, v in measured.items() if k != 'canonical_annotation'}
    return dict(schema=VERSION + '_record', id=row['id'], source_id=row['source_id'], unit_id=row['unit_id'],
        case=deepcopy(row['case']), text_sha256=item['text_sha256'], text=item['text'],
        panel_provenance=deepcopy(row['panel_provenance']),
        construction_metadata={k: deepcopy(v) for k, v in item.items() if k not in ('text', 'units')},
        property_table=deepcopy(item['units']), reference_measurement=measurement_record,
        canonical_first_annotation=deepcopy(annotation), reference_established=established,
        review_state='awaiting_independent_review' if established else 'unestablished',
        components=components, evidence=evidence, bindings=bindings, semantic_attribution_established=False)


def assemble(index, prepared, derived):
    validate_index(index)
    measurement.validate_prepared(prepared)
    pipeline.require(derived['schema'] == measurement.VERSION + '_derived_collection'
        and derived['preparation_content_sha256'] == digest(prepared) and derived['lineage'] == prepared['lineage'],
        'derived/prepared actual reference binding differs')
    units, measurements = pipeline._index(prepared['units']), pipeline._index(derived['rows'])
    templates = pipeline._index(index['templates'])
    expected = {i + suffix for i in index['ordered_source_ids'] for suffix in ('', '_template')}
    pipeline.require(set(units) == set(measurements) == expected and len(expected) == 1104
        and set(templates) == {i + '_template' for i in index['ordered_source_ids']}, 'full1104 reference inventory required')
    narrators = []
    for identifier, unit in units.items():
        measured = measurements[identifier]
        pipeline.require(all(unit[k] == measured[k] for k in ('id', 'unit_id', 'case', 'template', 'text_sha256', 'actor_text_available')),
                         'reference unit/text/case inventory differs')
        if unit['template']:
            pipeline.require(unit['text'] == templates[identifier]['index']['text'], 'prepared template text differs')
        else:
            narrators.append({k: deepcopy(v) for k, v in measured.items() if k != 'canonical_annotation'})
    ordered = sorted(templates, key=lambda i: (pipeline.digest((ORDER_SALT + i).encode('utf-8')), i))
    chunks = []
    for number, start in enumerate(range(0, 552, 24)):
        ids = ordered[start:start + 24]
        chunks.append(dict(schema=VERSION + '_chunk', chunk_number=number, ordered_template_ids=ids,
            review_protocol=descriptor(PROTOCOL, PROTOCOL_SHA), frozen_reference=descriptor(pipeline.REFERENCE, pipeline.REFERENCE_HASH),
            review_output_schema=descriptor(OUTPUT_SCHEMA, OUTPUT_SCHEMA_SHA),
            records=[build_record(templates[i], measurements[i]) for i in ids]))
    pipeline.require(len(chunks) == 23 and all(len(c['records']) == 24 for c in chunks)
        and len(narrators) == 552, 'all23x24 template chunks plus552 narrator availability rows required')
    manifest = dict(schema=VERSION + '_manifest', order_salt=ORDER_SALT, ordered_template_ids=ordered,
        template_opportunities=552, narrator_opportunities=552, narrator_availability=narrators,
        pair_opportunities=684, pairs=deepcopy(index['pairs']), reference_lineage=deepcopy(derived['lineage']),
        content_bindings=dict(index=digest(index), preparation=digest(prepared), derived=digest(derived)),
        chunks=[dict(path=f'chunk_{i:02d}.json', sha256=pipeline.digest(pipeline.encoded(c) + b'\n'),
                     ordered_template_ids=c['ordered_template_ids'], positions=24) for i, c in enumerate(chunks)],
        review_protocol=descriptor(PROTOCOL, PROTOCOL_SHA), frozen_reference=descriptor(pipeline.REFERENCE, pipeline.REFERENCE_HASH),
        review_output_schema=descriptor(OUTPUT_SCHEMA, OUTPUT_SCHEMA_SHA),
        frozen_reporting_protocol=descriptor(pipeline.PROTOCOL, pipeline.PROTOCOL_HASH),
        reviewer_streams=['reviewer_0', 'reviewer_1'], both_streams_receive_all_chunks_in_same_order=True,
        review_tasks=['All12 recorded components', 'Every evidence-list position, duplicates retained',
            'Every composite binding with independently reviewed compatible feature witnesses',
            'Precise relative property paths/values and all operands; overlap is geometry only', 'Additional concerns; no label repairs'],
        first_review_policy='Both streams retain all552 positions; missing/malformed/incomplete chunks stay unestablished with no replacements or peer feedback to unfinished contexts.',
        semantic_review_performed=False, semantic_attribution_established=False, narrator_fidelity_established=False)
    return manifest, chunks


def load_actual(args):
    """No raw/canonical reference-output loading before actual terminal checks."""
    pipeline.require(args.index_sha256 == INDEX_SHA and args.manifest_sha256 == MANIFEST_SHA
        and Path(args.manifest).resolve() == (pipeline.ROOT / MANIFEST).resolve(), 'pinned first index/V3 manifest required')
    manifest = pipeline.bound_json(args.manifest, MANIFEST_SHA)
    pipeline.require(manifest['stage'] == 'rate' and manifest['output_root'] == measurement.RATE_ROOT
        and manifest['request_count'] == 1104, 'wrong released V3 runtime')
    root = pipeline.ROOT / measurement.RATE_ROOT
    pipeline.require(Path(args.execution_receipt).resolve() == (root / 'first_execution.json').resolve(), 'actual first execution receipt path required')
    execution = pipeline.bound_json(args.execution_receipt, args.execution_receipt_sha256)
    log_sha = pipeline.digest((root / 'first_execution.log').read_bytes())
    terminal = None
    if args.terminal_receipt:
        pipeline.require(args.terminal_receipt_sha256 is not None and Path(args.terminal_receipt).resolve() == (root / 'first_terminal_receipt.json').resolve(),
                         'actual existing V3 terminal receipt required')
        terminal = pipeline.bound_json(args.terminal_receipt, args.terminal_receipt_sha256)
    completion_path = root / 'first_completion.json'
    mode = verify_terminal(execution, log_sha, completion_path.exists(), terminal, args.execution_receipt_sha256)
    claim = pipeline.decoded((root / 'first_batch_claim.json').read_bytes())
    pipeline.require(claim.get('manifest_sha256') == MANIFEST_SHA and claim.get('preservation_commit') == RELEASE
        and claim.get('stage') == 'rate' and claim.get('retry_performed') is False
        and claim.get('request_ids') == manifest['request_ids'] and claim.get('max_attempts') == 1104,
        'actual first V3 batch reservation differs')
    pipeline.require(pipeline.digest((pipeline.ROOT / PROTOCOL).read_bytes()) == PROTOCOL_SHA, 'frozen semantic-review protocol differs')
    pipeline.bound_json(pipeline.ROOT / OUTPUT_SCHEMA, OUTPUT_SCHEMA_SHA)
    pipeline.frozen_reference()
    sources = manifest['source_sha256']
    pipeline.require(all(sources.get(str(Path(m.__file__).relative_to(pipeline.ROOT))) == pipeline.digest(Path(m.__file__).read_bytes())
        for m in (measurement, quotes, pipeline)), 'actual frozen V3 derivation source differs')
    pipeline.require(sources.get(str(args.preparation)) == args.preparation_sha256, 'runtime did not bind actual preparation')
    prepared = pipeline.bound_json(args.preparation, args.preparation_sha256)
    requests = pipeline.bound_json(manifest['requests_path'], sources[manifest['requests_path']])
    lineage = pipeline.bound_json(manifest['lineage_path'], sources[manifest['lineage_path']])
    binding = measurement.build_binding(manifest, MANIFEST_SHA, requests, sources[manifest['requests_path']], lineage, sources[manifest['lineage_path']], prepared)
    collection = pipeline.bound_json(args.collection, args.collection_sha256)
    pipeline.require(collection['stage_binding'] == binding, 'collection is not actual released V3 binding')
    if mode == 'normal':
        completion_sha = pipeline.digest(completion_path.read_bytes())
        completion = pipeline.bound_json(completion_path, completion_sha)
        pipeline.require(completion.get('preservation_commit') == RELEASE and completion.get('manifest_sha256') == MANIFEST_SHA,
                         'actual first completion release/manifest differs')
        pipeline.require(collection.get('completion_sha256') == completion_sha and not collection.get('interrupted_batch', False),
                         'normal collection does not bind actual first completion')
        retained = pipeline.collect_first_responses(completion_path, completion_sha, binding['request_ids'], root, stage_binding=binding)
    else:
        deps = collection.get('input_dependencies', {})
        pipeline.require(collection.get('interrupted_batch') is True and collection.get('all_available_partial_files_preserved') is True
            and collection.get('normal_completion_fabricated') is False and 'completion_sha256' not in collection
            and deps.get('execution_receipt') == descriptor(args.execution_receipt, args.execution_receipt_sha256)
            and deps.get('terminal_receipt') == descriptor(args.terminal_receipt, args.terminal_receipt_sha256)
            and collection.get('source_sha256', {}).get(str(Path(interrupted.__file__))) == pipeline.digest(Path(interrupted.__file__).read_bytes()),
            'actual interrupted collector metadata/terminal bindings differ')
        retained = pipeline.collect_interrupted_responses(binding['request_ids'], Path(measurement.RATE_ROOT), stage_binding=binding)
    pipeline.require(all(collection.get(k) == v for k, v in retained.items()), 'collection differs from complete frozen first-outcome retention')
    derived = pipeline.bound_json(args.derived, args.derived_sha256)
    recomputed = measurement.derive_collection(prepared, collection)
    pipeline.require(all(derived.get(k) == v for k, v in recomputed.items()), 'derived rows differ from whole frozen V3 recomputation')
    return pipeline.bound_json(args.index, INDEX_SHA), prepared, recomputed, dict(mode=mode,
        execution_log=descriptor(root / 'first_execution.log', log_sha),
        first_batch_claim=descriptor(root / 'first_batch_claim.json', pipeline.digest((root / 'first_batch_claim.json').read_bytes())),
        first_completion=descriptor(completion_path, retained['completion_sha256']) if mode == 'normal' else None,
        live_tool_handle_observed_by_this_code=False)


def source_hashes():
    sources = measurement.source_hashes()
    sources.update(construction.source_hashes())
    paths = [Path(m.__file__) for m in (measurement, interrupted)] + [Path(__file__)]
    paths += [pipeline.ROOT / PROTOCOL, pipeline.ROOT / OUTPUT_SCHEMA, pipeline.ROOT / pipeline.REFERENCE, pipeline.ROOT / pipeline.PROTOCOL,
              pipeline.ROOT / 'tests/test_attention_relational_template_review_v1.py']
    sources.update({str(p.resolve().relative_to(pipeline.ROOT)): pipeline.digest(p.read_bytes()) for p in paths})
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'preparation', 'collection', 'derived', 'execution-receipt', 'manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--terminal-receipt', type=Path)
    parser.add_argument('--terminal-receipt-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first review packet directory')
    index, prepared, derived, terminal = load_actual(args)
    manifest, chunks = assemble(index, prepared, derived)
    manifest.update(input_dependencies={name: descriptor(getattr(args, name), getattr(args, name + '_sha256'))
        for name in ('index', 'preparation', 'collection', 'derived', 'execution_receipt', 'manifest')},
        terminal_verification=terminal, source_sha256=source_hashes(), no_api_or_model_calls=True)
    if args.terminal_receipt:
        manifest['input_dependencies']['terminal_receipt'] = descriptor(args.terminal_receipt, args.terminal_receipt_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    for ref, chunk in zip(manifest['chunks'], chunks, strict=True):
        pipeline.exclusive_json(args.output / ref['path'], chunk)
    pipeline.exclusive_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
