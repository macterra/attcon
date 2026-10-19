"""Prepare complete supplementary narrator source reviews; never infer semantics.

No API, model fitting, reviewer launch, semantic grading or property tracking is
implemented here. Actual loading requires the original V3 terminal observation
and all 47 runtime sources, before loading any reference labels. A supplied root
receipt binds an observation; this program cannot observe the live tool handle.
"""
import argparse
from copy import deepcopy
from pathlib import Path

import attention_relational_template_review_v1 as template
import attention_relational_quote_measurement_v3 as measurement
import attention_relational_reporting_pipeline_v1 as pipeline


VERSION = 'attention_relational_narrator_source_closure_v1'
PROTOCOL = 'docs/ATTENTION_RELATIONAL_NARRATOR_SOURCE_CLOSURE_V1_PLAN.md'
PROTOCOL_SHA = '0010bcc069864443ee4ae241eb06250f6900f8f3ce2a3f6cbd2c5429d2ff81db'
OUTPUT_SCHEMA = 'audits/attention_relational_report_v1/narrator_source_closure_v1/protocol/first_review_output_schema.json'
FACTUAL_SHA = '4d78e6ac95fb318a491ba9ca689252525d3892aaa1541d8e0f78fee429d5e5c2'
ORDER_SALT = 'narrator_source_closure_v1_order_20261019:'
STREAMS = ['reviewer_0', 'reviewer_1']


def digest(value):
    return template.digest(value)


def descriptor(path, sha):
    return dict(path=str(path), sha256=sha)


def path_id(kind, path):
    return template.path_id(kind, path)


def output_schema():
    """Closed structural contract; semantic/coverage enforcement belongs to collector.

    An unrestricted JSON value is deliberate only for exact source literals.
    Optional conclusions use explicit unresolved states or null property identity,
    never omission. Array coverage and all-operand sufficiency require review.
    """
    def obj(fields):
        return dict(type='object', properties=fields, required=list(fields), additionalProperties=False)

    def arr(items, **kw):
        return dict(type='array', items=items, **kw)

    def enum(*values):
        return dict(type='string', enum=list(values))

    string = dict(type='string')
    path = arr(dict(anyOf=[string, dict(type='integer', minimum=0)]))
    assessment = enum('supported', 'unsupported', 'unresolved')
    origin = enum('learned_current_A', 'learned_conditional_A', 'inherited_A', 'copied_W11',
                  'native_W', 'comparison_W', 'sensor', 'ordinary', 'analytic', 'mixed',
                  'global_convention', 'unresolved')
    operand = obj(dict(packet_path=path, value={}, semantic_role=enum('content', 'status',
        'processing_relation', 'context_only'), source_category=origin, provenance_paths=arr(path),
        origin_field_paths=arr(path), represented_time_paths=arr(path), reason=string))
    calculation = obj(dict(operator=string, expression=string, operand_indices=arr(dict(type='integer', minimum=0)),
        all_raw_operands_listed=dict(type='boolean'), independent_recomputation_performed=dict(type='boolean'),
        recomputed_value={}, supplied_value={}, comparison=enum('equal', 'different', 'unresolved'), notes=string))
    identity = obj(dict(source_operand_indices=arr(dict(type='integer', minimum=0)), operator=string,
        content_referent=string, attribute=string, quantifier=string, scope=string, time=string))
    feature = obj(dict(feature=enum('C', 'O', 'D', 'A'), original_status=enum('explicit', 'ambiguous', 'absent'),
        evidence_paths=arr(path), grounding=assessment, notes=string))
    meaning = obj(dict(assessment=assessment, asserted_meaning=string, content_referent=string,
        attribute=string, quantifier=string, scope=string, time=string, binding_evidence_paths=arr(path), notes=string))
    ownership = obj(dict(assessment=assessment, categories=arr(origin), a_content_established=enum('yes', 'no', 'unresolved'),
        native_estimated_operand_indices=arr(dict(type='integer', minimum=0)),
        exporter_derived_operand_indices=arr(dict(type='integer', minimum=0)), notes=string))
    tracking = obj(dict(property_identity=dict(anyOf=[identity, dict(type='null')]),
        eligible=enum('yes', 'no', 'unresolved'), unchanged_property_is_null=dict(type='boolean', const=True),
        missing_support_is_unknown=dict(type='boolean', const=True), unaffected_property_paths=arr(path), notes=string))
    binding = obj(dict(binding_path=path, original_status=enum('explicit', 'ambiguous', 'absent'),
        feature_reviews=arr(feature), recorded_feature_grounding=assessment, source_meaning=meaning,
        source_ownership=ownership, operands=arr(operand), calculations=arr(calculation), tracking_support=tracking,
        unresolved_requirements=arr(string), notes=string))
    record = obj(dict(id=string, unit_id=string, text_sha256=string, candidate_binding_reviews=arr(binding),
        additional_concerns=arr(string), notes=string))
    return obj(dict(schema=enum(VERSION + '_first_chunk_review'), reviewer_id=enum(*STREAMS), independent_context_id=string,
        first_pass=dict(type='boolean', const=True), independent_context=dict(type='boolean', const=True),
        peer_judgments_supplied=dict(type='boolean', const=False), manifest_sha256=string, chunk_sha256=string,
        protocol_sha256=string, output_schema_sha256=string, records=arr(record, minItems=1, maxItems=4), limitations=arr(string)))


def schema_sha():
    return pipeline.digest(pipeline.encoded(output_schema()) + b'\n')


def source_addresses(value, path=()):
    """Every exact leaf including empty containers, with no inferred ownership.

    Complete packet/provenance and indexed declared origin chains are also kept.
    Container paths can be selected directly from the packet by later reviewers.
    Derived summaries are exporter values, not additional native estimates.
    """
    if isinstance(value, dict) and value:
        return [row for key, child in value.items() for row in source_addresses(child, path + (key,))]
    if isinstance(value, list) and value:
        return [row for i, child in enumerate(value) for row in source_addresses(child, path + (i,))]
    return [dict(packet_path=list(path), value=deepcopy(value),
                 representation='exporter_derived_summary' if 'derived' in path else 'supplied_packet_field',
                 semantic_ownership='unresolved')]


def build_record(unit, measured, packet, provenance, indexed):
    pipeline.require(unit['template'] is False and measured['template'] is False
        and all(unit[k] == measured[k] for k in ('id', 'unit_id', 'text_sha256', 'case', 'actor_text_available'))
        and unit['unit_id'] == pipeline.blind_id(unit['id'])
        and unit['text_sha256'] == pipeline.digest(unit['text'].encode('utf-8'))
        and provenance['id'] == unit['id'] and provenance['case'] == unit['case']
        and indexed['packet_content_sha256'] == digest(packet)
        and indexed['provenance_content_sha256'] == digest(provenance), 'narrator/source/reference exact identity differs')
    annotation = measured['canonical_annotation']
    established = measured['established'] is True
    if established:
        pipeline.require(measured['transport_complete'] is True and unit['actor_text_available'] is True
            and measured['derivation']['valid'] is True
            and pipeline.validate_blind_annotation(unit['text'], unit['unit_id'], annotation)['valid'],
            'established narrator reference differs; no label salvage')
    else:
        pipeline.require(measured['established'] is False and annotation is None,
                         'unestablished narrator must have no canonical annotation')
    components, evidence, bindings = [], [], []
    if established:
        for section, name in template.COMPONENTS:
            path = [section, name]
            component = annotation[section][name]
            components.append(dict(component_path=path, recorded_component=deepcopy(component)))
            entries = [([], component['evidence'])] if section != 'composites' else [
                (['bindings', i], b['evidence']) for i, b in enumerate(component['bindings'])]
            for suffix, entries_list in entries:
                for i, entry in enumerate(entries_list):
                    pipeline.require(unit['text'][entry['start']:entry['end']] == entry['quote'], 'exact narrator quote differs')
                    ep = path + suffix + ['evidence', i]
                    evidence.append(dict(evidence_id=path_id('evidence', ep), evidence_path=ep,
                        recorded_evidence=deepcopy(entry), semantic_attribution='awaiting_independent_review'))
                if suffix:
                    bp = path + suffix
                    required = ('C', 'O', 'D') if name == 'COD' else ('C', 'A')
                    bindings.append(dict(binding_id=path_id('binding', bp), binding_path=bp,
                        original_status=component['status'], original_binding=deepcopy(annotation[section][name]['bindings'][suffix[1]]),
                        required_features=list(required),
                        feature_evidence_paths={f: [['features', f, 'evidence', j]
                            for j in range(len(annotation['features'][f]['evidence']))] for f in required},
                        binding_evidence_paths=[bp + ['evidence', j] for j in range(len(entries_list))],
                        recorded_feature_grounding='unresolved', source_meaning='unresolved', source_ownership='unresolved'))
    bindings.sort(key=lambda b: path_id('', b['binding_path']))
    state = ('unavailable_narrator' if not unit['actor_text_available'] else 'unestablished_reference'
             if not established else 'candidate' if bindings else 'no_original_binding')
    return dict(schema=VERSION + '_record', id=unit['id'], unit_id=unit['unit_id'], case=deepcopy(unit['case']),
        text=unit['text'], text_sha256=unit['text_sha256'], actor_text_available=unit['actor_text_available'],
        reference_established=established, inventory_disposition=state,
        reference_measurement=deepcopy(measured), canonical_first_annotation=deepcopy(annotation),
        complete_source_packet=deepcopy(packet), complete_source_provenance=deepcopy(provenance),
        source_addresses=source_addresses(packet),
        declared_source_property_index=[deepcopy(u) for u in indexed['units'] if u['kind'] in ('origin', 'role_peaks')],
        components=components, evidence=evidence, candidate_bindings=bindings,
        semantic_review_performed=False, narrator_fidelity_established=False,
        fidelity_conjoined_witness_established=False)


def assemble(index, prepared, derived, exports, factual):
    # Reuse all1104 reference checks and complete all552/684 construction checks.
    template.assemble(index, prepared, derived)
    inventory = pipeline.validate_export(exports)
    pipeline.require(template.construction.build_index(exports) == {k: v for k, v in index.items()
        if k not in ('input_dependencies', 'source_sha256', 'no_api_or_model_calls', 'chronology')},
        'first construction index differs from complete source recomputation')
    facts = pipeline._index(factual['all_text_results'])
    units, measured = pipeline._index(prepared['units']), pipeline._index(derived['rows'])
    pipeline.require(set(facts) == set(units) and len(facts) == 1104, 'whole preselected factual inventory required')
    for identifier, row in measured.items():
        fact = facts[identifier]
        pipeline.require(all(row[k] == fact[k] for k in ('unit_id', 'text_sha256', 'case', 'template', 'actor_text_available')),
                         'factual exact identity differs')
        if not row['template']:
            pipeline.require(fact['fidelity_established'] is False, 'narrator fidelity cannot be upgraded')
    indexed = {r['source_id']: r['index'] for r in index['templates']}
    ordered = sorted(inventory['request_ids'], key=lambda i: (pipeline.digest((ORDER_SALT + i).encode('utf-8')), i))
    records = [build_record(units[i], measured[i], inventory['packets'][i]['content'], inventory['provenance'][i], indexed[i]) for i in ordered]
    candidates = [r for r in records if r['candidate_bindings']]
    chunks = []
    for number, start in enumerate(range(0, len(candidates), 4)):
        selected = candidates[start:start + 4]
        chunks.append(dict(schema=VERSION + '_chunk', chunk_number=number, records=selected,
            review_protocol=descriptor(PROTOCOL, PROTOCOL_SHA), frozen_reference=descriptor(pipeline.REFERENCE, pipeline.REFERENCE_HASH),
            review_output_schema=descriptor(OUTPUT_SCHEMA, schema_sha()), source_meaning_established=False))
    pipeline.require(len(records) == 552 and len(index['pairs']) == 684
        and sum(r['actor_text_available'] for r in records) == 537, 'all552/537 narrator availability and684pairs required')
    refs = [dict(path=f'chunk_{n:03d}.json', sha256=pipeline.digest(pipeline.encoded(c) + b'\n'),
        narrator_ids=[r['id'] for r in c['records']], narrator_positions=len(c['records']),
        binding_ids=[dict(id=r['id'], binding_path=b['binding_path']) for r in c['records'] for b in r['candidate_bindings']])
        for n, c in enumerate(chunks)]
    pairs = [dict(pair=deepcopy(p), disposition='awaiting_property_source_review', property_tracking_established=False) for p in index['pairs']]
    manifest = dict(schema=VERSION + '_manifest', order_salt=ORDER_SALT, narrator_opportunities=552,
        narrator_availability=records, pair_opportunities=684, pairs=pairs,
        ordered_narrator_ids=ordered, ordered_candidate_narrator_ids=[r['id'] for r in candidates],
        candidate_binding_inventory=[dict(id=r['id'], binding_path=b['binding_path'], original_status=b['original_status'])
            for r in candidates for b in r['candidate_bindings']],
        candidate_narrator_positions=len(candidates), candidate_binding_positions=sum(len(r['candidate_bindings']) for r in candidates),
        chunks=refs, reviewer_streams=deepcopy(STREAMS),
        assignments=[dict(reviewer_id=s, chunk_number=n, chunk_path=ref['path'], chunk_sha256=ref['sha256'])
            for s in STREAMS for n, ref in enumerate(refs)],
        review_protocol=descriptor(PROTOCOL, PROTOCOL_SHA), review_output_schema=descriptor(OUTPUT_SCHEMA, schema_sha()),
        frozen_reference=descriptor(pipeline.REFERENCE, pipeline.REFERENCE_HASH),
        content_bindings={k: digest(v) for k, v in dict(index=index, preparation=prepared, derived=derived, exports=exports, factual_summary=factual).items()},
        preselected_factual_summary_sha256=FACTUAL_SHA, whole_factual_inventory_preserved=True,
        first_artifact_policy=dict(fresh_context_per_chunk=True, maximum_narrator_packets_per_context=4,
            peer_judgments_supplied=False, first_pass_only=True, replacements_allowed=False, gap_filling_allowed=False,
            incomplete_chunk_disposition='whole_chunk_unestablished', timeout_is_terminal=False,
            missing_support_is_unknown=True, no_status_or_scope_or_time_upgrades=True),
        semantic_review_performed=False, source_meaning_established=False, narrator_fidelity_established=False,
        fidelity_conjoined_witness_established=False, no_api_or_model_calls=True,
        independent_system_confirmation=False, repeated_development_cases=12,
        interpretation='Supplementary witness/source analysis; no whole-prose fidelity, appearance or felt-experience conclusion is generated.')
    return manifest, chunks


def load_actual(args):
    """Terminal and all47 source checks precede any reference-output loader."""
    pipeline.require(args.manifest_sha256 == template.MANIFEST_SHA
        and Path(args.manifest).resolve() == (pipeline.ROOT / template.MANIFEST).resolve(), 'pinned actual V3 manifest required')
    runtime = pipeline.bound_json(args.manifest, template.MANIFEST_SHA)
    root = pipeline.bound_json(args.root_terminal_attestation, args.root_terminal_attestation_sha256)
    sources = runtime['source_sha256']
    pipeline.require(root.get('schema') == VERSION + '_root_terminal' and root.get('terminal_observed') is True
        and type(root.get('tool_session_id')) is int and root['tool_session_id'] == 5286
        and root.get('manifest_sha256') == template.MANIFEST_SHA
        and root.get('execution_receipt_sha256') == args.execution_receipt_sha256
        and root.get('runtime_source_sha256') == sources and len(sources) == 47,
        'actual original terminal observation and all47 source attestations required')
    pipeline.require(all(pipeline.digest((pipeline.ROOT / p).read_bytes()) == sha for p, sha in sources.items()),
                     'all47 frozen runtime source files must remain unchanged')
    pipeline.require(pipeline.digest((pipeline.ROOT / PROTOCOL).read_bytes()) == PROTOCOL_SHA, 'frozen narrator closure plan differs')
    pipeline.require(pipeline.bound_json(pipeline.ROOT / OUTPUT_SCHEMA, schema_sha()) == output_schema(), 'closed first review schema differs')
    index, prepared, derived, terminal = template.load_actual(args)
    # This reload independently recomputes preparation from exact first sources.
    pipeline.require(measurement.load_preparation(args.preparation, args.preparation_sha256) == prepared,
                     'complete original preparation recomputation differs')
    inputs = prepared['lineage']['inputs']
    exports = pipeline.bound_json(inputs['exports']['path'], inputs['exports']['sha256'])
    pipeline.require(args.factual_summary_sha256 == FACTUAL_SHA, 'only whole preselected factual summary allowed')
    factual = pipeline.bound_json(args.factual_summary, FACTUAL_SHA)
    recomputed_factual = measurement.verify_factual_summary(factual, prepared)
    return index, prepared, derived, exports, recomputed_factual, terminal


def source_hashes():
    sources = template.source_hashes()
    paths = [Path(__file__), pipeline.ROOT / PROTOCOL, pipeline.ROOT / OUTPUT_SCHEMA,
             pipeline.ROOT / 'tests/test_attention_relational_narrator_source_closure_v1.py']
    sources.update({str(p.resolve().relative_to(pipeline.ROOT)): pipeline.digest(p.read_bytes()) for p in paths})
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'preparation', 'collection', 'derived', 'execution-receipt', 'manifest', 'factual-summary', 'root-terminal-attestation'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--terminal-receipt', type=Path)
    parser.add_argument('--terminal-receipt-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first narrator source-review preparation')
    index, prepared, derived, exports, factual, terminal = load_actual(args)
    manifest, chunks = assemble(index, prepared, derived, exports, factual)
    names = ('index', 'preparation', 'collection', 'derived', 'execution_receipt', 'manifest', 'factual_summary', 'root_terminal_attestation')
    manifest.update(input_dependencies={n: descriptor(getattr(args, n), getattr(args, n + '_sha256')) for n in names},
                    source_sha256=source_hashes(), terminal_verification=terminal)
    if args.terminal_receipt:
        manifest['input_dependencies']['terminal_receipt'] = descriptor(args.terminal_receipt, args.terminal_receipt_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    for ref, chunk in zip(manifest['chunks'], chunks, strict=True):
        pipeline.exclusive_json(args.output / ref['path'], chunk)
    pipeline.exclusive_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
