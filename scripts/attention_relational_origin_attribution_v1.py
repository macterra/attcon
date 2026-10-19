"""Post-development descriptive attribution; no API/model calls or endpoint changes.

Authored 2026-10-19 after narrator collection began and earlier narrative
inspection by the study team. Development did not inspect generated reports,
prose audits, or reference ratings. This is not a preregistered endpoint.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import attention_relational_reporting_pipeline_v1 as pipeline
import attention_relational_report_v1 as exporter
import inventory_attention_relational_origins_v1 as origins


PARAMETERS = {
    'endpoints': {'COD': ['C', 'O', 'D'], 'CA': ['C', 'A']},
    'offset_unit': 'Python Unicode code points; start inclusive, end exclusive',
    'containment': 'Exact supported auditor quote occurrence fully contains reference span',
    'repeated_auditor_quote': 'Retain every occurrence and mark attribution unresolved',
    'multiple_origin_classes': 'Retain every operand as mixed; never select one',
    'record_scope_or_invalid_binding': 'Unresolved; never infer an origin from panel alone',
    'evidence_inventory': 'All validated feature, composite, and auxiliary evidence entries',
    'templates': 'Primary summary verified; no template origin attribution or comparison',
    'unmatched_meaning': 'No containment attribution established; not negative consciousness evidence',
    'descriptor_categories': ['single_origin_class', 'mixed_origins', 'unresolved'],
}


def occurrences(text, quote):
    """Retain overlapping occurrences too; auditor quotes have no supplied offsets."""
    found = []
    if type(quote) is not str or not quote:
        return found
    start = text.find(quote)
    while start >= 0:
        found.append({'start': start, 'end': start + len(quote)})
        start = text.find(quote, start + 1)
    return found


def resolve_binding(binding, packet, provenance):
    result = {'binding': deepcopy(binding), 'state': 'unresolved'}
    if binding.get('view') == 'record_scope' or binding.get('basis') == 'record_scope':
        return dict(result, reason='record_scope_binding')
    if binding.get('basis') == 'not_supplied':
        return dict(result, reason='not_supplied_binding')
    key = tuple(binding.get(k) for k in ('panel', 'view', 'command', 'node'))
    # Lookup, not panel-name inference: both a supplied node and provenance are needed.
    node = next((n for k, n in pipeline._public_contexts(packet).items() if k == key), None)
    panel_sources = [p['source_key'] for p in provenance['panels']
                     if p['panel'] == binding.get('panel')]
    if node is None or len(panel_sources) != 1:
        return dict(result, reason='missing_or_ambiguous_context_or_provenance')
    origin = node.get('origin')
    try:
        category = origins.classify(origin)
        terminal = origin
        chain = [deepcopy(origin)]
        while terminal['kind'] == 'inherited_current':
            terminal = terminal['inherited_origin']
            chain.append(deepcopy(terminal))
    except (KeyError, TypeError, ValueError):
        return dict(result, reason='invalid_origin', supplied_origin=deepcopy(origin))
    source = panel_sources[0]
    if category in ('source_current', 'source_conditional', 'inherited_source_current'):
        actual_source = source if source in ('A', 'ordinary', 'analytic') else None
    elif category == 'copied_W11':
        actual_source = 'W' if source in ('A', 'ordinary', 'analytic') else None
    else:
        actual_source = {'native_W': 'W', 'sensor': 'sensor'}.get(category)
        if source != actual_source:
            actual_source = None
    if actual_source is None:
        return dict(result, reason='source_origin_inconsistency', panel_source_key=source,
                    supplied_origin=deepcopy(origin))
    return dict(result, state='resolved', panel_source_key=source, origin_chain=chain,
                origin_class={'origin_source_key': actual_source, 'category': category,
                              'view': binding['view'], 'origin_kind': origin['kind'],
                              'terminal_origin_kind': terminal['kind']})


def supported_claims(text, audit, packet, provenance):
    """Called only for a complete, validator-valid audit; retain all supported claims."""
    return [{'claim_index': i, 'quote': claim['quote'], 'reason': claim['reason'],
             'occurrences': occurrences(text, claim['quote']),
             'bindings': [resolve_binding(b, packet, provenance) for b in claim['bindings']]}
            for i, claim in enumerate(audit['claims']) if claim['verdict'] == 'supported']


def attribute_span(text, evidence, claims, audit_usable=True):
    """Describe exact containment only; no semantic entailment or quote repair."""
    result = {'reference_evidence': deepcopy(evidence), 'matches': [], 'origin_classes': []}
    start, end = evidence.get('start'), evidence.get('end')
    quote = evidence.get('quote')
    if (type(start) is not int or type(end) is not int or type(quote) is not str
            or not 0 <= start < end <= len(text) or text[start:end] != quote):
        return dict(result, descriptor='unresolved', origin_composition='none',
                    unresolved_reasons=['invalid_reference_offsets_or_quote'])
    if not audit_usable:
        return dict(result, descriptor='unresolved', origin_composition='none',
                    unresolved_reasons=['prose_audit_unestablished'])
    issues = set()
    classes = {}
    for claim in claims:
        for occurrence in claim['occurrences']:
            if occurrence['start'] <= start and end <= occurrence['end']:
                result['matches'].append({'claim_index': claim['claim_index'],
                    'claim_quote': claim['quote'], 'containing_occurrence': deepcopy(occurrence),
                    'all_quote_occurrences': deepcopy(claim['occurrences']),
                    'bindings': deepcopy(claim['bindings'])})
                if len(claim['occurrences']) != 1:
                    issues.add('repeated_auditor_quote_ambiguous')
                if not claim['bindings']:
                    issues.add('supported_claim_has_no_binding')
                for binding in claim['bindings']:
                    if binding['state'] == 'resolved':
                        value = binding['origin_class']
                        classes[pipeline.encoded(value)] = value
                    else:
                        issues.add(binding['reason'])
    if not result['matches']:
        issues.add('no_containing_supported_claim')
    result['origin_classes'] = [classes[k] for k in sorted(classes)]
    composition = 'mixed' if len(classes) > 1 else 'single' if classes else 'none'
    descriptor = 'unresolved' if issues else 'mixed_origins' if len(classes) > 1 else 'single_origin_class'
    return dict(result, descriptor=descriptor, origin_composition=composition,
                unresolved_reasons=sorted(issues))


def reference_entries(annotation):
    """Retain every evidence-list position; duplicate entries remain distinct."""
    for section in ('features', 'composites', 'auxiliary'):
        for name, item in annotation[section].items():
            if section == 'composites':
                for i, binding in enumerate(item['bindings']):
                    for j, entry in enumerate(binding['evidence']):
                        yield [section, name, 'bindings', i, 'evidence', j], entry
            else:
                for i, entry in enumerate(item['evidence']):
                    yield [section, name, 'evidence', i], entry


def narrative_attribution(primary_row, packet, provenance, report, audit, rating):
    """Primary labels/gates are copied intact, including failures and absent labels."""
    text = pipeline._report_text(report)
    audit_value = audit.get('parsed') if audit else None
    rating_value = rating.get('parsed') if rating else None
    prose_validation = pipeline.validate_prose_audit(packet, text, audit_value)
    rating_validation = pipeline.validate_blind_annotation(text, primary_row['unit_id'], rating_value)
    audit_usable = bool(audit and audit.get('transport_complete') and text.strip()
                        and prose_validation['valid'])
    rating_usable = bool(rating and rating.get('transport_complete') and text.strip()
                         and rating_validation['valid'])
    pipeline.require(primary_row['prose_audit_validation'] == prose_validation
                     and primary_row['rating_validation'] == rating_validation,
                     'primary validation differs from exact stage-response validation')
    claims = supported_claims(text, audit_value, packet, provenance) if audit_usable else []
    evidence = []
    if rating_usable:
        for path, entry in reference_entries(rating_value):
            evidence.append(dict(attribute_span(text, entry, claims, audit_usable), path=path))
    endpoints = {}
    for endpoint, features in PARAMETERS['endpoints'].items():
        indices = [i for i, row in enumerate(evidence)
                   if (row['path'][0] == 'features' and row['path'][1] in features)
                   or (row['path'][0] == 'composites' and row['path'][1] == endpoint)]
        endpoints[endpoint] = {'primary': deepcopy(primary_row['endpoints'][endpoint]),
            'reference_evidence_indices': indices,
            'evidence_descriptor_counts': dict(Counter(evidence[i]['descriptor'] for i in indices)),
            'empty_evidence_inventory_is_not_negative_consciousness_evidence': True}
    return {'id': primary_row['id'], 'primary_result': deepcopy(primary_row),
            'prose_audit_attribution_state': 'usable' if audit_usable else 'unestablished',
            'reference_attribution_state': 'usable' if rating_usable else 'unestablished',
            'supported_claims': claims, 'reference_evidence': evidence, 'endpoints': endpoints}


def validate_collected_records(collection):
    """Reparse raw first outcomes with the existing collector, without reading files."""
    pipeline.require(collection.get('schema') == 'attention_relational_report_v1_collected_first_responses'
                     and collection.get('retry_performed') is False, 'not a preserved first-response collection')
    requests = pipeline._index(collection['stage_binding']['requests'])
    pipeline.require(collection['request_count'] == len(requests), 'collection opportunity count differs')
    for identifier, record in pipeline._records(collection).items():
        pipeline.require(identifier in requests, 'foreign response identifier')
        raw = record.get('raw_outcome')
        if raw is None:
            pipeline.require(record.get('parsed') is None and record.get('transport_complete') is False,
                             'missing raw outcome cannot establish a response')
        else:
            outcome = pipeline._unwrap_outcome(raw)
            pipeline.require(outcome.get('request_sha256') == pipeline.runner_request_sha256(requests[identifier]),
                             'first outcome request hash differs')
            expected = pipeline._response_record(identifier, raw, record['storage_provenance'])
            pipeline.require(record == expected, 'collected response differs from raw first outcome')


def verify_primary(primary, recomputed):
    """Allow only the existing pipeline CLI envelope around an exactly equal summary."""
    metadata = {'preparation_dependencies', 'pipeline_source_sha256', 'no_api_or_model_calls'}
    extras = set(primary) - set(recomputed)
    pipeline.require(extras in (set(), metadata) and set(recomputed) <= set(primary)
                     and {k: primary[k] for k in recomputed} == recomputed,
                     'primary summary differs from exact frozen-pipeline recomputation')
    if extras:
        pipeline.require(primary['no_api_or_model_calls'] is True
                         and primary['pipeline_source_sha256'] == pipeline.digest(Path(pipeline.__file__).read_bytes()),
                         'primary summary pipeline source differs')


def build_attribution(exports, primary, reports, audits, ratings):
    for collection in (reports, audits, ratings):
        validate_collected_records(collection)
    # This recomputes all original 1104 ratings/552 prose audits, machine facts,
    # paired summaries and exact stage lineage. No primary result is replaced.
    recomputed = pipeline.summarize_reporting_results(reports, None, audits, ratings,
        pipeline.build_pair_index(exports['provenance']), exports)
    verify_primary(primary, recomputed)
    report_records, audit_records, rating_records = map(pipeline._records, (reports, audits, ratings))
    packets = pipeline._index(exports['packets'])
    provenance = pipeline._index(exports['provenance'])
    rows = []
    for row in recomputed['all_text_results']:
        if not row['template']:
            identifier = row['id']
            rows.append(narrative_attribution(row, packets[identifier]['content'], provenance[identifier],
                report_records.get(identifier), audit_records.get(identifier), rating_records.get(row['unit_id'])))
    pipeline.require(len(rows) == 552 and recomputed['opportunities']['narrator'] == 552,
                     'fixed552 narrator opportunity inventory differs')
    return {'schema': 'attention_relational_report_v1_descriptive_origin_attribution',
        'chronology': __doc__, 'parameters': deepcopy(PARAMETERS),
        'parameters_sha256': pipeline.digest(pipeline.encoded(PARAMETERS)),
        'primary_summary_exactly_recomputed': True,
        'changes_registered_endpoints_or_thresholds': False,
        'template_attribution_comparison_performed': False,
        'independent_observations_claimed': False,
        'opportunities': {'narrator': 552, 'endpoint_positions': 1104,
                          'primary': deepcopy(recomputed['opportunities'])},
        'endpoint_primary_status_counts': {endpoint: dict(Counter(
            row['endpoints'][endpoint]['primary']['status'] for row in rows))
            for endpoint in PARAMETERS['endpoints']},
        'reference_evidence_descriptor_counts': dict(Counter(
            entry['descriptor'] for row in rows for entry in row['reference_evidence'])),
        'reports': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('exports', 'primary_summary', 'reports', 'audits', 'ratings'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
        parser.add_argument('--' + name.replace('_', '-') + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'output exists; refusing replacement')
    pipeline.require(args.exports_sha256 == origins.EXPORT_SHA256, 'not the preserved first export hash')
    inputs = {}
    dependencies = {}
    for name in ('exports', 'primary_summary', 'reports', 'audits', 'ratings'):
        path, sha = getattr(args, name), getattr(args, name + '_sha256')
        inputs[name] = pipeline.bound_json(path, sha)
        dependencies[name] = {'path': str(path), 'sha256': sha}
    primary = inputs.pop('primary_summary')
    if 'preparation_dependencies' in primary:
        expected = {k: v for k, v in dependencies.items() if k != 'primary_summary'}
        saved = primary['preparation_dependencies']
        pipeline.require(set(saved) == set(expected) and all(
            saved[k]['sha256'] == v['sha256'] and Path(saved[k]['path']).resolve() == Path(v['path']).resolve()
            for k, v in expected.items()), 'primary summary bound input dependencies differ')
    result = build_attribution(primary=primary, **inputs)
    result['input_dependencies'] = dependencies
    result['source_sha256'] = {Path(module.__file__).name: pipeline.digest(Path(module.__file__).read_bytes())
                               for module in (pipeline, exporter, origins)}
    result['source_sha256'][Path(__file__).name] = pipeline.digest(Path(__file__).read_bytes())
    result['created_utc'] = datetime.now(timezone.utc).isoformat()
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
