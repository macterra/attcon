"""Prospective quote-only reference contract; pure, inert, and not an API runner.

No old annotation is repaired, rescored, or accepted as a raw V3 annotation.
Canonical offsets serve only the unchanged frozen semantic evaluator. Actor
availability and actual V3 request/collection lineage require a later layer.
"""
from copy import deepcopy

import attention_relational_reporting_pipeline_v1 as pipeline


QUOTE_SCHEMA = 'attention_relational_reference_v3_quote_annotation'
EVIDENCE_KEYS = {'quote', 'referent', 'attribute', 'scope', 'time'}


def response_schema():
    """Return an independent copy with only the declared quote mechanics changed."""
    schema = deepcopy(pipeline.RATER_RESPONSE_SCHEMA)
    props = schema['properties']
    props['schema']['enum'] = [QUOTE_SCHEMA]
    evidence_schemas = [item['properties']['evidence']['items']
                        for section in ('features', 'auxiliary')
                        for item in props[section]['properties'].values()]
    evidence_schemas += [item['properties']['bindings']['items']['properties']['evidence']['items']
                         for item in props['composites']['properties'].values()]
    for item in evidence_schemas:
        for name in ('start', 'end'):
            item['properties'].pop(name, None)
        item['required'] = [name for name in item['required'] if name not in ('start', 'end')]
    return schema


def exact_occurrences(text, quote):
    """All literal Unicode-code-point occurrences, including overlapping matches."""
    matches = []
    if type(text) is not str or type(quote) is not str or not quote:
        return matches
    start = text.find(quote)
    while start >= 0:
        matches.append({'start': start, 'end': start + len(quote)})
        start = text.find(quote, start + 1)
    return matches


def derive_annotation(text, unit_id, raw, *, expected_text_sha256):
    """Ground every quote or establish no annotation; never salvage feature labels.

The raw-object hash uses pipeline.encoded, not original provider JSON bytes.
No normalization, approximate match, or first-occurrence selection is allowed.
Malformed or semantically invalid input retains diagnostics and no canonical
annotation. The caller retains the original raw object, which is never mutated.
"""
    errors, diagnostics, assignments = [], [], []
    text_sha = pipeline.digest(text.encode('utf-8')) if type(text) is str else None
    try:
        raw_sha = pipeline.digest(pipeline.encoded(raw))
    except (TypeError, ValueError):
        raw_sha = None
        errors.append('raw_annotation_is_not_finite_JSON')
    if type(text) is not str:
        errors.append('text_is_not_a_string')
    if (type(expected_text_sha256) is not str or len(expected_text_sha256) != 64
            or expected_text_sha256 != text_sha):
        errors.append('exact_text_SHA256_mismatch')
    if type(unit_id) is not str or not unit_id:
        errors.append('invalid_expected_unit_id')
    if type(raw) is not dict or raw.get('schema') != QUOTE_SCHEMA:
        errors.append('raw_V3_quote_schema_required')
    if type(raw) is not dict or raw.get('unit_id') != unit_id:
        errors.append('raw_unit_id_mismatch')
    candidate = deepcopy(raw)

    def evidence_entries(entries, path):
        if type(entries) is not list:
            diagnostics.append({'path': path, 'error': 'evidence_is_not_a_list', 'occurrences': []})
            errors.append('malformed_evidence')
            return
        for i, entry in enumerate(entries):
            quote = entry.get('quote') if type(entry) is dict else None
            matches = exact_occurrences(text, quote)
            issues = []
            if type(entry) is not dict or set(entry) != EVIDENCE_KEYS:
                issues.append('malformed_evidence_entry')
            if type(quote) is not str:
                issues.append('quote_is_not_a_string')
            elif not quote:
                issues.append('empty_quote')
            elif len(matches) != 1:
                issues.append('quote_absent' if not matches else 'quote_has_multiple_occurrences')
            diagnostics.append({'path': path + [i], 'quote': quote, 'occurrences': matches,
                                'occurrence_count': len(matches), 'errors': issues})
            if issues:
                errors.append('evidence_grounding_failed')
            else:
                # Exactly one match is required; no favorable occurrence is chosen.
                assignments.append((entry, matches[0]))

    def walk(value, path):
        if type(value) is dict:
            for name, child in value.items():
                if name == 'evidence':
                    evidence_entries(child, path + [name])
                walk(child, path + [name])
        elif type(value) is list:
            for i, child in enumerate(value):
                walk(child, path + [i])

    # A finite JSON object has no cycles; invalid non-JSON input is not traversed.
    if raw_sha is not None:
        walk(candidate, [])
    for entry, offsets in assignments:
        entry.update(offsets)
    semantic = None
    if not errors:
        candidate['schema'] = pipeline.RATING_SCHEMA
        semantic = pipeline.validate_blind_annotation(text, unit_id, candidate)
        if not semantic['valid']:
            errors.append('frozen_semantic_validation_failed')
    return {'schema': 'attention_relational_reference_v3_quote_derivation',
            'raw_annotation_sha256': raw_sha, 'raw_hash_encoding': 'pipeline.encoded(raw)',
            'text_sha256': text_sha, 'expected_text_sha256': expected_text_sha256, 'unit_id': unit_id,
            'quote_diagnostics': diagnostics, 'errors': errors,
            'canonical_annotation': candidate if not errors else None,
            'semantic_validation': semantic, 'valid': not errors,
            'actor_availability_assessed': False, 'actual_request_lineage_verified': False}
