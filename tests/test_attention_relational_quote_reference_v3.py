"""Prospective handcrafted quote-contract fixtures only; no scientific outputs.

Fictional annotations below test mechanics and frozen validator compatibility,
not human reference qualification, actual reference judgments, or experience.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_quote_reference_v3 as quotes
import attention_relational_reporting_pipeline_v1 as pipeline


TEXT = 'Préface. Réd is left.'
UNIT = 'fictional-quote-fixture'


def raw_annotation(quote='Réd is left.'):
    entry = {'quote': quote, 'referent': 'fictional red record', 'attribute': 'location',
             'scope': 'model_organization', 'time': 'current'}
    binding = {k: entry[k] for k in ('referent', 'attribute', 'scope', 'time')}
    binding['evidence'] = [deepcopy(entry)]
    value = {'schema': quotes.QUOTE_SCHEMA, 'unit_id': UNIT,
        'features': {k: {'status': 'explicit', 'evidence': [deepcopy(entry)],
                        'rationale': 'Fictional schema fixture.'} for k in ('C', 'O', 'D', 'A')},
        'composites': {k: {'status': 'explicit', 'bindings': [deepcopy(binding)],
                          'rationale': 'Fictional schema fixture.'} for k in ('COD', 'CA')},
        'auxiliary': {k: {'status': 'absent', 'evidence': []} for k in pipeline.AUXILIARY},
        'limitations': ['Handwritten software fixture; no scientific annotation.']}
    value['auxiliary']['basic_spatial_relation'] = {'status': 'ambiguous', 'evidence': [deepcopy(entry)]}
    return value


def derive(raw=None, text=TEXT, **kwargs):
    return quotes.derive_annotation(text, UNIT, raw_annotation() if raw is None else raw,
                                   expected_text_sha256=kwargs.get('sha', pipeline.digest(text.encode('utf-8'))))


def evidence_lists(value):
    if type(value) is dict:
        for key, child in value.items():
            if key == 'evidence':
                yield child
            yield from evidence_lists(child)
    elif type(value) is list:
        for child in value:
            yield from evidence_lists(child)


def test_schema_changes_only_marker_and_evidence_offset_mechanics():
    original = deepcopy(pipeline.RATER_RESPONSE_SCHEMA)
    expected = deepcopy(original)
    expected['properties']['schema']['enum'] = [quotes.QUOTE_SCHEMA]

    def remove_offsets(value):
        if type(value) is dict:
            props = value.get('properties', {})
            if set(props) == {'quote', 'start', 'end', 'referent', 'attribute', 'scope', 'time'}:
                del props['start'], props['end']
                value['required'] = ['quote', 'referent', 'attribute', 'scope', 'time']
            for child in value.values():
                remove_offsets(child)
        elif type(value) is list:
            for child in value:
                remove_offsets(child)

    remove_offsets(expected)
    actual = quotes.response_schema()
    assert actual == expected
    assert pipeline.RATER_RESPONSE_SCHEMA == original
    # Every schema call is independent of both frozen schema and other callers.
    actual['properties']['features']['properties']['C']['properties']['status']['enum'].append('invented')
    assert quotes.response_schema() == expected
    assert pipeline.RATER_RESPONSE_SCHEMA == original


def test_unique_unicode_quotes_match_manual_codepoint_offsets_and_drop_no_fields():
    raw = raw_annotation()
    before = deepcopy(raw)
    result = derive(raw)
    assert result['valid'] and result['semantic_validation']['valid']
    assert raw == before
    expected = deepcopy(raw)
    expected['schema'] = pipeline.RATING_SCHEMA
    for entries in evidence_lists(expected):
        for entry in entries:
            entry.update(start=9, end=21)  # Hand-counted Unicode positions, not UTF-8 byte offsets.
    assert result['canonical_annotation'] == expected
    assert len(TEXT.encode('utf-8')) > 21
    assert len(result['quote_diagnostics']) == 7
    assert all(d['occurrences'] == [{'start': 9, 'end': 21}] for d in result['quote_diagnostics'])
    assert {d['path'][0] for d in result['quote_diagnostics']} == {'features', 'composites', 'auxiliary'}
    assert result['raw_annotation_sha256'] == pipeline.digest(pipeline.encoded(raw))
    assert result['text_sha256'] == pipeline.digest(TEXT.encode('utf-8'))
    result['canonical_annotation']['features']['C']['rationale'] = 'Changed returned copy.'
    assert raw == before


def test_repeated_and_overlapping_quotes_retain_every_occurrence_without_selection():
    for text, quote, matches in [
        ('Réd is left. Réd is left.', 'Réd is left.', [{'start': 0, 'end': 12}, {'start': 13, 'end': 25}]),
        ('aaa', 'aa', [{'start': 0, 'end': 2}, {'start': 1, 'end': 3}]),
    ]:
        result = derive(raw_annotation(quote), text=text)
        assert not result['valid'] and result['canonical_annotation'] is None
        assert result['semantic_validation'] is None
        assert len(result['quote_diagnostics']) == 7
        assert all(d['occurrences'] == matches for d in result['quote_diagnostics'])
        assert all('quote_has_multiple_occurrences' in d['errors'] for d in result['quote_diagnostics'])


def test_multiple_distinct_evidence_entries_preserve_order_and_attributes():
    text = TEXT + ' Blue is right.'
    raw = raw_annotation()
    extra = {'quote': 'Blue is right.', 'referent': 'fictional blue record', 'attribute': 'location',
             'scope': 'model_organization', 'time': 'current'}
    raw['features']['C']['evidence'].append(extra)
    result = derive(raw, text=text)
    assert result['valid']
    actual = result['canonical_annotation']['features']['C']['evidence']
    assert actual == [dict(raw['features']['C']['evidence'][0], start=9, end=21),
                      dict(extra, start=22, end=36)]
    assert result['quote_diagnostics'][1]['path'] == ['features', 'C', 'evidence', 1]


@pytest.mark.parametrize('bad_quote, error', [
    ('missing', 'quote_absent'), ('', 'empty_quote'), (None, 'quote_is_not_a_string'),
    (7, 'quote_is_not_a_string'), (['Réd is left.'], 'quote_is_not_a_string'),
])
def test_one_bad_quote_invalidates_whole_annotation_but_retains_all_diagnostics(bad_quote, error):
    raw = raw_annotation()
    raw['features']['C']['evidence'][0]['quote'] = bad_quote
    before = deepcopy(raw)
    result = derive(raw)
    assert not result['valid'] and result['canonical_annotation'] is None
    assert result['semantic_validation'] is None and raw == before
    assert len(result['quote_diagnostics']) == 7
    assert error in result['quote_diagnostics'][0]['errors']
    assert result['quote_diagnostics'][0]['occurrences'] == []
    assert all(d['occurrences'] == [{'start': 9, 'end': 21}] for d in result['quote_diagnostics'][1:])


@pytest.mark.parametrize('quote', ['réd is left.', 'Réd  is left.', 'Re\u0301d is left.'])
def test_no_case_whitespace_or_unicode_normalization(quote):
    result = derive(raw_annotation(quote))
    assert not result['valid'] and result['canonical_annotation'] is None
    assert all(d['occurrences'] == [] for d in result['quote_diagnostics'])


def test_unit_text_hash_and_raw_schema_are_mandatory_without_old_format_fallback():
    bad_unit = raw_annotation()
    bad_unit['unit_id'] = 'other-unit'
    old_schema = raw_annotation()
    old_schema['schema'] = pipeline.RATING_SCHEMA
    for result, error in [(derive(bad_unit), 'raw_unit_id_mismatch'),
                          (derive(sha='0' * 64), 'exact_text_SHA256_mismatch'),
                          (derive(old_schema), 'raw_V3_quote_schema_required')]:
        assert not result['valid'] and result['canonical_annotation'] is None
        assert error in result['errors'] and len(result['quote_diagnostics']) == 7
    with pytest.raises(TypeError):
        quotes.derive_annotation(TEXT, UNIT, raw_annotation())


@pytest.mark.parametrize('malformation', ['offsets', 'missing_field', 'extra_field', 'nonobject', 'nonlist'])
def test_malformed_evidence_never_salvages_other_features(malformation):
    raw = raw_annotation()
    entries = raw['features']['C']['evidence']
    if malformation == 'offsets':
        entries[0].update(start=9, end=21)  # Even correct legacy offsets are forbidden raw V3 fields.
    elif malformation == 'missing_field':
        del entries[0]['attribute']
    elif malformation == 'extra_field':
        entries[0]['invented'] = True
    elif malformation == 'nonobject':
        entries[0] = None
    else:
        raw['features']['C']['evidence'] = 'Réd is left.'
    result = derive(raw)
    assert not result['valid'] and result['canonical_annotation'] is None
    assert result['semantic_validation'] is None
    assert len(result['quote_diagnostics']) == 7


def test_unique_grounding_does_not_change_composite_compatibility_or_status_labels():
    raw = raw_annotation()
    raw['features']['A']['evidence'][0]['time'] = 'future_forecast'
    before = deepcopy(raw)
    result = derive(raw)
    assert not result['valid'] and result['canonical_annotation'] is None
    assert result['semantic_validation']['valid'] is False
    assert any('compatible feature evidence for A' in e for e in result['semantic_validation']['errors'])
    assert all(d['occurrence_count'] == 1 for d in result['quote_diagnostics'])
    assert raw == before and raw['composites']['CA']['status'] == 'explicit'


def test_empty_evidence_keeps_frozen_semantics_without_actor_availability_claim():
    raw = raw_annotation()
    for feature in raw['features'].values():
        feature.update(status='absent', evidence=[])
    for auxiliary in raw['auxiliary'].values():
        auxiliary.update(status='absent', evidence=[])
    for composite in raw['composites'].values():
        composite.update(status='absent', bindings=[])
    result = derive(raw, text='')
    assert result['valid'] and result['quote_diagnostics'] == []
    assert not result['actor_availability_assessed'] and not result['actual_request_lineage_verified']
    raw['features']['C']['status'] = 'explicit'
    invalid = derive(raw, text='')
    assert not invalid['valid'] and invalid['semantic_validation']['valid'] is False
    assert invalid['canonical_annotation'] is None


def test_shared_in_memory_entries_do_not_change_the_json_quote_contract():
    raw = raw_annotation()
    shared = raw['features']['C']['evidence'][0]
    raw['features']['O']['evidence'][0] = shared
    before = deepcopy(raw)
    result = derive(raw)
    assert result['valid'] and len(result['quote_diagnostics']) == 7
    assert raw == before and 'start' not in shared
