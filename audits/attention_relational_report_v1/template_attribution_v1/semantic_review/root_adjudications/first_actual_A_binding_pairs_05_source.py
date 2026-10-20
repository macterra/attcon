"""Diagnostic trace of one existing binding through all nine of its case pairs.

This is exposed-author analysis of archived development data. It neither selects
a new binding nor adjudicates semantic endpoints. No production imports or calls.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
INPUTS = {
    'audits/attention_relational_report_v1/template_attribution_v1/property_pairs/first_pairs.json':
        'c944d239cd52863fefa9331fc91800d048362d38d2fd7ee2659dc3b41d5706a5',
    'audits/attention_relational_report_v1/run/export/first_exports.json':
        '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837',
    'audits/attention_relational_report_v1/template_attribution_v1/semantic_review/root_adjudications/first_actual_A_binding_case_05.json':
        '7029c0871adf403651ef1b23b689297a932a7ade9ec15e29a0bc8f01a10020ba',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def maxima(grid):
    assert len(grid) == 4 and all(len(row) == 4 for row in grid)
    high = max(value for row in grid for value in row)
    return [[i, j] for i, row in enumerate(grid)
            for j, value in enumerate(row) if value == high]


def main():
    loaded = {}
    for relative, expected in INPUTS.items():
        raw = (ROOT / relative).read_bytes()
        assert sha(raw) == expected, relative
        loaded[relative] = json.loads(raw)
    bank, exports, disposition = loaded.values()
    baseline = disposition['id'].removesuffix('_template')
    assert baseline == 's1301_w02_t12_A_joined_baseline'
    selected = [pair for pair in bank['pairs']
                if pair['original_pair']['case'] == [1301, 2, 'A', 'joined']]
    assert len(selected) == 9
    assert {p['original_pair']['kind'] for p in selected} == {
        'current_donor', 'current_donor_to_restored', 'current_baseline_to_restored',
        'prospective_donor', 'prospective_donor_to_restored',
        'prospective_baseline_to_restored', 'W_changed', 'W_changed_to_restored',
        'W_baseline_to_restored'}
    ids = sorted({p['original_pair'][side] for p in selected
                  for side in ('first_id', 'second_id')})
    assert len(ids) == 7
    by_field = {field: {item['id']: item for item in exports[field]}
                for field in ('packets', 'provenance', 'templates')}
    table = bank['property_tables'][baseline]
    found = [(key, prop) for key, prop in table.items()
             if prop['construction_unit_id'] == 'u0022'
             and prop['identity']['value_path'] == []]
    assert len(found) == 1
    property_key, original = found[0]
    identity = original['identity']
    assert identity['actual_source'] == 'A'
    assert identity['context'] == dict(command=None, node='n3', panel='p0',
                                        role='red', view='current')
    assert identity['origin_category'] == 'source_current'
    assert sha(encoded(identity)) == property_key
    expected_path = ['panels', 0, 'current', 3, 'distributions', 'red']
    variants = {}
    for source_id in ids:
        packet_record = by_field['packets'][source_id]
        packet = packet_record['content']
        assert sha(encoded(packet)) == packet_record['content_sha256']
        provenance = by_field['provenance'][source_id]
        assert [p['source_key'] for p in provenance['panels']] == ['A', 'W']
        a, w = packet['panels']
        node = a['current'][3]
        assert node['node'] == 'n3'
        assert node['origin'] == dict(feedback_through=11,
                                      kind='history_estimate', represented_time=12)
        grid = node['distributions']['red']
        peaks = maxima(grid)
        assert peaks == node['derived']['red']['peaks']
        prop = bank['property_tables'][source_id][property_key]
        assert prop['identity'] == identity
        assert prop['raw_distribution_packet_path'] == expected_path
        assert prop['value'] == peaks
        assert prop['raw_distribution_content_sha256'] == sha(encoded(grid))
        text = by_field['templates'][source_id]['report']['text']
        clauses = [token for token in text.split(' ')
                   if token.startswith('p0/current/n3/red=')]
        assert len(clauses) == 1
        expected_clause = 'p0/current/n3/red={' + ','.join(
            f'({i},{j})' for i, j in peaks) + '}'
        assert clauses[0] == expected_clause
        variants[source_id] = dict(
            case=provenance['case'], complete_text_sha256=sha(text.encode('utf-8')),
            literal_A_red_clause=clauses[0], a_origin=node['origin'],
            a_raw_distribution=grid, a_exact_all_maxima=peaks,
            a_raw_distribution_content_sha256=sha(encoded(grid)),
            exported_comparison_W_panel_content_sha256=sha(encoded(w)),
            comparison_W_current_n3_red_all_maxima=maxima(
                w['current'][3]['distributions']['red']),
            literal_A_peak_readout_matches=True,
            endpoint_semantic_review_or_scope_upgrade_performed=False)
    rows = []
    for pair in selected:
        first, second = [variants[pair['original_pair'][side]]
                         for side in ('first_id', 'second_id')]
        value_equal = first['a_exact_all_maxima'] == second['a_exact_all_maxima']
        raw_equal = first['a_raw_distribution'] == second['a_raw_distribution']
        clause_equal = first['literal_A_red_clause'] == second['literal_A_red_clause']
        bank_rows = [r for r in pair['properties']
                     if r['property_identity_sha256'] == property_key]
        assert len(bank_rows) == 1
        bank_row = bank_rows[0]
        assert bank_row['state'] == 'matched'
        assert bank_row['value_unchanged'] == value_equal
        assert bank_row['raw_distribution_unchanged'] == raw_equal
        assert bank_row['peak_null_with_raw_distribution_change'] == (value_equal and not raw_equal)
        assert clause_equal == value_equal
        rows.append(dict(
            original_pair=deepcopy(pair['original_pair']),
            a_red_peak_equal=value_equal, a_red_raw_distribution_equal=raw_equal,
            a_red_clause_equal=clause_equal,
            true_peak_null_with_changed_raw_distribution=value_equal and not raw_equal,
            exported_comparison_W_panel_equal=(
                first['exported_comparison_W_panel_content_sha256'] ==
                second['exported_comparison_W_panel_content_sha256']),
            original_bank_property_comparison=deepcopy(bank_row),
            original_bank_W_property_counts=deepcopy(pair['by_actual_source']['W']),
            original_bank_own_baseline_restoration=deepcopy(pair['restoration_to_own_baseline']),
            consciousness_related_endpoint_tracking=None))
    result = dict(
        schema='attention_relational_root_actual_A_binding_case_pairs_v1',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=INPUTS, source_sha256=sha(Path(__file__).read_bytes()),
        selection='All nine fixed pairs for the previously selected chunk05 actual-A binding case; exposed development-data diagnostic, no replacement binding.',
        original_binding=deepcopy(disposition['original_binding']),
        property_identity_sha256=property_key, property_identity=identity,
        variants=variants, pairs=rows,
        original_binding_O_D_meaning='unresolved',
        full_corpus_semantic_analysis_performed=False,
        new_independent_observations=False, original_judgments_changed=False,
        feedback_to_unfinished_first_contexts=False,
        qualia_source_conclusion_established=False,
        limitations=[
            'One previously selected authored-template property, not a population estimate or neutral narrator result.',
            'Peak accuracy concerns the A estimate, not its truth about the external world or complete latent-state content.',
            'Unchanged peaks despite changed raw distributions are genuine peak-property nulls, not evidence that the intervention was ignored.',
            'W source/time transitions remain unmatched in the complete property bank; a panel hash change does not license semantic identity matching.',
            'No new semantic binding is inferred for the other six variants; complete fixed source reviews remain necessary.',
            'Literal renderer correspondence alone does not identify an experiencing subsystem.'])
    for relative, expected in INPUTS.items():
        assert sha((ROOT / relative).read_bytes()) == expected
    output = HERE / 'first_actual_A_binding_pairs_05.json'
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(variants=len(variants), pairs=len(rows),
        changed_A_peak_pairs=sum(not r['a_red_peak_equal'] for r in rows),
        true_peak_null_raw_change_pairs=sum(r['true_peak_null_with_changed_raw_distribution'] for r in rows),
        semantic_endpoint_assessment=False)))


if __name__ == '__main__':
    main()
