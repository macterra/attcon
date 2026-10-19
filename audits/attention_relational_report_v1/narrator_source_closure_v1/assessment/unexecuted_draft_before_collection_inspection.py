"""Post-review accounting of the retained first narrator source judgments.

This is descriptive accounting, not another semantic grader. Preserve both first
judgments and their original statuses. In this observed corpus neither reviewer
proposes an eligible property identity, so paired correspondence is unestablished;
do not invent a predicate, convert a missing witness to invariance, or measure
word changes as a substitute. Stop if that observed premise does not hold.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt_path = BASE / 'collection/first_execution.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['exit_code'] == 0
    collection_path = Path(receipt['output']['path'])
    assert digest(collection_path) == receipt['output']['sha256']
    data = json.loads(collection_path.read_text())
    assert data['schema'] == 'attention_relational_narrator_source_closure_collection_v1'
    assert len(data['chunks']) == 2
    assert all(c['structurally_valid'] for c in data['chunks'])
    assert len(data['narrator_availability']) == data['narrator_opportunities'] == 552
    assert len(data['pairs']) == data['pair_opportunities'] == 684
    assert data['narrator_fidelity_established'] is False
    assert data['fidelity_conjoined_witness_established'] is False
    streams = ('reviewer_0', 'reviewer_1')
    reviews = {s: {} for s in streams}
    for row in data['rows']:
        assert row['chunk_structurally_valid']
        assert row['review_state'] == 'retained_structurally_valid_judgment'
        judgment = row['first_judgment']
        assert row['id'] not in reviews[row['reviewer_id']]
        reviews[row['reviewer_id']][row['id']] = judgment
        for binding in judgment['candidate_binding_reviews']:
            assert binding['tracking_support']['eligible'] == 'no'
            assert binding['tracking_support']['property_identity'] is None
    assert set(reviews[streams[0]]) == set(reviews[streams[1]])
    assert len(reviews[streams[0]]) == data['candidate_narrator_positions'] == 3
    positions = []
    for available in data['narrator_availability']:
        original_id = available['id']
        by_stream = {}
        for stream in streams:
            judgment = reviews[stream].get(original_id)
            if judgment is None:
                continue
            by_stream[stream] = [dict(
                binding_path=b['binding_path'], original_status=b['original_status'],
                recorded_feature_grounding=b['recorded_feature_grounding'],
                features=[{k: f[k] for k in ('feature', 'original_status', 'grounding')}
                          for f in b['feature_reviews']],
                source_meaning=b['source_meaning'], source_ownership=b['source_ownership'],
                tracking_support=b['tracking_support'],
                unresolved_requirements=b['unresolved_requirements'])
                for b in judgment['candidate_binding_reviews']]
        if by_stream:
            disposition = 'candidate_reviewed_no_eligible_source_witness'
        elif available['reference_established']:
            disposition = 'established_reference_without_original_candidate_binding'
        else:
            disposition = 'reference_unestablished_no_candidate_salvaged'
        positions.append(dict(id=original_id, case=available['case'],
            actor_text_available=available['actor_text_available'],
            reference_established=available['reference_established'],
            disposition=disposition, first_stream_judgments=by_stream,
            source_witness_established=False, narrator_fidelity_established=False))
    indexed = {p['id']: p for p in positions}
    assert len(indexed) == 552
    pairs = []
    for pair in data['pairs']:
        first, second = indexed[pair['first_id']], indexed[pair['second_id']]
        pairs.append(dict(original_pair=pair,
            first_disposition=first['disposition'], second_disposition=second['disposition'],
            property_correspondence='unestablished_no_eligible_source_witness',
            actual_property_identity=None, actual_property_change=None,
            asserted_predicate_change=None, selective_tracking=None,
            true_null=None, unaffected_assertions_preserved=None,
            exact_restoration=None,
            reason='Neither endpoint has an eligible full-binding source witness. '
                   'No property or predicate is invented from coordinates, provenance, '
                   'a changed label or missing measurement. Native source restoration '
                   'and lower-level numerical checks are separate evidence.'))
    def counts(key):
        return dict(sorted(Counter(p[key] for p in positions).items()))
    summary = dict(narrator_opportunities=len(positions),
        reference_established=sum(p['reference_established'] for p in positions),
        actor_text_available=sum(p['actor_text_available'] for p in positions),
        position_dispositions=counts('disposition'),
        candidate_narrator_positions=data['candidate_narrator_positions'],
        original_candidate_bindings=len(data['candidate_binding_inventory']),
        first_review_chunks=2, structurally_valid_first_chunks=2,
        source_witness_established=0, fidelity_conjoined_witness_established=0,
        pair_opportunities=len(pairs), paired_correspondence_established=0,
        paired_correspondence_unestablished=len(pairs),
        eligible_property_identity_count=0,
        true_null_count=None, exact_restoration_count=None,
        property_values_or_predicates_compared=False)
    for dimension in ('source', 'encoder_seed', 'world_index'):
        summary['by_' + dimension] = {
            str(value): dict(Counter(p['disposition'] for p in positions
                                    if p['case'][dimension] == value))
            for value in sorted({p['case'][dimension] for p in positions})}
    output = dict(schema='attention_relational_narrator_source_closure_v1_first_readout',
        analysis_type='post_review_descriptive_accounting_no_semantic_regrading',
        input_collection=receipt['output'],
        input_execution_receipt=dict(path=str(receipt_path.relative_to(Path.cwd())),
                                     sha256=digest(receipt_path)),
        analysis_source=dict(path=str(Path(__file__).resolve().relative_to(Path.cwd())),
                             sha256=digest(Path(__file__))),
        summary=summary, positions=positions, pairs=pairs,
        semantic_correctness_established_by_code=False,
        original_v3_labels_changed=False, source_or_report_properties_repaired=False,
        template_analysis_complete=False, goal_achieved=False,
        limitations=[
            'Two fresh AI contexts share training and commissioning; agreement is '
            'not semantic infallibility or independent-system replication.',
            'Original source-aware judgments and exact operands/calculations are '
            'retained in the collection; this readout does not re-grade them.',
            'Unestablished references remain unknown, not absent features.',
            'No eligible full-binding property identity exists in these first '
            'judgments. All 684 report-pair correspondences are unestablished, '
            'not evidence of invariance or failed native restoration.',
            'Lower-level spatial organization and source-fact accuracy do not '
            'establish the complete consciousness-related binding.',
            'Ordinary also models attention effects; its source identity is not '
            'a reason to reject a witness or an attention-model-absence control.',
            'Whole-prose fidelity remains unestablished for all 552 positions. '
            'No claimed-appearance or felt-experience conclusion is established.',
            'The complete fixed template review remains in progress. No successor '
            'auditor, new Reporter run or theory-confirmation experiment is released.'])
    target = BASE / 'assessment/first_readout.json'
    with target.open('x') as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
