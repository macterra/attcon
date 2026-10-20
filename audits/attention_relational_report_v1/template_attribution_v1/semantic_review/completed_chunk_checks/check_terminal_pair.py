"""Preserve a terminal pair containing a missing first artifact.

Use the unchanged frozen collector's whole-chunk failure rule. This adds no
replacement judgment or semantic grade. The earlier successful-pair helpers
remain unchanged and their historical source hashes remain available.
"""
import argparse
import json
from pathlib import Path

from check_pair import ROOT, bound, collector, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk', type=int, required=True, choices=range(23))
    args = parser.parse_args()
    base = Path(__file__).parent
    targets = [base / f'first_pair_{args.chunk:02d}.json',
               base / f'first_pair_{args.chunk:02d}_differences.json']
    assert all(not p.exists() for p in targets), 'no replacement of first checks'
    release = ROOT / 'audits/attention_relational_report_v1/source_review_release_v1'
    assignments = bound(str((release / 'first_assignments.json').relative_to(ROOT)),
        '1bb582594d6c757deb82bb6058d55a51b1789ea56775f378fc53fc93fd787148')['assignments']
    selected = [a for a in assignments
                if a['kind'] == 'template' and a['chunk_number'] == args.chunk]
    assert len(selected) == 2 and {a['reviewer_id'] for a in selected} == {'reviewer_0', 'reviewer_1'}
    checks, dependencies, originals = [], {}, None
    for item in selected:
        chunk = bound(item['chunk']['path'], item['chunk']['sha256'])
        if originals is None:
            originals = chunk['records']
        else:
            assert originals == chunk['records']
        envelope = bound(item['envelope']['path'], item['envelope']['sha256'])
        schema = bound(item['output_schema']['path'], item['output_schema']['sha256'])
        observation_path = release / 'terminal_observations' / (item['task_name'] + '.json')
        observation = json.loads(observation_path.read_text())
        assert observation['observed_terminal'] is True
        assert observation['independent_context_id'] == item['independent_context_id']
        assert observation['chunk_sha256'] == item['chunk']['sha256']
        assert observation['observation_kind'] in ('completed', 'failed', 'interrupted')
        observation_ref = dict(path=str(observation_path.relative_to(ROOT)), sha256=digest(observation_path))
        dependencies[observation_ref['path']] = observation_ref['sha256']
        artifact = observation['first_artifact']
        if artifact is None:
            assert not (ROOT / item['first_artifact_path']).exists(), 'null receipt conflicts with present reserved first'
            raw = None
        else:
            assert artifact['path'] == item['first_artifact_path']
            path = ROOT / artifact['path']
            assert digest(path) == artifact['sha256']
            dependencies[artifact['path']] = artifact['sha256']
            raw = path.read_bytes()
        result = collector.validate_chunk(raw, chunk, envelope, schema)
        checks.append(dict(reviewer_id=item['reviewer_id'],
            independent_context_id=item['independent_context_id'], chunk=item['chunk'],
            first_artifact=artifact, reserved_first_artifact_path=item['first_artifact_path'],
            terminal_observation=observation_ref,
            structurally_valid=result['structurally_valid'],
            parse_errors=result['parse_errors'], schema_errors=result['schema_errors'],
            contract_errors=result['contract_errors'],
            record_checks=[{k: v for k, v in c.items() if k != 'property_references'}
                           for c in result['record_checks']],
            semantic_correctness_established=False))
    assert any(c['first_artifact'] is None for c in checks), 'use ordinary helper for two retained firsts'
    assert len(originals) == 24
    assert not all(c['structurally_valid'] for c in checks)
    source_sha = digest(Path(__file__))
    result = dict(schema='attention_relational_template_completed_pair_contract_check_v1',
        chunk_number=args.chunk, checks=checks,
        checker_source_sha256=digest(ROOT / 'scripts/attention_relational_template_review_collection_v1.py'),
        dispatch_source_sha256=source_sha,
        full46context_collection_performed=False,
        semantic_endpoint_assessment_performed=False, feedback_sent_to_first_reviewers=False,
        no_replacement_or_gap_filling=True)
    with targets[0].open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    dependencies[str(targets[0].relative_to(ROOT))] = digest(targets[0])
    comparison = dict(schema='attention_relational_completed_first_pair_explicit_differences_v1',
        chunk_number=args.chunk, input_sha256=dependencies,
        indexing_source_sha256=source_sha, contract_dispatch_source_sha256=source_sha,
        both_first_chunks_structurally_valid=False,
        records=[dict(id=r['id'], reference_established=r['reference_established'],
            original_statuses_unchanged=True,
            review_comparison_state='unestablished_whole_chunk',
            explicit_label_differences=[], property_inventory_differences=[],
            reason='One terminal context has no first artifact. No semantic comparison or salvage from the other stream.')
            for r in originals],
        semantic_adjudication_performed=False, endpoint_computed=False,
        feedback_sent_to_first_reviewers=False,
        equal_labels_do_not_establish_semantic_agreement=True,
        operand_inventory_difference_does_not_alone_establish_semantic_disagreement=True,
        entire_corpus_analysis_complete=False,
        missing_judgments_are_unknown_not_feature_negatives=True)
    with targets[1].open('x') as handle:
        json.dump(comparison, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(chunk=args.chunk,
        validity={c['reviewer_id']: c['structurally_valid'] for c in checks},
        compared_semantic_positions=0, retained_unestablished_pair_positions=len(originals))))


if __name__ == '__main__':
    main()
