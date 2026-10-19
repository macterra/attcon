"""Root contract checks of two terminal first contexts; no semantic grading.

This uses the unchanged frozen checker on a completed chunk pair. It does not
perform the full 46-context production collection or deliver repair feedback.
Full first artifacts remain retained at their reserved paths. Bad chunks remain
unestablished in full, with no row salvage or replacement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'scripts'))
import attention_relational_template_review_collection_v1 as collector


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(path, expected):
    path = ROOT / path
    assert digest(path) == expected, path
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk', type=int, required=True, choices=range(23))
    args = parser.parse_args()
    release = ROOT / 'audits/attention_relational_report_v1/source_review_release_v1'
    assignment_path = release / 'first_assignments.json'
    assignments = bound(assignment_path,
        '1bb582594d6c757deb82bb6058d55a51b1789ea56775f378fc53fc93fd787148')['assignments']
    selected = [a for a in assignments if a['kind'] == 'template' and a['chunk_number'] == args.chunk]
    assert len(selected) == 2 and {a['reviewer_id'] for a in selected} == {'reviewer_0', 'reviewer_1'}
    checks = []
    for item in selected:
        chunk = bound(item['chunk']['path'], item['chunk']['sha256'])
        envelope = bound(item['envelope']['path'], item['envelope']['sha256'])
        schema = bound(item['output_schema']['path'], item['output_schema']['sha256'])
        observation_path = release / 'terminal_observations' / (item['task_name'] + '.json')
        observation = json.loads(observation_path.read_text())
        assert observation['observed_terminal'] is True
        assert observation['independent_context_id'] == item['independent_context_id']
        assert observation['chunk_sha256'] == item['chunk']['sha256']
        artifact = ROOT / item['first_artifact_path']
        assert observation['first_artifact']['path'] == item['first_artifact_path']
        assert digest(artifact) == observation['first_artifact']['sha256']
        result = collector.validate_chunk(artifact.read_bytes(), chunk, envelope, schema)
        # Refer to immutable first bytes rather than duplicating their large
        # inventories. The complete production collector will retain them again.
        checks.append(dict(reviewer_id=item['reviewer_id'],
            independent_context_id=item['independent_context_id'],
            chunk=item['chunk'], first_artifact=observation['first_artifact'],
            terminal_observation=dict(path=str(observation_path.relative_to(ROOT)),
                                      sha256=digest(observation_path)),
            structurally_valid=result['structurally_valid'],
            parse_errors=result['parse_errors'], schema_errors=result['schema_errors'],
            contract_errors=result['contract_errors'],
            record_checks=[{k: v for k, v in check.items() if k != 'property_references'}
                           for check in result['record_checks']],
            semantic_correctness_established=False))
    result = dict(schema='attention_relational_template_completed_pair_contract_check_v1',
        chunk_number=args.chunk, checks=checks,
        checker_source_sha256=digest(ROOT / 'scripts/attention_relational_template_review_collection_v1.py'),
        dispatch_source_sha256=digest(Path(__file__)),
        full46context_collection_performed=False,
        semantic_endpoint_assessment_performed=False,
        feedback_sent_to_first_reviewers=False,
        no_replacement_or_gap_filling=True)
    output = Path(__file__).parent / f'first_pair_{args.chunk:02d}.json'
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(chunk=args.chunk,
        validity={c['reviewer_id']: c['structurally_valid'] for c in checks})))


if __name__ == '__main__':
    main()
