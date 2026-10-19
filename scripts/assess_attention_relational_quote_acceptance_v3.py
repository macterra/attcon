"""Assess one fictional, unscored V3 transport/grounding acceptance outcome."""
import argparse
from pathlib import Path

import attention_relational_quote_reference_v3 as quote
import attention_relational_reporting_pipeline_v1 as pipeline
import run_attention_relational_reports_v1 as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first acceptance')
    manifest, requests, manifest_sha = runner.validate(args.manifest, args.commit)
    pipeline.require(len(requests) == 1 and manifest['stage'] == 'rate'
        and manifest['settings']['output_schema'] == quote.response_schema(),
        'acceptance requires its one actual V3 request')
    lineage = pipeline.bound_json(manifest['lineage_path'],
        manifest['source_sha256'][manifest['lineage_path']])
    pipeline.require(lineage['schema'] == 'attention_relational_quote_v3_fictional_acceptance_lineage'
        and lineage['scientific_report'] is False
        and lineage['unit_id'] == requests[0]['id'], 'not a fictional acceptance lineage')
    instruction_path = lineage['instruction_path']
    instruction = Path(instruction_path).read_bytes()
    pipeline.require(pipeline.digest(instruction) == manifest['source_sha256'][instruction_path]
        and requests[0]['input'] == instruction.decode() + '\n' + pipeline.encoded({
            'reference': pipeline.frozen_reference(), 'unit_id': lineage['unit_id'],
            'report_text': lineage['report_text']}).decode('ascii'),
        'fictional text/reference/instruction differ from actual request')
    root = Path(manifest['output_root'])
    completion_path = root / 'first_completion.json'
    completion = pipeline.decoded(completion_path.read_bytes())
    pipeline.require(completion['manifest_sha256'] == manifest_sha
        and completion['preservation_commit'] == args.commit
        and completion['request_count'] == 1 and completion['retry_performed'] is False
        and len(completion['outcomes']) == 1, 'acceptance completion binding differs')
    row = completion['outcomes'][0]
    outcome_path = root / (requests[0]['id'] + '.json')
    pipeline.require(row['id'] == requests[0]['id']
        and Path(row['response_path']).resolve() == outcome_path.resolve(),
        'acceptance response path differs')
    outcome = pipeline.bound_json(outcome_path, row['response_sha256'])
    pipeline.require(outcome['request_sha256'] == pipeline.runner_request_sha256(requests[0]),
                     'acceptance first request binding differs')
    record = pipeline._response_record(requests[0]['id'], outcome, {})
    derived = quote.derive_annotation(lineage['report_text'], lineage['unit_id'],
        record['parsed'], expected_text_sha256=lineage['text_sha256'])
    passed = bool(record['transport_complete'] and derived['valid']
                  and derived['quote_diagnostics'])
    pipeline.exclusive_json(args.output, {
        'schema': 'attention_relational_quote_v3_fictional_acceptance_assessment',
        'fictional_unscored': True, 'scientific_labels_scored': False,
        'manifest_sha256': manifest_sha, 'raw_response_sha256': row['response_sha256'],
        'completion_sha256': pipeline.digest(completion_path.read_bytes()),
        'transport_complete': record['transport_complete'], 'derived': derived,
        'transport_and_nonempty_quote_grounding_pass': passed,
        'semantic_rating_accuracy_or_reliability_established': False,
        'full_target_release_approved': False, 'API_calls': 0})


if __name__ == '__main__':
    main()
