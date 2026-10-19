"""One bounded contract-only operational check; no scientific endpoint scoring."""
import argparse
from collections import Counter
from pathlib import Path

import attention_relational_reporting_pipeline_v1 as pipeline
import run_attention_relational_reports_v1 as runner


BASE = Path('audits/attention_relational_report_v1')
MANIFEST = BASE / 'resource_measurement_v2/audit_preparation/runtime_manifest.json'
RELEASE = '88be2da1d8ace605e29890706d8a11891b591feb'
EXPORT = BASE / 'run/export/first_exports.json'
REPORTS = BASE / 'run_v2/report_collection/first_collection.json'
PREFIX_SIZE = 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing a second operational assessment')
    plan = pipeline.bound_json(args.plan, args.plan_sha256)
    manifest, requests, manifest_sha = runner.validate(str(MANIFEST), RELEASE)
    prefix = requests[:PREFIX_SIZE]
    pipeline.require(plan['schema'] == 'attention_relational_resource_v2_contract_prefix_plan'
        and plan['prefix_ids'] == [r['id'] for r in prefix]
        and plan['prefix_size'] == PREFIX_SIZE
        and plan['maximum_valid_annotations_for_stop'] == 3
        and plan['manifest_sha256'] == manifest_sha
        and str(args.output) == plan['result_path']
        and plan['source_sha256'] == pipeline.digest(Path(__file__).read_bytes()),
        'operational plan differs from frozen check')
    exports = pipeline.bound_json(EXPORT, plan['exports_sha256'])
    reports = pipeline.bound_json(REPORTS, plan['reports_sha256'])
    pipeline._validate_collection_lineage(reports, 'report', exports['requests'], exports)
    pipeline.require(requests == pipeline.build_prose_audit_requests(reports, exports['packets']),
                     'actual audit requests differ from original exact full inventory')
    packet_by_id = pipeline._index(exports['packets'])
    report_by_id = pipeline._records(reports)
    root = Path(manifest['output_root'])
    batch_path = root / 'first_batch_claim.json'
    batch_raw = batch_path.read_bytes()
    batch = runner.strict_json_loads(batch_raw)
    pipeline.require(batch.get('schema') == 'attention_relational_report_v1_first_batch_claim'
        and batch.get('stage') == 'audit' and batch.get('manifest_sha256') == manifest_sha
        and batch.get('preservation_commit') == RELEASE
        and batch.get('request_ids') == manifest['request_ids']
        and batch.get('max_attempts') == len(requests)
        and batch.get('retry_performed') is False,
        'live batch reservation differs from the exact released full inventory')
    for req in prefix:
        pipeline.require((root / (req['id'] + '.json')).is_file(),
                         'fixed prefix not fully returned; do not run check yet')
    pipeline.exclusive_json(plan['claim_path'], {
        'schema': 'attention_relational_resource_v2_first_contract_prefix_claim',
        'plan_sha256': args.plan_sha256, 'claimed_at_utc': runner.now(),
        'manifest_sha256': manifest_sha, 'result_path': str(args.output),
        'batch_claim_sha256': pipeline.digest(batch_raw), 'retry_performed': False})
    statuses, errors = Counter(), Counter()
    valid = 0
    hashes = {}
    for req in prefix:
        path = root / (req['id'] + '.json')
        raw = path.read_bytes()
        outcome = runner.strict_json_loads(raw)
        pipeline.require(outcome['id'] == req['id']
            and outcome['request_sha256'] == pipeline.runner_request_sha256(req),
            'first raw outcome request binding differs')
        record = pipeline._response_record(req['id'], outcome, {})
        statuses[str(record.get('api_status'))] += 1
        result = pipeline.validate_prose_audit(packet_by_id[req['id']]['content'],
            pipeline._report_text(report_by_id[req['id']]), record.get('parsed'))
        valid += int(record['transport_complete'] and result['valid'])
        errors.update(result['errors'])
        hashes[str(path)] = pipeline.digest(raw)
    pipeline.exclusive_json(args.output, {
        'schema': 'attention_relational_resource_v2_contract_prefix_result',
        'plan_sha256': args.plan_sha256, 'manifest_sha256': manifest_sha,
        'batch_claim_sha256': pipeline.digest(batch_raw),
        'prefix_size': PREFIX_SIZE, 'transport_status_counts': dict(statuses),
        'contract_valid_count': valid, 'contract_unestablished_count': PREFIX_SIZE - valid,
        'error_occurrence_counts': dict(errors), 'raw_outcome_sha256': hashes,
        'operational_stop_condition_met': valid <= 3,
        'not_a_scientific_pass_rule': True, 'no_endpoint_or_fidelity_scores_emitted': True,
        'no_quotes_verdicts_or_feature_labels_emitted': True, 'API_calls': 0})


if __name__ == '__main__':
    main()
