"""Retain the whole stopped resource-v2 audit using the frozen raw collector.

Root's terminal receipt attests the observed tool handle; this read-only adapter
does not itself observe that handle. No completion, missing response or repaired
annotation is invented. Imports do not load credentials or invoke an API.
"""
import argparse
from pathlib import Path

import attention_relational_reporting_pipeline_v1 as pipeline
import run_attention_relational_reports_v1 as runner


BASE = Path('audits/attention_relational_report_v1/resource_measurement_v2')
MANIFEST = BASE / 'audit_preparation/runtime_manifest.json'
RELEASE = '88be2da1d8ace605e29890706d8a11891b591feb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first collection')
    manifest, requests, manifest_sha = runner.validate(str(MANIFEST), RELEASE)
    pipeline.require(manifest_sha == '973b6d64c76b6dc46468f0185a182a210176c6e756caa43a5b7800ea89ddb44a'
        and manifest['stage'] == 'audit' and len(requests) == manifest['request_count'] == 537
        and manifest['output_root'] == str(BASE / 'audit'), 'wrong supplementary audit version')
    root = Path(manifest['output_root'])
    execution_path = root / 'first_execution.json'
    stop_path = BASE / 'audit_operational_stop/terminal_receipt.json'
    execution_raw, stop_raw = execution_path.read_bytes(), stop_path.read_bytes()
    execution, stop = pipeline.decoded(execution_raw), pipeline.decoded(stop_raw)
    pipeline.require(stop.get('schema') == 'attention_relational_resource_v2_auditor_terminal_stop_receipt'
        and stop.get('terminal_verified') is True and stop.get('stage') == 'audit'
        and stop.get('tool_session_id') == 75382 and stop.get('child_exit_code') == -2
        and stop.get('child_process_absent_after_terminal') is True
        and stop.get('execution_receipt_sha256') == pipeline.digest(execution_raw)
        and stop.get('first_completion_present') is False and stop.get('signal_count') == 1,
        'root terminal attestation differs')
    for key, path in [('manifest_path', MANIFEST), ('output_root', root),
                      ('execution_receipt_path', execution_path)]:
        pipeline.require(Path(stop[key]).resolve() == path.resolve(), 'terminal path differs: ' + key)
    expected_command = ['.venv/bin/python', 'scripts/run_attention_relational_reports_v1.py',
        '--manifest', str(MANIFEST), '--commit', RELEASE, '--output-root', str(root)]
    pipeline.require(execution['command'] == expected_command and execution['preservation_commit'] == RELEASE
        and execution['stage'] == 'audit' and execution['exit_code'] == stop['child_exit_code']
        and execution['retry_performed'] is False, 'first execution differs from stopped reviewed run')
    log = root / 'first_execution.log'
    pipeline.require(pipeline.digest(log.read_bytes()) == execution['log_sha256'], 'execution log differs')
    pipeline.require(not (root / 'first_completion.json').exists(), 'use normal collector for a completed run')
    sources = manifest['source_sha256']
    lineage_path = manifest['lineage_path']
    lineage = pipeline.bound_json(lineage_path, sources[lineage_path])
    binding = pipeline.build_collection_binding(manifest, manifest_sha, requests,
        sources[manifest['requests_path']], lineage, sources[lineage_path])
    result = pipeline.collect_interrupted_responses(manifest['request_ids'], root, stage_binding=binding)
    pipeline.require(len(result['records']) == result['request_count'] == 537
        and [r['id'] for r in result['records']] == manifest['request_ids'], 'whole inventory differs')
    result['preparation_dependencies'] = {
        'manifest': {'path': str(MANIFEST), 'sha256': manifest_sha, 'commit': RELEASE},
        'execution_receipt': {'path': str(execution_path), 'sha256': pipeline.digest(execution_raw)},
        'stop_receipt': {'path': str(stop_path), 'sha256': pipeline.digest(stop_raw)},
        'execution_log': {'path': str(log), 'sha256': execution['log_sha256']}}
    result['source_sha256'] = {str(Path(m.__file__)): pipeline.digest(Path(m.__file__).read_bytes())
                               for m in (pipeline, runner)}
    result['source_sha256'][str(Path(__file__))] = pipeline.digest(Path(__file__).read_bytes())
    result['no_api_or_model_calls'] = True
    result['normal_completion_fabricated'] = False
    result['terminal_evidence_basis'] = 'Root terminal receipt; adapter does not observe the live tool handle'
    pipeline.require(not (root / 'first_completion.json').exists(), 'completion appeared during collection')
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
