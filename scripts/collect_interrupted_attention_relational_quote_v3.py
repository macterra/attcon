"""Preserve an interrupted V3 rater with actual V3 lineage and all positions.

No completion is fabricated. Root must first observe the original tool handle
terminal and attest it in rate/first_terminal_receipt.json. This adapter cannot
observe tool handles, invoke an API, resume a run, or repair annotations.
"""
import argparse
from pathlib import Path

import attention_relational_quote_measurement_v3 as measurement
import attention_relational_reporting_pipeline_v1 as pipeline
import run_attention_relational_reports_v1 as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--preparation-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of interrupted collection')
    prepared = measurement.load_preparation(args.preparation, args.preparation_sha256)
    manifest, requests, manifest_sha = runner.validate(args.manifest, args.commit)
    pipeline.require(manifest['source_sha256'].get(args.preparation) == args.preparation_sha256,
                     'reviewed runtime did not bind actual V3 preparation')
    root = Path(manifest['output_root'])
    pipeline.require(str(root) == measurement.RATE_ROOT, 'not the actual V3 rate root')
    execution_path, terminal_path = root / 'first_execution.json', root / 'first_terminal_receipt.json'
    raw_execution, raw_terminal = execution_path.read_bytes(), terminal_path.read_bytes()
    execution, terminal = pipeline.decoded(raw_execution), pipeline.decoded(raw_terminal)
    pipeline.require(terminal.get('schema') == 'attention_relational_quote_v3_terminal_receipt'
        and terminal.get('stage') == 'rate' and terminal.get('terminal_verified') is True
        and type(terminal.get('tool_session_id')) is int
        and terminal.get('manifest_sha256') == manifest_sha
        and terminal.get('execution_receipt_sha256') == pipeline.digest(raw_execution)
        and terminal.get('output_root') == str(root), 'root V3 terminal attestation differs')
    expected = ['.venv/bin/python', 'scripts/run_attention_relational_reports_v1.py',
        '--manifest', args.manifest, '--commit', args.commit, '--output-root', str(root)]
    pipeline.require(execution['command'] == expected and execution['preservation_commit'] == args.commit
        and execution.get('stage') == 'rate' and type(execution.get('exit_code')) is int
        and execution['exit_code'] != 0 and terminal.get('child_exit_code') == execution['exit_code']
        and execution.get('retry_performed') is False, 'not a terminal first interrupted V3 execution')
    log = root / 'first_execution.log'
    pipeline.require(pipeline.digest(log.read_bytes()) == execution['log_sha256'], 'first V3 log differs')
    pipeline.require(not (root / 'first_completion.json').exists(), 'use normal V3 collector for completion')
    sources = manifest['source_sha256']
    lineage = pipeline.bound_json(manifest['lineage_path'], sources[manifest['lineage_path']])
    binding = measurement.build_binding(manifest, manifest_sha, requests, sources[manifest['requests_path']],
        lineage, sources[manifest['lineage_path']], prepared)
    result = pipeline.collect_interrupted_responses(binding['request_ids'], root, stage_binding=binding)
    pipeline.require(result['request_count'] == len(result['records']) == 1104,
                     'all V3 positions must be retained')
    result['input_dependencies'] = {
        'preparation': {'path': args.preparation, 'sha256': args.preparation_sha256},
        'manifest': {'path': args.manifest, 'sha256': manifest_sha, 'commit': args.commit},
        'execution_receipt': {'path': str(execution_path), 'sha256': pipeline.digest(raw_execution)},
        'terminal_receipt': {'path': str(terminal_path), 'sha256': pipeline.digest(raw_terminal)}}
    result['source_sha256'] = measurement.source_hashes()
    result['source_sha256'][str(Path(__file__))] = pipeline.digest(Path(__file__).read_bytes())
    result['no_api_or_model_calls'] = True
    result['normal_completion_fabricated'] = False
    result['terminal_evidence_basis'] = 'Root attests observed original tool handle; adapter does not observe it'
    pipeline.require(not (root / 'first_completion.json').exists(), 'completion appeared during collection')
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
