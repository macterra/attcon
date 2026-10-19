"""Collect an operationally stopped original rater once, without API/client use.

The root stop receipt attests verification of the original live terminal handle;
this adapter cannot verify that handle. Frozen collection alone parses response
files; no annotation labels are analyzed or repaired and no completion is invented.
"""
import argparse
from pathlib import Path

import attention_relational_reporting_pipeline_v1 as pipeline
import run_attention_relational_reports_v1 as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--execution-receipt', type=Path, required=True)
    parser.add_argument('--stop-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first collection')
    manifest, requests, manifest_sha = runner.validate(str(args.manifest), args.commit)
    pipeline.require(manifest['stage'] == 'rate' and len(requests) == manifest['request_count'] == 1104,
                     'only the complete original1104 rate inventory is allowed')
    root = Path(manifest['output_root'])
    pipeline.require(args.execution_receipt.resolve() == (root / 'first_execution.json').resolve(),
                     'execution receipt is not at the original stage root')
    pipeline.require(not (root / 'first_completion.json').exists(), 'normal completion exists; use frozen completed collector')
    raw_execution = args.execution_receipt.read_bytes()
    execution_sha = pipeline.digest(raw_execution)
    execution = pipeline.decoded(raw_execution)
    raw_stop = args.stop_receipt.read_bytes()
    stop = pipeline.decoded(raw_stop)
    expected_stop = {'stage': 'rate', 'terminal_verified': True, 'original_session_id': 62873,
                     'execution_receipt_sha256': execution_sha}
    pipeline.require(all(stop.get(k) == v for k, v in expected_stop.items()),
                     'root stop receipt lacks exact original terminal/receipt attestation')
    pipeline.require(type(stop['terminal_verified']) is bool and type(stop['original_session_id']) is int,
                     'invalid terminal attestation types')
    for key, expected in (('execution_receipt_path', args.execution_receipt),
                          ('original_manifest_path', args.manifest), ('original_output_root', root)):
        pipeline.require(Path(stop[key]).resolve() == expected.resolve(), 'stop receipt path differs: ' + key)
    command = execution['command']
    pipeline.require(type(command) is list and len(command) == 8
                     and type(command[0]) is str and bool(command[0])
                     and command[1:] == ['scripts/run_attention_relational_reports_v1.py',
                         '--manifest', str(args.manifest), '--commit', args.commit,
                         '--output-root', manifest['output_root']], 'original execution command differs')
    pipeline.require(execution['preservation_commit'] == args.commit
                     and execution.get('stage') == 'rate'
                     and type(execution['exit_code']) is int and execution['exit_code'] != 0
                     and type(stop['child_exit_code']) is int
                     and stop['child_exit_code'] == execution['exit_code'], 'original child exit/commit/stage differs')
    pipeline.require(execution.get('retry_performed') is False, 'retried execution is not firstpass')
    log = root / 'first_execution.log'
    pipeline.require(pipeline.digest(log.read_bytes()) == execution['log_sha256'], 'first execution log SHA differs')
    sources = manifest['source_sha256']
    lineage_path, requests_path = manifest['lineage_path'], manifest['requests_path']
    lineage = pipeline.bound_json(lineage_path, sources[lineage_path])
    binding = pipeline.build_collection_binding(manifest, manifest_sha, requests, sources[requests_path],
                                                 lineage, sources[lineage_path])
    result = pipeline.collect_interrupted_responses(manifest['request_ids'], root, stage_binding=binding)
    pipeline.require(len(result['records']) == result['request_count'] == 1104
                     and [r['id'] for r in result['records']] == manifest['request_ids'], 'interrupted inventory differs')
    result['preparation_dependencies'] = {
        'manifest': {'path': str(args.manifest), 'sha256': manifest_sha, 'commit': args.commit},
        'requests': {'path': requests_path, 'sha256': sources[requests_path]},
        'lineage': {'path': lineage_path, 'sha256': sources[lineage_path]},
        'execution_receipt': {'path': str(args.execution_receipt), 'sha256': execution_sha},
        'execution_log': {'path': str(log), 'sha256': execution['log_sha256']},
        'stop_receipt': {'path': str(args.stop_receipt), 'sha256': pipeline.digest(raw_stop)}}
    result['pipeline_source_sha256'] = pipeline.digest(Path(pipeline.__file__).read_bytes())
    result['source_sha256'] = {str(Path(module.__file__)): pipeline.digest(Path(module.__file__).read_bytes())
                               for module in (pipeline, runner)}
    result['source_sha256'][str(Path(__file__))] = pipeline.digest(Path(__file__).read_bytes())
    result['no_api_or_model_calls'] = True
    result['terminal_evidence_basis'] = 'Root stop-receipt attestation; adapter did not inspect the live tool handle'
    result['normal_completion_fabricated'] = False
    result['annotation_labels_analyzed_or_repaired'] = False
    pipeline.require(not (root / 'first_completion.json').exists(), 'completion appeared during interrupted collection')
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
