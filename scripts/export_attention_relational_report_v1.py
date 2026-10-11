"""Export the fixed retained-state reporting inventory once; no API or fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from attention_relational_report_v1 import build_reporting_packets
from attention_relational_reporting_pipeline_v1 import validate_export, build_pair_index

OUTPUT_ROOT = Path('audits/attention_relational_report_v1/run/export')


def encoded(value):
    return (json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False) + '\n').encode()


def exclusive(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))


def now():
    return datetime.now(timezone.utc).isoformat()


def partial_export(exc):
    """Retain available pure JSON assembly state; never serialize live tensors."""
    frames = []
    trace = exc.__traceback__
    names = ('result', 'context', 'case', 'packet', 'request_id', 'seed', 'world',
             'source', 'view', 'variant', 'previous', 'candidate', 'queried_role')
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code.co_filename.endswith('attention_relational_report_v1.py'):
            values, unavailable = {}, []
            for name in names:
                if name not in frame.f_locals:
                    continue
                value = frame.f_locals[name]
                try:
                    encoded(value)
                except (TypeError, ValueError, OverflowError):
                    unavailable.append({'name': name, 'type': type(value).__name__})
                else:
                    values[name] = value
            frames.append({'function': frame.f_code.co_name, 'line': trace.tb_lineno,
                           'available_json_state': values, 'unserialized_values': unavailable})
        trace = trace.tb_next
    return {'schema': 'attention_relational_report_v1_first_partial_export',
            'error_type': type(exc).__name__, 'error': str(exc), 'frames': frames,
            'missing_packet_means_assembly_not_completed': True, 'retry_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    inputs_path = Path(args.inputs)
    if inputs_path.is_absolute() or '..' in inputs_path.parts:
        raise ValueError('repository-relative input path required')
    raw = inputs_path.read_bytes()
    if raw != subprocess.check_output(['git', 'show', args.commit + ':' + str(inputs_path)]):
        raise ValueError('inputs differ from preserved commit')
    subprocess.run(['git', 'merge-base', '--is-ancestor', args.commit, 'HEAD'], check=True)
    bindings = json.loads(raw)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exclusive(OUTPUT_ROOT / 'first_claim.json', {
        'schema': 'attention_relational_report_v1_first_export_claim',
        'started_at_utc': now(), 'preservation_commit': args.commit,
        'inputs_path': str(inputs_path), 'inputs_sha256': hashlib.sha256(raw).hexdigest(),
        'retry_performed': False, 'no_api_or_fitting': True})
    receipt = {'schema': 'attention_relational_report_v1_first_export_receipt',
               'started_at_utc': now(), 'preservation_commit': args.commit,
               'retry_performed': False, 'no_api_or_fitting': True}
    try:
        paths = {seed: f'audits/attention_content_model_v1/run/heldout/states-{seed}.pt.gz'
                 for seed in (1301, 1311, 1321)}
        result = build_reporting_packets(paths, bindings)
        # Retain the first completed export before downstream inventory validation.
        export_path = OUTPUT_ROOT / 'first_exports.json'
        exclusive(export_path, result)
        receipt['export_path'] = str(export_path)
        receipt['export_sha256'] = hashlib.sha256(export_path.read_bytes()).hexdigest()
        inventory = validate_export(result)
        receipt.update(status='completed', request_count=len(inventory['request_ids']),
                       required_peak_count=inventory['required_peak_count'],
                       paired_check_count=len(result['paired_checks']))
        artifacts = {
            'requests.json': result['requests'],
            'neutral_packets.json': result['packets'],
            'audit_packets.json': {'packets': result['packets'], 'provenance': result['provenance']},
            'pair_index.json': build_pair_index(result['provenance']),
            'template_reports.json': result['templates'],
            'response_schema.json': result['response_schema'],
        }
        hashes = {}
        for filename, value in artifacts.items():
            path = OUTPUT_ROOT / filename
            exclusive(path, value)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        exclusive(OUTPUT_ROOT / 'export_manifest.json', {
            'schema': 'attention_relational_report_v1_export_manifest',
            'first_export_sha256': receipt['export_sha256'], 'artifact_sha256': hashes,
            'request_count': 552, 'required_peak_count': 25440,
            'causal_pair_count': len(artifacts['pair_index.json']),
            'no_api_or_fitting': True, 'retry_performed': False})
        receipt['artifact_sha256'] = hashes
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        partial_path = OUTPUT_ROOT / 'first_partial_export.json'
        partial = partial_export(exc)
        try:
            exclusive(partial_path, partial)
            receipt['partial_path'] = str(partial_path)
            receipt['partial_sha256'] = hashlib.sha256(partial_path.read_bytes()).hexdigest()
        except Exception as storage_exc:
            receipt['partial_write_error_type'] = type(storage_exc).__name__
            receipt['retained_partial_export'] = partial
            # The exclusive external first log preserves this even if receipt writing fails.
            print(encoded(partial).decode(), file=sys.stderr, flush=True)
        raise
    finally:
        receipt['completed_at_utc'] = now()
        exclusive(OUTPUT_ROOT / 'first_receipt.json', receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
