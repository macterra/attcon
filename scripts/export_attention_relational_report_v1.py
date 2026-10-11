"""Export the fixed retained-state reporting inventory once; no API or fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from attention_relational_report_v1 import build_reporting_packets
from attention_relational_reporting_pipeline_v1 import validate_export

OUTPUT_ROOT = Path('audits/attention_relational_report_v1/run/export')


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def exclusive(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))


def now():
    return datetime.now(timezone.utc).isoformat()


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
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt['completed_at_utc'] = now()
        exclusive(OUTPUT_ROOT / 'first_receipt.json', receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
