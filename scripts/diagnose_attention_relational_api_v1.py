"""One reviewed handcrafted transport diagnostic; no scientific states or reports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

OUTPUT_ROOT = Path('audits/attention_relational_report_v1/api_transport_diagnostic/run')


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def exclusive(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))


def scrub(value):
    if isinstance(value, str):
        key = os.environ.get('OPENAI_API_KEY')
        if key:
            value = value.replace(key, '[REDACTED]')
        return re.sub(r'sk-[A-Za-z0-9_-]+', '[REDACTED]', value)
    if isinstance(value, dict):
        return {str(k): scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    raw = Path(args.manifest).read_bytes()
    if raw != subprocess.check_output(['git', 'show', args.commit + ':' + args.manifest]):
        raise ValueError('diagnostic manifest differs from preservation commit')
    subprocess.run(['git', 'merge-base', '--is-ancestor', args.commit, 'HEAD'], check=True)
    manifest = json.loads(raw)
    for path, sha in manifest['source_sha256'].items():
        if digest(Path(path).read_bytes()) != sha or digest(subprocess.check_output(
                ['git', 'show', args.commit + ':' + path])) != sha:
            raise ValueError('diagnostic source changed: ' + path)
    review = json.loads(Path(manifest['review_path']).read_bytes())
    expected = {p: h for p, h in manifest['source_sha256'].items() if p != manifest['review_path']}
    if (review.get('runtime_approved') is not True or review.get('max_attempts') != 1
            or review.get('reviewed_sha256') != expected
            or review.get('reviewed_input') != manifest['input']
            or review.get('reviewed_settings') != manifest['settings']
            or review.get('reviewed_output_root') != str(OUTPUT_ROOT)):
        raise ValueError('exact single diagnostic review required')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exclusive(OUTPUT_ROOT / 'first_claim.json', {
        'schema': 'attention_relational_api_transport_diagnostic_claim',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': digest(raw), 'preservation_commit': args.commit,
        'max_attempts': 1, 'retry': False, 'scientific_report': False})
    result = {'schema': 'attention_relational_api_transport_diagnostic_first_outcome',
              'scientific_report': False, 'retry': False, 'max_attempts': 1}
    try:
        from openai import OpenAI
        from attcon.nl_report import load_dotenv
        load_dotenv()
        settings = manifest['settings']
        client = OpenAI(max_retries=0, timeout=settings['timeout_seconds'])
        try:
            response = client.responses.create(
                model=settings['model'], input=manifest['input'],
                reasoning={'effort': settings['reasoning_effort']},
                max_output_tokens=settings['max_output_tokens'], store=False,
                text={'verbosity': 'low', 'format': {'type': 'json_schema',
                    'name': 'record_response', 'strict': True, 'schema': settings['output_schema']}})
            result.update(status='received', response=scrub(response.model_dump(mode='json')))
        finally:
            client.close()
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__,
                      http_status=getattr(exc, 'status_code', None),
                      provider_body=scrub(getattr(exc, 'body', None)))
    result['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    exclusive(OUTPUT_ROOT / 'first_outcome.json', result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
