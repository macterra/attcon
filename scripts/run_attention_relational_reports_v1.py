"""Run a committed, reviewed request inventory once, retaining every first outcome.

No source fields, reference criteria or state histories are added to requests here.
The same runner can execute a separately committed, source-blind rating inventory.
Imports do not load credentials, contact an API, or read experiment data.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


SCHEMA = 'attention_relational_report_v1_runtime_manifest'


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def strict_json_loads(raw):
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('nonfinite JSON numeric exponent')
        return number
    def reject_constant(value):
        raise ValueError('nonfinite JSON constant')
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, parse_constant=reject_constant, parse_float=finite_float,
                      object_pairs_hook=unique_object)


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('expected a repository-relative path')
    return path


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'])


def validate_review_inventory(manifest, review):
    for key in ('settings', 'requests_path', 'request_count', 'request_ids', 'output_root',
                'lineage_path'):
        if review.get('reviewed_'+key) != manifest[key]:
            raise ValueError('runtime review does not bind exact '+key)


def validate(manifest_path, commit):
    """Validate committed exact sources and inventory before reserving attempts."""
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('full preservation commit required')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], check=True)
    manifest_path = relative_path(manifest_path)
    raw = manifest_path.read_bytes()
    if raw != git_bytes(commit, manifest_path):
        raise ValueError('manifest differs from preservation commit')
    manifest = json.loads(raw)
    if manifest.get('schema') != SCHEMA or manifest.get('stage') not in ('report', 'audit', 'rate'):
        raise ValueError('wrong manifest schema or stage')
    sources = manifest['source_sha256']
    own = str(Path(__file__).resolve().relative_to(Path.cwd().resolve()))
    for required in (own, manifest['requests_path'], manifest['review_path'], manifest['lineage_path']):
        if required not in sources:
            raise ValueError('missing committed required source: '+required)
    for name, sha in sources.items():
        path = relative_path(name)
        if digest(path.read_bytes()) != sha or digest(git_bytes(commit, path)) != sha:
            raise ValueError('changed source: '+name)
    review = json.loads(Path(manifest['review_path']).read_bytes())
    if review.get('runtime_approved') is not True or review.get('stage') != manifest['stage']:
        raise ValueError('exact stage runtime review required')
    validate_review_inventory(manifest, review)
    for name, sha in sources.items():
        if name != manifest['review_path'] and review.get('reviewed_sha256', {}).get(name) != sha:
            raise ValueError('review does not bind executable/request source: '+name)
    settings = manifest['settings']
    relative_path(manifest['output_root'])
    if set(settings) != {'model', 'reasoning_effort', 'max_output_tokens', 'timeout_seconds', 'concurrency', 'output_schema'}:
        raise ValueError('unexpected settings, refusing hidden API inputs')
    if not isinstance(settings['model'], str) or not settings['model']:
        raise ValueError('explicit model required')
    for key in ('max_output_tokens', 'timeout_seconds', 'concurrency'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError('positive bounded settings required')
    requests = json.loads(Path(manifest['requests_path']).read_bytes())
    ids = [r['id'] for r in requests]
    if ids != manifest['request_ids'] or len(ids) != manifest['request_count'] or len(set(ids)) != len(ids):
        raise ValueError('inventory differs from fixed schedule')
    for req in requests:
        if set(req) != {'id', 'input'} or not re.fullmatch(r'[A-Za-z0-9_-]+', req['id']):
            raise ValueError('requests must contain only safe ID and public input')
        if not isinstance(req['input'], str) or not req['input'].strip():
            raise ValueError('missing literal public request')
    return manifest, requests, digest(raw)


def exclusive_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


async def execute(manifest, requests, manifest_sha256, commit, output_root, client=None,
                  client_factory=None):
    """No retry/resume: an existing batch reservation always rejects execution."""
    root = Path(output_root)
    if relative_path(output_root) != relative_path(manifest['output_root']):
        raise ValueError('output root differs from fixed reviewed stage root')
    root.mkdir(parents=True, exist_ok=True)
    exclusive_json(root/'first_batch_claim.json', {
        'schema': 'attention_relational_report_v1_first_batch_claim',
        'stage': manifest['stage'], 'started_at_utc': now(),
        'manifest_sha256': manifest_sha256, 'preservation_commit': commit,
        'request_ids': [r['id'] for r in requests], 'max_attempts': len(requests),
        'retry_performed': False, 'conversation_or_previous_response': False})
    owned_client = client is None
    if owned_client:
        if client_factory is None:
            raise ValueError('client factory required')
        client = client_factory()
    semaphore = asyncio.Semaphore(manifest['settings']['concurrency'])
    settings = manifest['settings']

    async def one(req):
        async with semaphore:
            path = root/(req['id']+'.json')
            claimed = {'id': req['id'], 'status': 'attempt_reserved',
                       'started_at_utc': now(), 'request_sha256': digest(encoded(req))}
            exclusive_json(root/(req['id']+'.claim.json'), claimed)
            result = dict(claimed)
            text = {'verbosity': 'low'}
            if settings['output_schema'] is not None:
                text['format'] = {'type': 'json_schema', 'name': 'record_response',
                                  'strict': True, 'schema': settings['output_schema']}
            try:
                response = await client.responses.create(
                    model=settings['model'], input=req['input'],
                    reasoning={'effort': settings['reasoning_effort']},
                    max_output_tokens=settings['max_output_tokens'], text=text,
                    store=False)
                # Retain the full first response even if incomplete, refused or invalid.
                result.update(status='received', response=response.model_dump(mode='json'),
                              output_text=response.output_text,
                              api_response_status=response.status)
                if settings['output_schema'] is not None:
                    try:
                        result['parsed'] = strict_json_loads(response.output_text)
                    except (ValueError, TypeError):
                        result['parse_error'] = True
            except Exception as exc:
                # Exception text can contain credentials/request fragments; keep types/status.
                result.update(status='error', error_type=type(exc).__name__,
                              http_status=getattr(exc, 'status_code', None))
            result['completed_at_utc'] = now()
            # Keep the reservation separate, and never overwrite a first outcome.
            storage_error = None
            try:
                exclusive_json(path, result)
            except Exception as exc:
                envelope = {'id': req['id'], 'first_outcome': result,
                            'primary_write_error': type(exc).__name__}
                path = root/(req['id']+'.secondary.json')
                try:
                    exclusive_json(path, envelope)
                except Exception as second_exc:
                    envelope['secondary_write_error'] = type(second_exc).__name__
                    storage_error = envelope
                    # The external first CLI log and batch completion retain this
                    # envelope if neither per-attempt outcome file can be written.
                    print(json.dumps(envelope, allow_nan=False), file=sys.stderr, flush=True)
            print(req['id'], result['status'], result.get('api_response_status', ''), flush=True)
            if storage_error is not None:
                return {'id': req['id'], 'status': 'storage_failure',
                        'retained_first_outcome_envelope': storage_error}
            return {'id': req['id'], 'status': result['status'], 'response_path': str(path),
                    'response_sha256': digest(path.read_bytes())}

    try:
        outcomes = await asyncio.gather(*(one(req) for req in requests))
    finally:
        if owned_client:
            await client.close()
    exclusive_json(root/'first_completion.json', {
        'schema': 'attention_relational_report_v1_first_batch_completion',
        'completed_at_utc': now(), 'manifest_sha256': manifest_sha256,
        'preservation_commit': commit, 'request_count': len(requests),
        'outcomes': outcomes, 'retry_performed': False})
    return outcomes


async def main_async(args):
    manifest, requests, sha = validate(args.manifest, args.commit)
    if relative_path(args.output_root) != relative_path(manifest['output_root']):
        raise ValueError('output root differs from fixed reviewed stage root')
    # Reject an existing claim before credential loading or API client construction.
    if (Path(args.output_root)/'first_batch_claim.json').exists():
        raise ValueError('first batch already reserved; refusing retry or resume')
    def make_client():
        from openai import AsyncOpenAI
        from attcon.nl_report import load_dotenv
        load_dotenv()
        return AsyncOpenAI(max_retries=0, timeout=manifest['settings']['timeout_seconds'])
    await execute(manifest, requests, sha, args.commit, args.output_root,
                  client_factory=make_client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--output-root', required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
