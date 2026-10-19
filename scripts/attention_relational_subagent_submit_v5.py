"""Standalone stdlib-only constrained form for exclusive V5 first submissions.

No repository imports, credentials, semantic/source validator or gold feedback.
Workers supply every assertion, coverage row and both inventory flags manually.
Interface failures are appended to a separate event log and never repair a file.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


BATCH_SCHEMA = 'attention_relational_subagent_auditor_v5_public_batch'
RAW_SCHEMA = 'attention_relational_grounded_auditor_v4_annotation'
FIRST_SCHEMA = 'attention_relational_subagent_auditor_v5_first_batch'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def decoded(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exclusive(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)


def request_sha(request):
    return sha((json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def load_batch(path, output_root):
    raw = Path(path).read_bytes()
    batch = decoded(raw)
    require(type(batch) is dict and set(batch) == {'schema', 'batch_id', 'output_root', 'worker_instructions',
        'bindings', 'annotation_schema', 'requests'}, 'public batch required fields differ')
    require(batch['schema'] == BATCH_SCHEMA and batch['bindings']['submission_helper_sha256'] == sha(Path(__file__).read_bytes()),
            'batch schema/helper SHA differs')
    require(type(batch['worker_instructions']) is str and sha(batch['worker_instructions'].encode('utf-8')) == batch['bindings']['worker_instructions_sha256']
        and sha(encoded(batch['annotation_schema'])) == batch['bindings']['response_schema_content_sha256'],
        'public instruction/schema content SHA differs')
    require(Path(output_root).resolve() == Path(batch['output_root']).resolve(), 'assigned output root differs')
    requests = {}
    for request in batch['requests']:
        require(set(request) == {'id', 'input'} and type(request['id']) is str
                and re.fullmatch(r'[A-Za-z0-9_-]+', request['id']) and type(request['input']) is str,
                'invalid public request shape/ID')
        require(request['id'] not in requests, 'duplicate public request ID')
        requests[request['id']] = request
    return batch, requests, sha(raw)


def make_annotation(unit_id, assertions, coverage, *, complete_assertion_inventory, unresolved_ambiguity):
    """No automatic claim extraction, coverage filling or positive flag defaults."""
    return dict(schema=RAW_SCHEMA, unit_id=unit_id, assertions=assertions, coverage=coverage,
                complete_assertion_inventory=complete_assertion_inventory, unresolved_ambiguity=unresolved_ambiguity)


def validate_interface(request, annotation):
    """Closed fields/types/enums and legal IDs only; no semantic correctness."""
    payload = decoded(request['input'].rsplit('\n', 1)[1])
    require(type(annotation) is dict and set(annotation) == {'schema', 'unit_id', 'assertions', 'coverage',
            'complete_assertion_inventory', 'unresolved_ambiguity'}, 'annotation required fields differ')
    require(annotation['schema'] == RAW_SCHEMA and annotation['unit_id'] == payload['unit_id'], 'annotation schema/unit ID differs')
    require(all(type(annotation[k]) is bool for k in ('complete_assertion_inventory', 'unresolved_ambiguity')),
            'inventory/ambiguity flags must be explicit booleans')
    require(type(annotation['assertions']) is list and type(annotation['coverage']) is list, 'assertions/coverage must be lists')
    span_ids = {s['span_id'] for s in payload['spans']}
    context_ids = {c['context_id'] for c in payload['source_contexts']}
    assertion_ids = set()

    def refs(values, legal, name):
        require(type(values) is list and all(type(v) is str and v in legal for v in values), 'illegal ' + name + ' IDs')

    for assertion in annotation['assertions']:
        require(type(assertion) is dict and set(assertion) == {'assertion_id', 'paraphrase', 'span_ids',
            'source_context_ids', 'verdict', 'reason'}, 'assertion required fields differ')
        require(type(assertion['assertion_id']) is str and bool(assertion['assertion_id'])
            and assertion['assertion_id'] not in assertion_ids, 'invalid/duplicate assertion ID')
        assertion_ids.add(assertion['assertion_id'])
        require(all(type(assertion[k]) is str for k in ('paraphrase', 'reason'))
            and assertion['verdict'] in ('supported', 'contradicted', 'not_auditable'), 'assertion field types/enum differ')
        refs(assertion['span_ids'], span_ids, 'span')
        refs(assertion['source_context_ids'], context_ids, 'source-context')
    for entry in annotation['coverage']:
        require(type(entry) is dict and set(entry) == {'span_id', 'assertion_ids', 'classification', 'reason'}, 'coverage required fields differ')
        require(type(entry['span_id']) is str and entry['span_id'] in span_ids
            and entry['classification'] in ('assertive', 'nonassertive') and type(entry['reason']) is str, 'coverage field types/IDs differ')
        refs(entry['assertion_ids'], assertion_ids, 'assertion')
    encoded(annotation)  # Finite JSON, with no generated locations or judgments.


def log_event(output_root, request_id, error=None, annotation_sha256=None):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    event = dict(schema='attention_relational_subagent_auditor_v5_submission_event',
        at_utc=datetime.now(timezone.utc).isoformat(), request_id=request_id,
        event='interface_failure' if error else 'first_submission', interface_error=error,
        annotation_sha256=annotation_sha256, semantic_feedback_provided=False)
    # A forbidden attempt after finalization cannot mutate the already bound log.
    name = 'after_finalization_events.jsonl' if (root / 'first_batch.json').exists() else 'submission_events.jsonl'
    fd = os.open(root / name, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded(event) + b'\n')


def submit(batch_path, output_root, request_id, annotation, *, raw_bytes=None):
    try:
        batch, requests, batch_sha = load_batch(batch_path, output_root)
        root = Path(output_root)
        require(not (root / 'first_batch.json').exists(), 'first batch is finalized; further submissions prohibited')
        require(request_id in requests, 'request ID is not in the assigned batch')
        destination = root / (request_id + '.first.json')
        require(not destination.exists(), 'exclusive first annotation already exists; no resubmission')
        validate_interface(requests[request_id], annotation)
        raw = encoded(annotation) + b'\n' if raw_bytes is None else raw_bytes
        require(encoded(decoded(raw)) == encoded(annotation), 'manual JSON bytes differ from supplied annotation')
        exclusive(destination, raw)
        receipt = dict(schema='attention_relational_subagent_auditor_v5_submission_receipt', request_id=request_id,
            batch_sha256=batch_sha, request_sha256=request_sha(requests[request_id]),
            annotation_sha256=sha(raw), submission_helper_sha256=batch['bindings']['submission_helper_sha256'])
        exclusive(root / (request_id + '.receipt.json'), encoded(receipt) + b'\n')
        log_event(root, request_id, annotation_sha256=sha(raw))
        return receipt
    except (ValueError, KeyError, TypeError, OSError) as exc:
        log_event(output_root, request_id, error=type(exc).__name__ + ': ' + str(exc))
        raise


def finalize(batch_path, output_root, *, calculator_artifacts=(), limitations=()):
    batch, requests, batch_sha = load_batch(batch_path, output_root)
    root = Path(output_root)

    def artifact(path):
        path = Path(path)
        require(path.resolve().is_relative_to(root.resolve()), 'artifact outside assigned output root')
        return dict(path=str(path), sha256=sha(path.read_bytes()))

    rows = []
    for identifier, request in requests.items():
        annotation, receipt = root / (identifier + '.first.json'), root / (identifier + '.receipt.json')
        rows.append(dict(request_id=identifier, expected_request_sha256=request_sha(request),
            annotation=artifact(annotation) if annotation.exists() else None,
            submission_receipt=artifact(receipt) if receipt.exists() else None))
    events = root / 'submission_events.jsonl'
    result = dict(schema=FIRST_SCHEMA, batch_sha256=batch_sha, batch_id=batch['batch_id'], output_root=batch['output_root'],
        positions=rows, requested_count=len(requests), submission_events=artifact(events) if events.exists() else None,
        calculator_artifacts=[artifact(p) for p in calculator_artifacts], limitations=list(limitations),
        submission_helper_sha256=sha(Path(__file__).read_bytes()), semantic_assessment_performed=False)
    exclusive(root / 'first_batch.json', encoded(result) + b'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--request-id')
    parser.add_argument('--submit-manual', type=Path)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--calculator-artifact', type=Path, action='append', default=[])
    parser.add_argument('--limitation', action='append', default=[])
    args = parser.parse_args()
    require(bool(args.submit_manual) != args.finalize, 'choose manual submission or finalization')
    if args.finalize:
        finalize(args.batch, args.output_root, calculator_artifacts=args.calculator_artifact, limitations=args.limitation)
    else:
        try:
            raw = args.submit_manual.read_bytes()
            annotation = decoded(raw)
        except (ValueError, OSError) as exc:
            log_event(args.output_root, args.request_id, error=type(exc).__name__ + ': ' + str(exc))
            raise
        submit(args.batch, args.output_root, args.request_id, annotation, raw_bytes=raw)


if __name__ == '__main__':
    main()
