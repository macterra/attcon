"""Read bounded source views and preserve first working judgments on disk.

This formatting helper performs no semantic or schema grading. A first working
record cannot be overwritten, including when it is malformed. Final assembly
copies all 24 first records once; it never supplies a missing judgment. Actual
launch, terminal observation, source independence and semantic inspection remain
the commissioning agent's responsibilities under the frozen review protocol.
"""
import argparse
import json
from pathlib import Path

import attention_relational_reporting_pipeline_v1 as pipeline
import attention_relational_template_review_v1 as preparation


VERSION = 'attention_relational_template_review_workspace_v1'


def bound_chunk(path, sha):
    chunk = pipeline.bound_json(path, sha)
    rows = chunk['records']
    pipeline.require(chunk['schema'] == preparation.VERSION + '_chunk'
        and len(rows) == 24 and len({r['id'] for r in rows}) == 24
        and chunk['ordered_template_ids'] == [r['id'] for r in rows],
        'assigned complete 24-record chunk required')
    return chunk


def resolve(value, path):
    for part in path:
        if type(value) is dict and type(part) is str:
            value = value[part]
        elif type(value) is list and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError('exact source path does not resolve')
    return value


def source_view(chunk, identifier, path, offset=0, limit=20):
    """Exact subtree pages, exposing complete counts and continuation explicitly."""
    pipeline.require(type(offset) is int and offset >= 0
        and type(limit) is int and 1 <= limit <= 10000, 'bounded nonnegative page required')
    matches = [r for r in chunk['records'] if r['id'] == identifier]
    pipeline.require(len(matches) == 1 and type(path) is list, 'assigned record and JSON path required')
    value = resolve(matches[0], path)
    if type(value) in (dict, list, str):
        pipeline.require(offset <= len(value), 'page offset exceeds complete source length')
    result = dict(schema=VERSION + '_source_view', id=identifier, source_path=path,
        complete_record_count=len(chunk['records']), offset=offset, limit=limit,
        semantic_selection_performed=False)
    if type(value) is dict:
        entries = list(value.items())
        selected = entries[offset:offset + limit]
        result.update(kind='object', total_items=len(entries), value=dict(selected))
    elif type(value) is list:
        result.update(kind='array', total_items=len(value), value=value[offset:offset + limit])
    elif type(value) is str:
        result.update(kind='string', total_items=len(value), value=value[offset:offset + limit])
    else:
        pipeline.require(offset == 0, 'scalar has no later page')
        result.update(kind='scalar', total_items=1, value=value)
    end = min(offset + limit, result['total_items'])
    result.update(page_end=end, complete=offset == 0 and end == result['total_items'],
        next_offset=end if end < result['total_items'] else None)
    return result


def first_record_path(workspace, identifier):
    return Path(workspace) / ('record_' + pipeline.digest(identifier.encode('utf-8')) + '.json')


def save_first_record(chunk, identifier, workspace, raw):
    pipeline.require(identifier in chunk['ordered_template_ids'], 'record is outside assigned chunk')
    path = first_record_path(workspace, identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    # Do not parse/grade or provide a repair loop after saving a first judgment.
    return dict(path=str(path), sha256=pipeline.digest(raw), bytes=len(raw),
                schema_or_semantic_validation_performed=False)


def assemble_first(chunk, chunk_sha, workspace, envelope, output):
    """Copy unchanged first judgments; fail without inventing absent records."""
    pipeline.require(type(envelope) is dict and 'records' not in envelope
        and envelope.get('chunk_sha256') == chunk_sha,
        'bound envelope without supplied records required')
    rows = []
    for identifier in chunk['ordered_template_ids']:
        raw = first_record_path(workspace, identifier).read_bytes()
        # Parsing is necessary for JSON assembly, but no judgment is edited.
        rows.append(pipeline.decoded(raw))
    value = dict(envelope, records=rows)
    pipeline.exclusive_json(output, value)
    raw = Path(output).read_bytes()
    return dict(path=str(output), sha256=pipeline.digest(raw), records=len(rows),
                schema_or_semantic_validation_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk', type=Path, required=True)
    parser.add_argument('--chunk-sha256', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    view = sub.add_parser('view')
    view.add_argument('--id', required=True)
    view.add_argument('--path', default='[]')
    view.add_argument('--offset', type=int, default=0)
    view.add_argument('--limit', type=int, default=20)
    save = sub.add_parser('save-first-record')
    save.add_argument('--id', required=True)
    save.add_argument('--input', type=Path, required=True)
    save.add_argument('--workspace', type=Path, required=True)
    assemble = sub.add_parser('assemble-first')
    assemble.add_argument('--workspace', type=Path, required=True)
    assemble.add_argument('--envelope', type=Path, required=True)
    assemble.add_argument('--envelope-sha256', required=True)
    assemble.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    chunk = bound_chunk(args.chunk, args.chunk_sha256)
    if args.command == 'view':
        result = source_view(chunk, args.id, pipeline.decoded(args.path), args.offset, args.limit)
    elif args.command == 'save-first-record':
        result = save_first_record(chunk, args.id, args.workspace, args.input.read_bytes())
    else:
        envelope = pipeline.bound_json(args.envelope, args.envelope_sha256)
        result = assemble_first(chunk, args.chunk_sha256, args.workspace, envelope, args.output)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
