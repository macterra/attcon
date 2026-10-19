"""Lossless repository evidence storage; analysis continues to use original bytes.

Archive never changes/deletes the source or runs Git. Restore never overwrites a
different file. No credentials, API, model, or scientific scoring is involved.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'attcon_lossless_evidence_archive_v1'
BLOCK = 1024 * 1024


def checksum(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def within(root, name):
    root = Path(root).resolve()
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('repository-relative evidence path required')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root / 'audits'):
        raise ValueError('evidence must remain within repository audits')
    return resolved


def archive(root, source_name):
    source = within(root, source_name)
    compressed, metadata = Path(str(source) + '.gz'), Path(str(source) + '.archive.json')
    if compressed.exists() or metadata.exists():
        raise FileExistsError('archive already exists; refusing replacement')
    before_sha, before_size = checksum(source)
    read_digest, read_size = hashlib.sha256(), 0
    with source.open('rb') as original, compressed.open('xb') as target:
        with gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0, compresslevel=9) as output:
            for block in iter(lambda: original.read(BLOCK), b''):
                output.write(block)
                read_digest.update(block)
                read_size += len(block)
    after = checksum(source)
    if (read_digest.hexdigest(), read_size) != (before_sha, before_size) or after != (before_sha, before_size):
        raise ValueError('source changed during archival; original and failed archive retained')
    check, count = hashlib.sha256(), 0
    with gzip.open(compressed, 'rb') as restored:
        for block in iter(lambda: restored.read(BLOCK), b''):
            check.update(block)
            count += len(block)
    if (check.hexdigest(), count) != (before_sha, before_size):
        raise ValueError('archive does not reproduce original bytes')
    compressed_sha, compressed_size = checksum(compressed)
    result = dict(schema=SCHEMA, created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_path=str(source.relative_to(Path(root).resolve())), source_sha256=before_sha,
        source_size_bytes=before_size, gzip_path=str(compressed.relative_to(Path(root).resolve())),
        gzip_sha256=compressed_sha, gzip_size_bytes=compressed_size,
        gzip_mtime=0, exact_decompression_verified=True, source_modified_or_deleted=False,
        scientific_bytes_or_criteria_changed=False)
    with metadata.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return result


def restore(root, metadata_name):
    metadata = within(root, metadata_name)
    data = json.loads(metadata.read_text())
    if data.get('schema') != SCHEMA or data.get('exact_decompression_verified') is not True:
        raise ValueError('wrong or unverified archive metadata')
    target, compressed = within(root, data['source_path']), within(root, data['gzip_path'])
    if target == compressed or metadata in (target, compressed):
        raise ValueError('archive paths alias one another')
    if checksum(compressed) != (data['gzip_sha256'], data['gzip_size_bytes']):
        raise ValueError('compressed archive checksum/size differs')
    expected = data['source_sha256'], data['source_size_bytes']
    if target.exists():
        if checksum(target) != expected:
            raise FileExistsError('existing scientific file differs; refusing overwrite')
        return dict(source_path=data['source_path'], source_sha256=expected[0], already_present=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(target) + '.restore.partial')
    check, count = hashlib.sha256(), 0
    created_partial = False
    try:
        with partial.open('xb') as output:
            created_partial = True
            with gzip.open(compressed, 'rb') as original:
                for block in iter(lambda: original.read(BLOCK), b''):
                    count += len(block)
                    if count > expected[1]:
                        raise ValueError('decompressed content exceeds declared size')
                    output.write(block)
                    check.update(block)
        if (check.hexdigest(), count) != expected:
            raise ValueError('decompressed bytes differ from scientific artifact')
        os.link(partial, target)  # Atomic, fails if another writer created target.
    finally:
        # Only our newly created disposable copy is removed; source/archive stay.
        if created_partial and partial.exists():
            partial.unlink()
    return dict(source_path=data['source_path'], source_sha256=expected[0], already_present=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--archive', help='repository-relative completed artifact')
    action.add_argument('--restore', help='repository-relative .archive.json sidecar')
    args = parser.parse_args()
    result = archive(ROOT, args.archive) if args.archive else restore(ROOT, args.restore)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
