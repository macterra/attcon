from pathlib import Path
import hashlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import preserve_large_evidence_v1 as evidence


def original(root, name='audits/fiction.json', payload=None):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if payload is not None else ('{"text":"Préface 🟦"}\n' * 1000).encode())
    return path


def test_exact_round_trip_and_original_retention(tmp_path):
    source = original(tmp_path)
    exact = source.read_bytes()
    result = evidence.archive(tmp_path, 'audits/fiction.json')
    assert source.read_bytes() == exact
    assert result['source_sha256'] == hashlib.sha256(exact).hexdigest()
    assert result['gzip_size_bytes'] < result['source_size_bytes']
    source.unlink()  # Simulate a fresh checkout containing only archive/sidecar.
    assert not evidence.restore(tmp_path, 'audits/fiction.json.archive.json')['already_present']
    assert source.read_bytes() == exact
    assert evidence.restore(tmp_path, 'audits/fiction.json.archive.json')['already_present']


def test_deterministic_compression_and_no_replacement(tmp_path):
    a, b = original(tmp_path, 'audits/a.json'), original(tmp_path, 'audits/b.json')
    evidence.archive(tmp_path, 'audits/a.json')
    evidence.archive(tmp_path, 'audits/b.json')
    assert Path(str(a) + '.gz').read_bytes() == Path(str(b) + '.gz').read_bytes()
    with pytest.raises(FileExistsError):
        evidence.archive(tmp_path, 'audits/a.json')


def test_existing_different_file_is_never_overwritten(tmp_path):
    source = original(tmp_path)
    evidence.archive(tmp_path, 'audits/fiction.json')
    source.write_bytes(b'other work')
    with pytest.raises(FileExistsError, match='refusing overwrite'):
        evidence.restore(tmp_path, 'audits/fiction.json.archive.json')
    assert source.read_bytes() == b'other work'


def test_tampering_fails_before_creating_destination(tmp_path):
    source = original(tmp_path)
    evidence.archive(tmp_path, 'audits/fiction.json')
    source.unlink()
    compressed = Path(str(source) + '.gz')
    compressed.write_bytes(compressed.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        evidence.restore(tmp_path, 'audits/fiction.json.archive.json')
    assert not source.exists()


def test_uncompressed_hash_and_size_are_checked_before_install(tmp_path):
    source = original(tmp_path)
    evidence.archive(tmp_path, 'audits/fiction.json')
    source.unlink()
    metadata = Path(str(source) + '.archive.json')
    value = json.loads(metadata.read_text())
    value['source_sha256'] = '0' * 64
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='scientific artifact'):
        evidence.restore(tmp_path, 'audits/fiction.json.archive.json')
    assert not source.exists() and not Path(str(source) + '.restore.partial').exists()


def test_preexisting_partial_is_preserved(tmp_path):
    source = original(tmp_path)
    evidence.archive(tmp_path, 'audits/fiction.json')
    source.unlink()
    partial = Path(str(source) + '.restore.partial')
    partial.write_bytes(b'another writer')
    with pytest.raises(FileExistsError):
        evidence.restore(tmp_path, 'audits/fiction.json.archive.json')
    assert partial.read_bytes() == b'another writer'
    assert not source.exists()


@pytest.mark.parametrize('path', ['../outside.json', '/tmp/outside.json', 'docs/file.json'])
def test_paths_remain_inside_repository_audits(tmp_path, path):
    with pytest.raises(ValueError):
        evidence.archive(tmp_path, path)
