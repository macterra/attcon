"""Fictional copying/availability checks; no semantic qualification."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_template_review_workspace_v1 as workspace
import attention_relational_reporting_pipeline_v1 as pipeline


def fictional_chunk():
    rows = [dict(id=f'invented_{i}', values=[True, 1, 1.0, None], text='αβγδε',
                 nested=dict(origin=dict(represented_time=11, copied=True))) for i in range(24)]
    return dict(schema=workspace.preparation.VERSION + '_chunk', records=rows,
                ordered_template_ids=[r['id'] for r in rows])


def test_pages_preserve_values_and_complete_inventory_counts():
    chunk = fictional_chunk()
    first = workspace.source_view(chunk, 'invented_0', ['values'], limit=2)
    second = workspace.source_view(chunk, 'invented_0', ['values'], offset=first['next_offset'], limit=2)
    assert first['value'] == [True, 1] and type(first['value'][0]) is bool
    assert second['value'] == [1.0, None] and type(second['value'][0]) is float
    assert first['total_items'] == second['total_items'] == 4
    assert not first['complete'] and second['next_offset'] is None
    text = workspace.source_view(chunk, 'invented_0', ['text'], offset=1, limit=3)
    assert text['value'] == 'βγδ' and text['total_items'] == 5
    exact = workspace.source_view(chunk, 'invented_0', ['nested', 'origin', 'represented_time'])
    assert exact['value'] == 11 and exact['complete']


@pytest.mark.parametrize('path', [['values', True], ['values', -1], ['values', 4], ['absent']])
def test_view_rejects_inexact_or_bool_paths(path):
    with pytest.raises((ValueError, KeyError)):
        workspace.source_view(fictional_chunk(), 'invented_0', path)


def test_chunk_hash_and_complete_inventory(tmp_path):
    path = tmp_path / 'fictional.json'
    pipeline.exclusive_json(path, fictional_chunk())
    sha = pipeline.digest(path.read_bytes())
    assert len(workspace.bound_chunk(path, sha)['records']) == 24
    with pytest.raises(ValueError):
        workspace.bound_chunk(path, '0' * 64)
    chunk = fictional_chunk()
    chunk['records'].pop()
    bad = tmp_path / 'incomplete.json'
    pipeline.exclusive_json(bad, chunk)
    with pytest.raises(ValueError):
        workspace.bound_chunk(bad, pipeline.digest(bad.read_bytes()))


def test_first_bytes_immutable_even_when_malformed(tmp_path):
    chunk = fictional_chunk()
    raw = b'{malformed first judgment'
    receipt = workspace.save_first_record(chunk, 'invented_0', tmp_path, raw)
    assert Path(receipt['path']).read_bytes() == raw
    assert receipt['schema_or_semantic_validation_performed'] is False
    with pytest.raises(FileExistsError):
        workspace.save_first_record(chunk, 'invented_0', tmp_path, b'{}')
    assert Path(receipt['path']).read_bytes() == raw


def test_complete_assembly_copies_judgments_without_semantic_defaults(tmp_path):
    chunk = fictional_chunk()
    expected = []
    for row in chunk['records']:
        value = dict(id=row['id'], assessment='invented uninterpreted judgment', exact=[True, 1, 1.0])
        expected.append(value)
        workspace.save_first_record(chunk, row['id'], tmp_path / 'work', pipeline.encoded(value))
    envelope = dict(chunk_sha256='a' * 64, reviewer_id='invented reviewer')
    output = tmp_path / 'first.json'
    result = workspace.assemble_first(chunk, 'a' * 64, tmp_path / 'work', envelope, output)
    assert pipeline.decoded(output.read_bytes()) == dict(envelope, records=expected)
    assert result['records'] == 24 and result['schema_or_semantic_validation_performed'] is False
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        workspace.assemble_first(chunk, 'a' * 64, tmp_path / 'work', envelope, output)
    assert output.read_bytes() == before


def test_missing_or_malformed_fragment_is_not_filled(tmp_path):
    chunk = fictional_chunk()
    output = tmp_path / 'first.json'
    envelope = dict(chunk_sha256='a' * 64)
    with pytest.raises(FileNotFoundError):
        workspace.assemble_first(chunk, 'a' * 64, tmp_path, envelope, output)
    assert not output.exists()
    workspace.save_first_record(chunk, chunk['records'][0]['id'], tmp_path, b'{bad')
    with pytest.raises(ValueError):
        workspace.assemble_first(chunk, 'a' * 64, tmp_path, envelope, output)
    assert not output.exists()


def test_no_unassigned_record_or_injected_final_records(tmp_path):
    chunk = fictional_chunk()
    with pytest.raises(ValueError):
        workspace.save_first_record(chunk, 'unassigned', tmp_path, b'{}')
    with pytest.raises(ValueError):
        workspace.assemble_first(chunk, 'a' * 64, tmp_path, dict(chunk_sha256='a' * 64, records=[]), tmp_path / 'first.json')
