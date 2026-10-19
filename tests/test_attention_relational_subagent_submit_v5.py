"""Standalone helper fixtures: structural assistance, never semantic feedback."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_subagent_submit_v5 as submitter
import attention_relational_grounded_auditor_v4 as core


def fictional_request(identifier='fictional_request'):
    red = [[0.0] * 4 for _ in range(4)]
    blue = deepcopy(red)
    red[0][0], blue[3][3] = 1.0, 1.0
    packet = dict(schema='attention_relational_report_v1_packet', panels=[dict(panel='p0', candidates=None,
        current=[dict(node='n0', origin=dict(kind='history_estimate', represented_time=12, feedback_through=11),
                      distributions=dict(red=red, blue=blue))])])
    return core.build_request(identifier, 'The fictional red peak is at row 0, column 0.', packet)


def manual_annotation(request, verdict='supported'):
    payload = submitter.decoded(request['input'].rsplit('\n', 1)[1])
    context = next(c['context_id'] for c in payload['source_contexts'] if c['kind'] == 'role_distribution')
    return submitter.make_annotation(payload['unit_id'],
        [dict(assertion_id='a0', paraphrase='Fictional manual assertion.', span_ids=[payload['spans'][0]['span_id']],
              source_context_ids=[context], verdict=verdict, reason='Fictional reason; not a semantic key.')],
        [dict(span_id=payload['spans'][0]['span_id'], assertion_ids=['a0'], classification='assertive', reason='Manual.')],
        complete_assertion_inventory=True, unresolved_ambiguity=False)


def batch_fixture(tmp_path):
    request = fictional_request()
    root = tmp_path / 'own-output'
    batch = dict(schema=submitter.BATCH_SCHEMA, batch_id='fictional_batch', output_root=str(root),
        worker_instructions='Fictional worker instructions.', annotation_schema=core.response_schema(),
        bindings=dict(submission_helper_sha256=submitter.sha(Path(submitter.__file__).read_bytes()),
            worker_instructions_sha256=submitter.sha(b'Fictional worker instructions.'),
            response_schema_content_sha256=submitter.sha(submitter.encoded(core.response_schema()))), requests=[request])
    path = tmp_path / 'own-public-batch.json'
    path.write_bytes(submitter.encoded(batch) + b'\n')
    return path, root, request


def test_helper_imports_only_stdlib_and_never_calls_frozen_validator():
    tree = ast.parse(Path(submitter.__file__).read_text())
    modules = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    modules |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert modules <= {'argparse', 'datetime', 'hashlib', 'json', 'os', 'pathlib', 're'}
    assert all(not isinstance(n, ast.Attribute) or n.attr != 'derive_annotation' for n in ast.walk(tree))


def test_semantically_wrong_verdict_is_retained_without_feedback_and_manual_bytes_are_exact(tmp_path):
    path, root, request = batch_fixture(tmp_path)
    annotation = manual_annotation(request, verdict='contradicted')
    raw = json.dumps(annotation, indent=2).encode() + b'\n'
    receipt = submitter.submit(path, root, request['id'], annotation, raw_bytes=raw)
    assert (root / (request['id'] + '.first.json')).read_bytes() == raw
    assert receipt['annotation_sha256'] == submitter.sha(raw)
    events = [submitter.decoded(line) for line in (root / 'submission_events.jsonl').read_bytes().splitlines()]
    assert events[0]['event'] == 'first_submission' and not events[0]['semantic_feedback_provided']
    assert 'supportedness' not in receipt and 'correct' not in receipt


def test_all_interface_failures_logged_and_first_submission_cannot_be_repaired(tmp_path):
    path, root, request = batch_fixture(tmp_path)
    annotation = manual_annotation(request)
    invalid = deepcopy(annotation)
    invalid['assertions'][0]['source_context_ids'] = ['unknown_context']
    with pytest.raises(ValueError, match='illegal'):
        submitter.submit(path, root, request['id'], invalid)
    assert not (root / (request['id'] + '.first.json')).exists()
    with pytest.raises(ValueError, match='not in the assigned batch'):
        submitter.submit(path, root, 'foreign_request', annotation)
    submitter.submit(path, root, request['id'], annotation)
    first = (root / (request['id'] + '.first.json')).read_bytes()
    with pytest.raises(ValueError, match='no resubmission'):
        submitter.submit(path, root, request['id'], manual_annotation(request, 'not_auditable'))
    assert (root / (request['id'] + '.first.json')).read_bytes() == first
    events = [submitter.decoded(line) for line in (root / 'submission_events.jsonl').read_bytes().splitlines()]
    assert len(events) == 4 and sum(e['event'] == 'interface_failure' for e in events) == 3
    assert all(e['semantic_feedback_provided'] is False for e in events)


def test_finalize_binds_every_position_calculator_and_errors_without_quality_checks(tmp_path):
    path, root, request = batch_fixture(tmp_path)
    invalid = manual_annotation(request)
    invalid['complete_assertion_inventory'] = 'yes'
    with pytest.raises(ValueError):
        submitter.submit(path, root, request['id'], invalid)
    calculator = root / 'own_calculation.py'
    calculator.write_text('# Fictional independent calculation.\n')
    first = submitter.finalize(path, root, calculator_artifacts=[calculator], limitations=['Fictional missing annotation.'])
    assert first['positions'][0]['annotation'] is None
    assert first['requested_count'] == 1 and first['submission_events']['sha256']
    assert first['calculator_artifacts'][0]['sha256'] == submitter.sha(calculator.read_bytes())
    assert not first['semantic_assessment_performed']
    before = (root / 'first_batch.json').read_bytes()
    with pytest.raises(FileExistsError):
        submitter.finalize(path, root)
    with pytest.raises(ValueError, match='finalized'):
        submitter.submit(path, root, request['id'], manual_annotation(request))
    assert (root / 'first_batch.json').read_bytes() == before
    assert (root / 'after_finalization_events.jsonl').exists()


def test_shape_only_assistance_does_not_fill_coverage_or_choose_inventory_flags(tmp_path):
    path, root, request = batch_fixture(tmp_path)
    payload = submitter.decoded(request['input'].rsplit('\n', 1)[1])
    annotation = submitter.make_annotation(payload['unit_id'], [], [], complete_assertion_inventory=False, unresolved_ambiguity=True)
    submitter.submit(path, root, request['id'], annotation)
    stored = submitter.decoded((root / (request['id'] + '.first.json')).read_bytes())
    assert stored == annotation and stored['coverage'] == [] and not stored['complete_assertion_inventory']
    assert stored['unresolved_ambiguity'] is True
