"""Handcrafted first-attempt retention checks; no experiment data or network."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


path = Path(__file__).resolve().parents[1]/'scripts/run_attention_relational_reports_v1.py'
spec = importlib.util.spec_from_file_location('relational_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeResponses:
    def __init__(self, result=None, failure=None):
        self.calls = []
        self.result = result
        self.failure = failure

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return SimpleNamespace(output_text=self.result, status='incomplete',
                               model_dump=lambda **_: {'status': 'incomplete', 'raw': self.result})


class FirstAttemptTests(unittest.TestCase):
    def manifest(self, schema=None):
        return {'stage': 'report', 'settings': {
            'model': 'handcrafted-fixture', 'reasoning_effort': 'medium',
            'max_output_tokens': 12, 'timeout_seconds': 10, 'concurrency': 1,
            'output_schema': schema}}

    def run_batch(self, root, client, records=None, schema=None):
        return asyncio.run(runner.execute(self.manifest(schema), records or [
            {'id': 'p001', 'input': 'neutral record'}], 'a'*64, 'b'*40, root, client))

    def test_incomplete_raw_response_and_parse_failure_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            responses = FakeResponses(result='unfinished {')
            self.run_batch(tmp, SimpleNamespace(responses=responses), schema={'type': 'object'})
            saved = json.loads((Path(tmp)/'p001.json').read_text())
            self.assertEqual(saved['response']['raw'], 'unfinished {')
            self.assertEqual(saved['api_response_status'], 'incomplete')
            self.assertTrue(saved['parse_error'])
            self.assertEqual(len(responses.calls), 1)
            self.assertFalse(responses.calls[0]['store'])
            self.assertNotIn('previous_response_id', responses.calls[0])
            self.assertNotIn('conversation', responses.calls[0])

    def test_failed_attempt_is_retained_without_retry_or_exception_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            responses = FakeResponses(failure=RuntimeError('secret fixture string'))
            self.run_batch(tmp, SimpleNamespace(responses=responses))
            raw = (Path(tmp)/'p001.json').read_text()
            self.assertEqual(json.loads(raw)['status'], 'error')
            self.assertNotIn('secret fixture string', raw)
            self.assertEqual(len(responses.calls), 1)
            self.assertTrue((Path(tmp)/'first_completion.json').exists())

    def test_existing_first_batch_rejects_before_any_api_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            responses = FakeResponses(result='first')
            client = SimpleNamespace(responses=responses)
            self.run_batch(tmp, client)
            original = (Path(tmp)/'p001.json').read_bytes()
            with self.assertRaises(FileExistsError):
                self.run_batch(tmp, client)
            self.assertEqual(len(responses.calls), 1)
            self.assertEqual((Path(tmp)/'p001.json').read_bytes(), original)

    def test_identical_restored_inputs_remain_separate_recorded_opportunities(self):
        with tempfile.TemporaryDirectory() as tmp:
            responses = FakeResponses(result='same input independent wording')
            records = [{'id': 'baseline', 'input': 'x'}, {'id': 'restored', 'input': 'x'}]
            outcomes = self.run_batch(tmp, SimpleNamespace(responses=responses), records)
            self.assertEqual([v['id'] for v in outcomes], ['baseline', 'restored'])
            self.assertEqual([v['input'] for v in responses.calls], ['x', 'x'])
            self.assertEqual(json.loads((Path(tmp)/'first_completion.json').read_text())['request_count'], 2)


if __name__ == '__main__':
    unittest.main()
