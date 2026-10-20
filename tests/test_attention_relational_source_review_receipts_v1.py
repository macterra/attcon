"""Fictional observation histories: missing evidence must not become review data."""
from copy import deepcopy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_source_review_receipts_v1 as receipts


def fiction():
    assignments, plans, spawns, terminals = [], [], [], {}
    for stream in range(2):
        for number in range(23):
            context = f'/fiction/stream{stream}/chunk{number}'
            chunk = dict(path=f'fiction/chunk{number}.json', sha256=f'{number:02x}' * 32)
            artifact = f'fiction/stream{stream}/chunk{number}/first.json'
            assignment = dict(kind='template', independent_context_id=context,
                reviewer_id=f'reviewer_{stream}', chunk_number=number,
                chunk=chunk, first_artifact_path=artifact)
            assignments.append(assignment)
            plans.append(dict(reviewer_id=assignment['reviewer_id'],
                independent_context_id=context, chunk_number=number,
                chunk_sha256=chunk['sha256'], first_artifact_path=artifact,
                fresh_context=True, first_pass=True, peer_judgments_supplied=False))
            spawns.append(dict(independent_context_id=context,
                spawn_tool_result=dict(task_name=context), observed_status='running',
                requested_fork_turns='none', requested_model_override=None,
                requested_reasoning_effort_override=None, assigned_chunk=chunk,
                assigned_first_artifact_path=artifact))
            terminals[context] = (dict(path=f'fiction/terminal{stream}_{number}.json', sha256='d' * 64),
                dict(schema=receipts.TERMINAL_SCHEMA, independent_context_id=context,
                    chunk_sha256=chunk['sha256'], observed_terminal=True,
                    observation_kind='completed',
                    first_artifact=dict(path=artifact, sha256='e' * 64)))
    planned = dict(schema=receipts.COLLECTOR_VERSION + '_launch',
                   manifest_sha256='a' * 64, contexts=plans)
    launches = [(dict(path='fiction/launch.json', sha256='b' * 64),
                 dict(schema=receipts.LAUNCH_SCHEMA, assignments_sha256=receipts.ASSIGNMENTS_SHA,
                      semantic_judgments_or_peer_feedback_supplied=False,
                      observed_spawn_calls=spawns))]
    return assignments, planned, launches, terminals


class ObservationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.assignments, self.planned, self.launches, self.terminals = fiction()

    def assemble(self):
        return receipts.assemble(self.assignments, self.planned, 'c' * 64,
                                 self.launches, self.terminals)

    def test_complete_history_retains_exact_46_descriptors(self):
        result = self.assemble()
        self.assertTrue(result['ready'])
        contexts = result['outcome_receipt']['contexts']
        self.assertEqual(len(contexts), 46)
        self.assertEqual({c['independent_context_id'] for c in contexts}, set(self.terminals))
        for context in contexts:
            original = self.terminals[context['independent_context_id']][1]['first_artifact']
            self.assertEqual(context['first_artifact'], original)
        self.assertEqual(set(result['outcome_receipt']),
                         {'schema', 'launch_receipt_sha256', 'manifest_sha256', 'contexts'})

    def test_terminal_failed_task_keeps_null_first_no_replacement(self):
        key = self.assignments[7]['independent_context_id']
        terminal = self.terminals[key][1]
        terminal['first_artifact'] = None
        terminal['review_task_complete'] = False
        result = self.assemble()
        self.assertTrue(result['ready'])
        output = next(c for c in result['outcome_receipt']['contexts'] if c['independent_context_id'] == key)
        self.assertIsNone(output['first_artifact'])
        self.assertEqual(output['terminal_attestation']['observation_kind'], 'completed')

    def test_no_terminal_observation_cannot_use_planned_or_declared_first(self):
        key = self.assignments[0]['independent_context_id']
        del self.terminals[key]
        result = self.assemble()
        self.assertFalse(result['ready'])
        self.assertIsNone(result['outcome_receipt'])
        self.assertEqual(result['pending'], [dict(independent_context_id=key,
                                                 reasons=['root_terminal_observation_missing'])])

    def test_terminal_without_actual_launch_cannot_substitute_plan(self):
        spawn = self.launches[0][1]['observed_spawn_calls'].pop()
        result = self.assemble()
        self.assertFalse(result['ready'])
        self.assertEqual(result['pending'], [dict(independent_context_id=spawn['independent_context_id'],
                                                 reasons=['actual_launch_observation_missing'])])

    def test_wrong_chunk_terminal_refused(self):
        next(iter(self.terminals.values()))[1]['chunk_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'terminal observation binding'):
            self.assemble()

    def test_alternate_first_path_refused(self):
        next(iter(self.terminals.values()))[1]['first_artifact']['path'] = 'fiction/repaired.json'
        with self.assertRaisesRegex(ValueError, 'first artifact descriptor'):
            self.assemble()

    def test_duplicate_launch_is_not_an_additional_independent_review(self):
        spawns = self.launches[0][1]['observed_spawn_calls']
        spawns.append(deepcopy(spawns[0]))
        with self.assertRaisesRegex(ValueError, 'duplicate actual launch'):
            self.assemble()

    def test_out_of_order_launch_refused(self):
        spawns = self.launches[0][1]['observed_spawn_calls']
        spawns[0], spawns[1] = spawns[1], spawns[0]
        with self.assertRaisesRegex(ValueError, 'within-stream fixed launch order'):
            self.assemble()

    def test_unchanged_first_descriptor_does_not_certify_first_bytes(self):
        # The frozen collector must later retain malformed/unreadable firsts as
        # failed measurements. This metadata builder must not drop/rewrite them.
        descriptor = next(iter(self.terminals.values()))[1]['first_artifact']
        descriptor['sha256'] = '0' * 64
        result = self.assemble()
        self.assertTrue(result['ready'])
        self.assertEqual(result['outcome_receipt']['contexts'][0]['first_artifact'], descriptor)

    def test_history_inputs_are_not_mutated(self):
        before = deepcopy((self.assignments, self.planned, self.launches, self.terminals))
        self.assemble()
        self.assertEqual(before, (self.assignments, self.planned, self.launches, self.terminals))


if __name__ == '__main__':
    unittest.main()
