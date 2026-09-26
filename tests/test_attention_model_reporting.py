import unittest
import torch
from attcon.attention_model_reporting import (capture, make_contexts, renderer, parse_rendered,
    Reporter, scores, scene_bootstrap)
from attcon.models import RecurrentAttentionController, ModelConfig
from attcon.data import TaskConfig


class AttentionModelReportingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.agent = RecurrentAttentionController(TaskConfig(), ModelConfig(hard_attention=True)).eval()
        # Nonzero feedback tests the actual path even for an untrained test model.
        torch.nn.init.normal_(self.agent.policy_self_model_head.weight, std=.2)
        self.data = make_contexts(TaskConfig(), 107, 'test', 8)

    def test_capture_does_not_change_legacy_outputs(self):
        with torch.no_grad():
            before = self.agent(self.data.scene, self.data.cues[:, 0], cue_seq=self.data.cues)
            traced = capture(self.agent, self.data)
        self.assertTrue(torch.equal(before['attention_seq'], traced['attention']))
        self.assertEqual(len(self.agent.hidden_self_model_head._forward_pre_hooks), 0)

    def test_override_changes_model_and_policy_not_prior_history_or_hidden(self):
        before = capture(self.agent, self.data)
        changed = capture(self.agent, self.data, 1 - before['m'][:, 3])
        self.assertTrue(torch.equal(before['attention'][:, :3], changed['attention'][:, :3]))
        self.assertTrue(torch.equal(before['physical'][:, 3], changed['physical'][:, 3]))
        self.assertTrue(torch.equal(before['hidden'][:, 3], changed['hidden'][:, 3]))
        self.assertTrue(torch.allclose(changed['m'][:, 3], 1 - before['m'][:, 3]))
        self.assertFalse(torch.equal(before['attention'][:, 3], changed['attention'][:, 3]))

    def test_snapshot_has_no_current_action_in_physical_history(self):
        trace = capture(self.agent, self.data)
        self.assertFalse(trace['physical'][:, 0].any())
        expected = torch.zeros(8, 25, dtype=torch.bool)
        for step in range(6):
            self.assertTrue(torch.equal(expected, trace['physical'][:, step]))
            expected[torch.arange(8), trace['attention'][:, step].argmax(-1)] = True

    def test_partitions_and_switches(self):
        fit = make_contexts(TaskConfig(), 107, 'fit', 16)
        self.assertFalse(set(fit.ids) & set(self.data.ids))
        fit_pairs = set(map(tuple, fit.cues[::2][:, [0, 3]].tolist()))
        test_pairs = set(map(tuple, self.data.cues[::2][:, [0, 3]].tolist()))
        self.assertFalse(fit_pairs & test_pairs)

    def test_reporter_state_path_cannot_access_other_features(self):
        trace = capture(self.agent, self.data)
        model = Reporter()
        pred = model.report(trace['features']['state'])
        trace['features']['scene'].normal_()
        trace['features']['answer'].normal_()
        again = model.report(trace['features']['state'])
        self.assertTrue(torch.equal(pred['belief'], again['belief']))
        self.assertTrue(torch.equal(pred['preference'], again['preference']))

    def test_renderer_roundtrip_and_rejects_claims(self):
        truth = torch.arange(25) % 3 == 0
        text = renderer(truth, 12)
        parsed = parse_rendered(text)
        self.assertTrue(torch.equal(truth, parsed['belief']))
        self.assertEqual(parsed['preference'], 12)
        with self.assertRaises(ValueError): parse_rendered('I am conscious.')

    def test_sparse_belief_scores_do_not_reward_all_negative(self):
        truth = {'belief': torch.eye(25).bool(), 'preference': torch.zeros(25, dtype=torch.long)}
        pred = {'belief': torch.zeros(25, 25, dtype=torch.bool), 'preference': truth['preference']}
        metric = scores(pred, truth)
        self.assertEqual(metric['balanced_belief_accuracy'], .5)
        self.assertEqual(metric['exact_report_accuracy'], 0.)

    def test_bootstrap_clusters_timesteps(self):
        a = torch.tensor([0., 1., 0., 1.])
        self.assertEqual(scene_bootstrap(a, 107), scene_bootstrap(a[:, None].expand(-1, 6), 107))

if __name__ == '__main__': unittest.main()

class ThresholdAndArchiveTests(unittest.TestCase):
    def test_exact_95_percent_shift_passes(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from attention_model_study import correspondence
        # 64 mistakes / 1280 transitions, with no true shift.
        truth = {'preference': torch.zeros(256, 6, dtype=torch.long),
                 'belief': torch.zeros(256, 6, 25, dtype=torch.bool),
                 'physical': torch.zeros(256, 6, 25, dtype=torch.bool)}
        pred = {'preference': truth['preference'].clone(), 'belief': truth['belief'].clone()}
        pred['preference'][:64, -1] = 1
        self.assertEqual(correspondence(pred, truth)['preference_shift_agreement'], .95)

    def test_archive_rejects_traversal_before_writing(self):
        import hashlib
        import io
        import sys
        import tarfile
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        import summarize_attention_model as packaging
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'archive.tar.gz'
            with tarfile.open(archive, 'w:gz') as tf:
                member = tarfile.TarInfo('../escape')
                member.size = 1
                tf.addfile(member, io.BytesIO(b'x'))
            reference = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                         'checkpoint_sha256': {name: 'untrusted' for name in packaging.checkpoint_paths()}}
            with patch.object(packaging, 'ROOT', root), patch.object(packaging, 'ARCHIVE', archive):
                with self.assertRaisesRegex(ValueError, 'unexpected archive member'):
                    packaging.restore(reference)
            self.assertFalse((root / 'outputs').exists())

    def test_restore_validates_all_hashes_before_writing(self):
        import hashlib
        import io
        import sys
        import tarfile
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        import summarize_attention_model as packaging
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'archive.tar.gz'
            expected = {}
            with tarfile.open(archive, 'w:gz') as tf:
                for index, name in enumerate(packaging.checkpoint_paths()):
                    content = str(index).encode()
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    tf.addfile(member, io.BytesIO(content))
                    expected[name] = hashlib.sha256(content).hexdigest() if index < 2 else 'wrong'
            reference = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'checkpoint_sha256': expected}
            with patch.object(packaging, 'ROOT', root), patch.object(packaging, 'ARCHIVE', archive):
                with self.assertRaisesRegex(ValueError, 'checkpoint hash mismatch'):
                    packaging.restore(reference)
            self.assertFalse((root / 'outputs').exists())
