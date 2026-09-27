import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from confirm_predictive_reports import view, truth
from attcon.predictive_attention import Forecast, simulate


class InterfaceTests(unittest.TestCase):
    def test_explicit_target_view_matches_both_channels(self):
        e = simulate(31, 8)
        f = Forecast(e.allocation[:, 7], e.access[:, 7], e.next_allocation[:, 7])
        for i in range(8):
            for channel in range(2):
                for slot in range(4):
                    t = truth(view(f, i, channel, slot))
                    self.assertEqual(t['focal'], bool(e.allocation[i, 7, channel, slot]))
                    self.assertAlmostEqual(t['access_now'], float(e.access[i, 7, 0, channel, slot]), places=4)
                    self.assertEqual(t['responsive_channel'], int(e.controlled[i]))

    def test_missing_and_constant_are_not_positive_labels(self):
        self.assertTrue(all(v is None for v in truth(None).values()))
        f = Forecast(torch.full((1, 2, 4), .25), torch.full((1, 3, 2, 4), .5), torch.full((1, 4, 2, 4), .25))
        t = truth(view(f, 0, 1, 3))
        self.assertIsNone(t['focal']); self.assertIsNone(t['responsive_channel'])
        self.assertEqual(t['access_now'], .5)
