from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attcon.prospective import *


class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.serial = make_splits(2503, 'serial')['validation'].subset(torch.arange(108))
        self.routing = make_splits(2503, 'routing')['validation'].subset(torch.arange(108))

    def test_quality_is_crossed_and_root_query_does_not_leak_it(self):
        for data in (self.serial, self.routing):
            self.assertTrue(torch.equal(data.value[::2], data.value[1::2]))
            self.assertTrue(torch.equal(data.condition[::2], data.condition[1::2]))
            self.assertTrue(torch.equal(root_events(data)[::2, -1], root_events(data)[1::2, -1]))
            self.assertTrue((root_events(data)[:, 4, 12] == data.quality).all())

    def test_architecture_counts_and_family_initialization_match(self):
        from attcon.regulation import weight_fingerprint
        for architecture, expected in (('gru',13704), ('rnn',13683)):
            hashes = []
            for family in ('state','blind','cue'):
                torch.manual_seed(4)
                agent = ProspectiveAgent(architecture, family)
                self.assertEqual(sum(p.numel() for p in agent.parameters()), expected)
                hashes.append(weight_fingerprint(agent))
            self.assertEqual(len(set(hashes)), 1)

    def test_serial_oracle_changes_verification_with_quality_at_same_confidence(self):
        r = analytic_policy(self.serial)
        mask = (self.serial.condition != 0) & (self.serial.cost == .25)
        self.assertTrue((r['inspections'][mask & (self.serial.quality < .7)] == 2).all())
        self.assertTrue((r['inspections'][mask & (self.serial.quality > .7)] == 1).all())

    def test_routing_oracle_selects_high_quality_source(self):
        r = analytic_policy(self.routing)
        mask = self.routing.condition != 0
        self.assertTrue((r['initial_action'][mask & (self.routing.quality < .7)] == 8).all())
        self.assertTrue((r['initial_action'][mask & (self.routing.quality > .7)] == 7).all())

    def test_forced_counts_and_environment_returns(self):
        agent = ProspectiveAgent()
        for data in (self.serial, self.routing):
            for forced in (0,1,2):
                r = rollout(agent, data, forced=forced)
                expected = forced if data.task == 'serial' else int(forced > 0)
                self.assertTrue((r['inspections'] == expected).all())
                self.assertTrue(torch.allclose(r['return'], r['correct'] - expected * data.cost))
                self.assertTrue((r['answer'] < 6).all())

    def test_sensor_misspecification_keeps_displayed_cues_unchanged(self):
        baseline = make_splits(2503,'routing')['stress']
        shifted = make_splits(2503,'routing',.15)['stress']
        self.assertTrue(torch.equal(root_events(baseline), root_events(shifted)))
        self.assertTrue(torch.allclose(baseline.actual_a - shifted.actual_a, torch.full_like(baseline.quality,.15)))

    def test_terminal_nodes_cannot_acquire(self):
        agent = ProspectiveAgent()
        for data in (self.serial,self.routing):
            state = states_for(agent,data)[2]
            _, values = action_values(agent,state,data,2,native=True)
            self.assertTrue((values[:,7:] < -1e8).all())
            self.assertTrue((values[:,6] == .30).all())

if __name__ == '__main__':
    unittest.main()
