from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attcon.active_inspection import *


class AcquisitionEnvironmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.splits = make_splits(2309)
        self.data = self.splits['validation']

    def test_context_partitions_are_disjoint_and_fully_crossed(self):
        used = set()
        for data in self.splits.values():
            groups = set(data.group.tolist())
            self.assertFalse(groups & used)
            used |= groups
            self.assertTrue((data.group.unique(return_counts=True)[1] == 54).all())
        self.assertEqual(self.data.fingerprint(), make_splits(2309)['validation'].fingerprint())

    def test_query_hides_condition_and_answer(self):
        query = initial_events(self.data)[:, -1]
        self.assertEqual(query[:, :10].abs().sum().item(), 0)
        self.assertTrue(torch.equal(query[:, 10], self.data.cost))
        self.assertTrue((query[:, 11] == 1).all())

    def test_missing_and_stale_do_not_encode_hidden_answer(self):
        for condition in (1, 2):
            indices = torch.where((self.data.group == self.data.group[0]) & (self.data.condition == condition) & (self.data.cost == self.data.cost[0]))[0]
            events = initial_events(self.data.subset(indices))
            self.assertTrue(torch.equal(events, events[:1].expand_as(events)))

    def test_sensor_and_cost_variants_share_samples(self):
        for value in range(6):
            rows = (self.data.group == self.data.group[0]) & (self.data.value == value)
            self.assertEqual(len(self.data.sample[rows].unique()), 1)
        self.assertTrue(torch.equal(acquired_events(self.data, 2)[:, 0, :6].argmax(-1), self.data.value))

    def test_bayes_stopping_accounts_for_each_cost(self):
        result = analytic_policy(self.data)
        self.assertTrue((result['inspections'][self.data.condition == 0] == 0).all())
        for cost, expected in ((0.1, 2), (0.25, 1), (0.4, 1)):
            mask = (self.data.condition != 0) & (self.data.cost == cost)
            self.assertTrue((result['inspections'][mask] == expected).all())
        self.assertTrue(torch.allclose(result['return'], result['correct'] - self.data.cost * result['inspections']))

    def test_architectures_are_parameter_and_initialization_matched(self):
        weights = []
        for family in ('state','action','confidence'):
            torch.manual_seed(3)
            model = AcquisitionAgent(family)
            weights.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        self.assertTrue(all(torch.equal(weights[0], value) for value in weights[1:]))

    def test_incremental_recurrence_and_forced_cost_accounting(self):
        model = AcquisitionAgent()
        data = self.data.subset(torch.arange(18))
        states = all_states(model, data)
        joined = torch.cat((initial_events(data), acquired_events(data, 1), acquired_events(data, 2)), dim=1)
        self.assertTrue(torch.allclose(states[-1], model.advance(joined), atol=1e-6))
        for count in (0, 1, 2):
            run = rollout(model, data, forced=count)
            self.assertTrue((run['inspections'] == count).all())
            self.assertTrue(torch.allclose(run['return'], run['correct'] - count * data.cost))
            self.assertEqual(run['visited'].sum().item(), len(data) * (count + 1))

    def test_reporting_labels_track_verification_not_choice_success(self):
        self.assertTrue((verified_labels(self.data, 0)[self.data.condition != 0] == 6).all())
        self.assertTrue(torch.equal(verified_labels(self.data, 2), self.data.value))
        self.assertTrue(torch.equal(verified_labels(self.data, 0), verified_labels(self.data, 1)))

if __name__ == '__main__':
    unittest.main()
