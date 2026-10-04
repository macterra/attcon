import unittest
from attcon.functional_wiring import CONDITIONS, ORDERS, fixture, identify


class FunctionalWiringTests(unittest.TestCase):
    def test_identical_initial_states_do_not_reveal_condition(self):
        for order in ORDERS:
            records = [fixture(15, c, order, True) for c in CONDITIONS]
            self.assertEqual(records[0], records[1])
            self.assertEqual(records[0], records[2])
            self.assertEqual(identify(records[0]), 'insufficient_evidence')

    def test_physical_command_effects_and_readout(self):
        for condition in CONDITIONS:
            record = fixture(15, condition)
            before = record['trials'][0]['nodes']
            after = record['trials'][1]['nodes']
            self.assertEqual(after[0]['values'], after[2]['values'])
            changed = [before[i]['values'] != after[i]['values'] for i in range(3)]
            self.assertEqual(changed, {'own_access': [True, False, True],
                'external_device': [False, True, False], 'decoupled': [False]*3}[condition])

    def test_identifier_permutations_preserve_identifiability(self):
        for seed in range(12):
            for condition in CONDITIONS:
                for order in ORDERS:
                    self.assertEqual(identify(fixture(seed, condition, order)), condition)


if __name__ == '__main__':
    unittest.main()
