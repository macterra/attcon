import unittest
from scripts.functional_power_feasibility import (
    independent_counts, model_counts, paired_pvalue, pooled_counts, rejection_rate,
    model_sign_probabilities)


class FunctionalPowerFeasibilityTests(unittest.TestCase):
    def test_exact_tail_resolution_and_degenerate_power(self):
        self.assertEqual(paired_pvalue(0, 0), 1)
        self.assertEqual(paired_pvalue(3, 0), 1/8)
        self.assertEqual(paired_pvalue(4, 0), 1/16)
        self.assertEqual(paired_pvalue(5, 0), 1/32)
        # Five purely positive discordances are necessary at n=5, alpha=.05.
        self.assertAlmostEqual(rejection_rate(independent_counts(5, .25, 0)), .25**5)
        self.assertEqual(rejection_rate(independent_counts(48, 0, 0)), 0)

    def test_mixture_variance_matches_derived_intramodel_correlation(self):
        n, positive, negative, rho = 8, .35, .10, .30
        probabilities = model_counts(n, positive, negative, rho)
        mean = sum((b-c)*p for (b,c),p in probabilities.items())
        variance = sum((b-c-mean)**2*p for (b,c),p in probabilities.items())
        self.assertAlmostEqual(mean, n*(positive-negative))
        self.assertAlmostEqual(variance, n*(positive+negative-(positive-negative)**2)*(1+(n-1)*rho))

    def test_independent_model_convolution_equals_independent_pairs(self):
        expected = independent_counts(12, .35, .10)
        actual = pooled_counts(3, model_counts(4, .35, .10, 0))
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected: self.assertAlmostEqual(expected[key], actual[key], places=14)

    def test_shared_outcome_null_inflation_and_model_sign_resolution(self):
        within = model_counts(16, .10, .10, 1)
        # Hand calculation: .192 + .024 + .001 + .003 = .22.
        self.assertAlmostEqual(rejection_rate(pooled_counts(3, within)), .22)
        positive, negative, _ = model_sign_probabilities(within)
        self.assertEqual(rejection_rate(independent_counts(3, positive, negative)), 0)


if __name__ == '__main__':
    unittest.main()
