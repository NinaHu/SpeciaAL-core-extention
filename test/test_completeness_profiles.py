import unittest

from process_completeness_estimation.estimation.metrics import completeness, coverage


class TestCompleteness(unittest.TestCase):
    def test_completeness_no_observations(self):
        self.assertEqual(completeness({}), 0.0)

    def test_completeness_complete_observations(self):
        self.assertEqual(completeness({"A": 10, "B": 5, "C": 2, "D": 2}), 1.0)

    def test_completeness_incomplete_observations(self):
        self.assertLess(completeness({"A": 10, "B": 5, "C": 2, "D": 1}), 1.0)
        self.assertGreater(completeness({"A": 10, "B": 5, "C": 2, "D": 1}), 0.0)

    def test_completeness_comparing_observations(self):
        self.assertLess(completeness({"A": 10, "B": 5, "C": 1, "D": 1}),
                        completeness({"A": 10, "B": 5, "C": 2, "D": 1}))


class TestCoverage(unittest.TestCase):
    def test_coverage_no_observations(self):
        self.assertEqual(coverage({}, 0), 0.0)

    def test_coverage_complete_observations_abundance(self):
        self.assertEqual(coverage({"A": 10, "B": 5, "C": 2, "D": 2}, 19), 1.0)

    def test_coverage_complete_observations_incidence(self):
        self.assertEqual(coverage({"A": 10, "B": 5, "C": 2, "D": 2}, 10), 1.0)

    def test_coverage_incomplete_observations_abundance(self):
        self.assertLess(coverage({"A": 10, "B": 5, "C": 2, "D": 1}, 19), 1.0)
        self.assertGreater(coverage({"A": 10, "B": 5, "C": 2, "D": 1}, 19), 0.0)

    def test_coverage_incomplete_observations_incidence(self):
        self.assertLess(coverage({"A": 10, "B": 5, "C": 2, "D": 1}, 10), 1.0)
        self.assertGreater(coverage({"A": 10, "B": 5, "C": 2, "D": 1}, 10), 0.0)

    def test_coverage_comparing_observations_abundance(self):
        self.assertLess(coverage({"A": 10, "B": 5, "C": 1, "D": 1}, 19),
                        coverage({"A": 10, "B": 5, "C": 2, "D": 1}, 19))

    def test_coverage_comparing_observations_incidence_same_sample_size(self):
        self.assertLess(coverage({"A": 10, "B": 5, "C": 1, "D": 1}, 10),
                        coverage({"A": 10, "B": 5, "C": 2, "D": 1}, 10))

    def test_coverage_comparing_observations_incidence_same_observations(self):
        self.assertLess(coverage({"A": 10, "B": 5, "C": 2, "D": 1}, 15),
                        coverage({"A": 10, "B": 5, "C": 2, "D": 1}, 10))