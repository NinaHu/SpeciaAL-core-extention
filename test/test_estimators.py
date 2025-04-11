import unittest
import numpy as np
#import math
from collections import defaultdict
from collections import Counter

from special4pm.estimation.metrics import (
    ace, ace_modified, get_abundance_data, ice, ice_modified, analyze_ice_data, analyze_ice_data_modified,
    get_incidence_count, get_species_counts_and_log, jackknife1_abundance,
    jackknife1_incidence, jackknife2_abundance, jackknife2_incidence, iChao1, iChao2, chao1, chao2, calculate_C_ace,
    calculate_C_ice, calculate_gamma_sq_ace, calculate_gamma_sq_ice, calculate_frequencies, calculate_Q_frequencies,
    calculate_gamma_sq_ice_modified, calculate_gamma_sq_ace_modified)


class TestAceEstimators(unittest.TestCase):
    def setUp(self):
        # Standalone datasets for testing
        self.data_normal = [1, 2, 2, 3, 4, 4, 5, 10, 12]  # General case
        self.data_empty = []  # Edge case: empty data
        self.data_all_abundant = [20, 15, 12, 11]  # All species are abundant (>10)
        self.data_only_rare = [10, 9, 8, 7, 6]  # All species are rare (≤10)
        self.data_singletons_only = [1, 1, 1]  # Only singletons
        self.data_no_singletons = [2, 3, 4]  # No singletons
        self.data_one_rare_species = [1]  # Single rare species
        self.data_all_singletons = [1, 1, 1, 1]  # All species are singletons

    def test_ace_normal(self):
        """ACE calculation with normal data"""
        S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund = (
            1, 4, 2, 10, [2, 2, 1, 1, 0, 0, 0, 0, 0, 0]
        )
        result, gamma_sq = ace(S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund)
        self.assertGreater(result, 0, "ACE result should be greater than 0 for normal data")
        self.assertGreaterEqual(gamma_sq, 0, "Gamma² should be non-negative")

    def test_ace_empty(self):
        """ACE with empty data"""
        result, gamma_sq = ace(0, 0, 0, 0, [])
        self.assertEqual(result, 0, "ACE result for empty data should be 0")
        self.assertEqual(gamma_sq, 0, "Gamma² for empty data should be 0")

    def test_ace_all_abundant(self):
        """ACE with all species being abundant ( >10 )"""
        result, gamma_sq = ace(3, 0, 0, 0, [])
        self.assertEqual(result, 3, "ACE should equal the number of abundant species")
        self.assertEqual(gamma_sq, 0, "Gamma² should be 0 when no rare species exist")

    def test_ace_only_rare(self):
        """ACE with only rare species ( ≤10 )"""
        result, gamma_sq = ace(0, 5, 1, 15, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertGreater(result, 5, "ACE should be greater than observed rare species count")
        self.assertGreaterEqual(gamma_sq, 0, "Gamma² should be non-negative")

    def test_ace_singletons_only(self):
        """ACE with only singletons in the dataset"""
        result, gamma_sq = ace(0, 3, 3, 3, [3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result, 3, "ACE should equal observed species richness")
        self.assertEqual(gamma_sq, 0, "Gamma² should be 0 when all species are singletons")

    def test_ace_no_singletons(self):
        """ACE with no singletons in the dataset"""
        result, gamma_sq = ace(0, 3, 0, 9, [0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertGreaterEqual(result, 3, "ACE should be at least equal to rare species count")
        self.assertGreaterEqual(gamma_sq, 0, "Gamma² should be non-negative")

    def test_ace_one_rare_species(self):
        """ACE with a single rare species"""
        result, gamma_sq = ace(0, 1, 1, 1, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result, 1, "ACE should equal observed species richness")
        self.assertEqual(gamma_sq, 0, "Gamma² should be 0 for single species")

    def test_ace_all_singletons(self):
        """ACE with all rare species being singletons"""
        result, gamma_sq = ace(0, 4, 4, 4, [4, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result, 4, "ACE should equal observed species richness")
        self.assertEqual(gamma_sq, 0, "Gamma² should be 0 when all species are singletons")


class TestAceModified(unittest.TestCase):
    def test_ace_modified_normal(self):
        """ACE-modified with a standard dataset"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        result, gamma_sq = ace_modified(1, 5, 2, 10, [2, 2, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertGreater(result, 0, "ACE-modified should be greater than 0 for normal data")
        self.assertGreaterEqual(gamma_sq, 0, "Gamma² should be non-negative")

    def test_ace_modified_empty(self):
        result, gamma_sq = ace_modified(0, 0, 0, 0, [])
        self.assertEqual(result, 0, "ACE-modified should be 0 for empty data")
        self.assertEqual(gamma_sq, 0, "Gamma² should be 0 for empty data")

    def test_ace_modified_high_diversity(self):
        data = [1] * 10 + [2] * 5 + [3] * 3
        Fi_abund = [10, 5, 3] + [0] * 7
        result, gamma_sq = ace_modified(1, 10, 5, 30, Fi_abund)
        self.assertGreater(result, len(set(data)), "ACE-modified should account for high diversity")
        self.assertGreaterEqual(gamma_sq, 0, "Gamma² should be non-negative")

    class TestIceEstimators(unittest.TestCase):
        def setUp(self):
            self.species_counts_3 = {"A": 2, "B": 2, "C": 2, "D": 2, "E": 1, "F": 1}
            self.data_log_3 = [
                ["A", "B"],  # 1x A, 1x B
                ["A", "C"],  # 1x A, 1x C
                ["B", "D"],  # 1x B, 1x D
                ["C", "E"],  # 1x C, 1x E
                ["D", "F"],  # 1x D, 1x F
            ]

            # calculated parameters
            S_freq = sum(1 for species, count in self.species_counts_3.items() if count > 10)  # 0 frequent species
            S_inf = sum(1 for species, count in self.species_counts_3.items() if count <= 10)  # 6 infrequent species
            Q1 = sum(1 for species, count in self.species_counts_3.items() if count == 1)  # 2 singletons (E, F)
            N_inf = sum(count for species, count in self.species_counts_3.items() if
                        count <= 10)  # 10 total counts of infrequent species
            Qj = [sum(1 for species, count in self.species_counts_3.items() if count == j) for j in
                  range(1, 11)]  # [2, 4, 0, 0, ...]

            # number of samples containing infrequent species
            m_inf = sum(
                1 for trace in self.data_log_3
                if any(self.species_counts_3[species] <= 10 for species in trace)
            )  # 5 samples contain infrequent species

            # test ICE estimator
            result, gamma_sq = ice(S_freq, S_inf, Q1, N_inf, Qj, m_inf)
            self.assertGreater(result, 0, "ICE result should be greater than 0 for dataset 3")
            self.assertGreaterEqual(gamma_sq, 0, "Gamma² should be non-negative for dataset 3")


# ICE modified tests
class TestIceModified(unittest.TestCase):
    def setUp(self):
        # Custom datasets for testing
        self.species_counts_1 = [12, 12, 6, 5, 3, 4, 3, 2, 1]
        self.data_log_1 = [
            ["A", "B", "C", "D"],
            ["A", "B", "E", "F"],
            ["B", "G", "H"],
            ["A", "C", "I", "G"],
            ["A", "B", "C", "D"],
        ]

        self.species_counts_2 = [10, 9, 8, 7, 6, 5, 4, 3, 2]
        self.data_log_2 = [
            ["A", "B"], ["C", "D"], ["E", "F"], ["G", "H"], ["I", "A"]
        ]

        self.species_counts_3 = [2, 2, 2, 1, 1, 1]
        self.data_log_3 = [
            ["A", "B"], ["A", "C"], ["B", "D"], ["C", "E"], ["D", "F"]
        ]

    def test_ice_modified_dataset_1(self):
        result, gamma_sq = analyze_ice_data_modified(self.species_counts_1, self.data_log_1)
        self.assertGreater(result, 0, "ICE-modified result should be greater than 0 for dataset 1")
        self.assertGreaterEqual(gamma_sq, 0, "Gamma²-modified should be non-negative for dataset 1")

    def test_ice_modified_dataset_2(self):
        result, gamma_sq = analyze_ice_data_modified(self.species_counts_2, self.data_log_2)
        self.assertGreater(result, 0, "ICE-modified result should be greater than 0 for dataset 2")
        self.assertGreaterEqual(gamma_sq, 0, "Gamma²-modified should be non-negative for dataset 2")

    def test_ice_modified_dataset_3(self):
        result, gamma_sq = analyze_ice_data_modified(self.species_counts_3, self.data_log_3)
        self.assertGreater(result, 0, "ICE-modified result should be greater than 0 for dataset 3")
        self.assertGreaterEqual(gamma_sq, 0, "Gamma²-modified should be non-negative for dataset 3")

    def test_edge_case_empty_data(self):
        with self.assertRaises(ValueError):
            analyze_ice_data_modified([], [])

    def test_individual_component_C_ice(self):
        """C_ice calculation"""
        self.assertAlmostEqual(calculate_C_ice(1, 10), 0.9, "C_ice should be correctly calculated")
        self.assertEqual(calculate_C_ice(0, 0), 0, "C_ice should be 0 when N_inf = 0")

    def test_individual_component_gamma_sq_ice(self):
        """γ²_ice calculation"""
        Qj = [1, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        gamma_sq = calculate_gamma_sq_ice(5, 0.8, 3, 15, Qj)
        self.assertGreaterEqual(gamma_sq, 0, "γ²_ice should be non-negative")

    def test_individual_component_gamma_sq_ice_modified(self):
        """γ²_ice_modified calculation"""
        Qj = [1, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        gamma_sq_mod = calculate_gamma_sq_ice_modified(15, 3, 15, Qj)
        self.assertGreaterEqual(gamma_sq_mod, 0, "γ²_ice_modified should be non-negative")

    def test_edge_case_zero_coverage(self):
        """ICE-modified when C_ice = 0"""
        S_freq = 3
        S_inf = 5
        Q1 = 10
        N_inf = 10
        Qj = [1, 1, 2, 1, 0, 0, 0, 0, 0, 0]
        m_inf = 5
        result, gamma_sq = ice_modified(S_freq, S_inf, Q1, N_inf, Qj, m_inf)
        self.assertEqual(result, S_freq + S_inf, "ICE-modified should fallback to S_freq + S_inf when C_ice = 0")
        self.assertEqual(gamma_sq, 0, "Gamma²-modified should be 0 when C_ice = 0")

    class TestJackknife1Abundance(unittest.TestCase):
        def setUp(self):
            # Custom datasets for testing
            self.data_normal = [10, 15, 20, 25, 30]  # General case
            self.data_with_singletons = [1, 1, 5, 10, 15]  # Includes singletons
            self.data_no_singletons = [5, 10, 15, 20]  # No singletons
            self.data_empty = []  # Empty dataset

        def test_jackknife1_abundance_normal(self):
            """Jackknife-1 with normal abundance-based data"""
            S_obs = len(self.data_normal)
            Q1 = sum(1 for x in self.data_normal if x == 1)
            result = jackknife1_abundance(S_obs, Q1)
            if Q1 > 0:
                self.assertGreater(result, S_obs,
                                   "Jackknife-1 should estimate richness greater than observed when singletons exist")
            else:
                self.assertEqual(result, S_obs, "Jackknife-1 should equal observed richness when no singletons exist")

        def test_jackknife1_abundance_with_singletons(self):
            """Jackknife-1 with abundance-based data including singletons"""
            S_obs = len(self.data_with_singletons)
            Q1 = sum(1 for x in self.data_with_singletons if x == 1)
            result = jackknife1_abundance(S_obs, Q1)
            if Q1 > 0:
                self.assertGreater(result, S_obs,
                                   "Jackknife-1 should estimate richness greater than observed when singletons exist")
            else:
                self.assertEqual(result, S_obs, "Jackknife-1 should equal observed richness when no singletons exist")

        def test_jackknife1_abundance_no_singletons(self):
            """Jackknife-1 with abundance-based data without singletons"""
            S_obs = len(self.data_no_singletons)
            Q1 = sum(1 for x in self.data_no_singletons if x == 1)
            result = jackknife1_abundance(S_obs, Q1)
            self.assertEqual(result, S_obs, "Jackknife-1 should equal observed richness when no singletons exist")

        def test_jackknife1_abundance_empty_data(self):
            """Jackknife-1 with empty abundance-based data"""
            S_obs = 0
            Q1 = 0
            result = jackknife1_abundance(S_obs, Q1)
            self.assertEqual(result, S_obs, "Jackknife-1 should return 0 for empty data")

    class TestJackknife1Incidence(unittest.TestCase):
        def setUp(self):
            # Custom datasets for testing
            self.species_counts_1 = [12, 12, 6, 5, 3, 4, 3, 2, 1]
            self.data_log_1 = [
                ["A", "B", "C", "D"],
                ["A", "B", "E", "F"],
                ["B", "G", "H"],
                ["A", "C", "I", "G"],
                ["A", "B", "C", "D"],
            ]
            self.species_counts_no_singletons = [2, 2, 3, 3, 4, 4]
            self.data_log_no_singletons = [
                ["A", "B"], ["C", "D"], ["E", "F"]
            ]

        def test_jackknife1_incidence_normal(self):
            """Jackknife-1 with normal incidence-based data"""
            S_obs = len(self.species_counts_1)
            Q1 = sum(1 for x in self.species_counts_1 if x == 1)
            m = len(self.data_log_1)
            result = jackknife1_incidence(S_obs, Q1, m)
            self.assertGreater(result, S_obs,
                               "Jackknife-1 should estimate richness greater than observed for normal data")

        def test_jackknife1_incidence_no_singletons(self):
            """Jackknife-1 with incidence-based data without singletons"""
            S_obs = len(self.species_counts_no_singletons)
            Q1 = sum(1 for x in self.species_counts_no_singletons if x == 1)
            m = len(self.data_log_no_singletons)
            result = jackknife1_incidence(S_obs, Q1, m)
            self.assertEqual(result, S_obs, "Jackknife-1 should equal observed richness when no singletons exist")

    class TestJackknife2Abundance(unittest.TestCase):
        def setUp(self):
            # Custom datasets for testing
            self.data_normal = [10, 15, 20, 25, 30]  # General case
            self.data_with_singletons = [1, 1, 5, 10, 15]  # Includes singletons
            self.data_with_doubletons = [2, 2, 5, 10, 15]  # Includes doubletons
            self.data_no_singletons_or_doubletons = [5, 10, 15, 20, 25]  # No singletons or doubletons
            self.data_empty = []  # Empty dataset

        def test_jackknife2_abundance_normal(self):
            """Jackknife-2 with normal abundance-based data"""
            S_obs = len(self.data_normal)
            Q1 = sum(1 for x in self.data_normal if x == 1)
            Q2 = sum(1 for x in self.data_normal if x == 2)
            result = jackknife2_abundance(S_obs, Q1, Q2)
            self.assertGreaterEqual(result, S_obs,
                                    "Jackknife-2 should estimate richness greater than or equal to observed")

        def test_jackknife2_abundance_with_singletons(self):
            """Jackknife-2 with abundance-based data including singletons"""
            S_obs = len(self.data_with_singletons)
            Q1 = sum(1 for x in self.data_with_singletons if x == 1)
            Q2 = sum(1 for x in self.data_with_singletons if x == 2)
            result = jackknife2_abundance(S_obs, Q1, Q2)
            self.assertGreater(result, S_obs,
                               "Jackknife-2 should estimate richness greater than observed when singletons exist")

        def test_jackknife2_abundance_with_doubletons(self):
            """Jackknife-2 with abundance-based data including doubletons"""
            S_obs = len(self.data_with_doubletons)
            Q1 = sum(1 for x in self.data_with_doubletons if x == 1)
            Q2 = sum(1 for x in self.data_with_doubletons if x == 2)
            result = jackknife2_abundance(S_obs, Q1, Q2)
            self.assertGreater(result, S_obs,
                               "Jackknife-2 should estimate richness greater than observed when doubletons exist")

    class TestJackknife2Incidence(unittest.TestCase):
        def setUp(self):
            # Custom datasets for testing
            self.species_counts_1 = [12, 12, 6, 5, 3, 4, 3, 2, 1]  # General case
            self.data_log_1 = [
                ["A", "B", "C", "D"],
                ["A", "B", "E", "F"],
                ["B", "G", "H"],
                ["A", "C", "I", "G"],
                ["A", "B", "C", "D"],
            ]
            self.species_counts_no_singletons_or_doubletons = [3, 3, 4, 4, 5, 5]
            self.data_log_no_singletons_or_doubletons = [
                ["A", "B"], ["C", "D"], ["E", "F"]
            ]

        def test_jackknife2_incidence_normal(self):
            """Jackknife-2 with normal incidence-based data"""
            S_obs = len(self.species_counts_1)
            Q1 = sum(1 for x in self.species_counts_1 if x == 1)
            Q2 = sum(1 for x in self.species_counts_1 if x == 2)
            m = len(self.data_log_1)
            result = jackknife2_incidence(S_obs, Q1, Q2, m)
            self.assertGreaterEqual(result, S_obs,
                                    "Jackknife-2 should estimate richness greater than or equal to observed for normal data")

        def test_jackknife2_incidence_no_singletons_or_doubletons(self):
            """Jackknife-2 with incidence-based data without singletons or doubletons"""
            S_obs = len(self.species_counts_no_singletons_or_doubletons)
            Q1 = sum(1 for x in self.species_counts_no_singletons_or_doubletons if x == 1)
            Q2 = sum(1 for x in self.species_counts_no_singletons_or_doubletons if x == 2)
            m = len(self.data_log_no_singletons_or_doubletons)
            result = jackknife2_incidence(S_obs, Q1, Q2, m)
            self.assertEqual(result, S_obs,
                             "Jackknife-2 should equal observed richness when no singletons or doubletons exist")


    # Chao1 tests
    class TestChao1Estimator(unittest.TestCase):
        def setUp(self):
            # Custom datasets for testing
            self.data_normal = [1, 1, 2, 2, 3, 4, 4, 4, 10, 12]  # General case
            self.data_no_doubletons = [1, 1, 3, 4, 4, 4, 10, 12]  # No doubletons
            self.data_no_singletons = [2, 2, 3, 4, 4, 4, 10, 12]  # No singletons
            self.data_singletons_only = [1, 1, 1]  # Only singletons
            self.data_empty = []  # Empty dataset

        def test_chao1_normal(self):
            """Chao1 with normal abundance-based data"""
            S_obs = len(self.data_normal)
            abundance_counts = Counter(self.data_normal)
            f1 = abundance_counts[1]
            f2 = abundance_counts[2]
            result = chao1(S_obs, f1, f2)
            expected = S_obs + (f1 ** 2) / (2 * f2)
            self.assertAlmostEqual(result, expected, places=5,
                                   msg="Chao1 result does not match expected value for normal data")

        def test_chao1_no_doubletons(self):
            """Chao1 when no doubletons are present"""
            S_obs = len(self.data_no_doubletons)
            abundance_counts = Counter(self.data_no_doubletons)
            f1 = abundance_counts[1]
            f2 = abundance_counts[2]  # Should be 0
            result = chao1(S_obs, f1, f2)
            expected = S_obs + (f1 ** 2) / 2
            self.assertAlmostEqual(result, expected, places=5,
                                   msg="Chao1 result does not match expected value when no doubletons exist")

        def test_chao1_no_singletons(self):
            """Chao1 when no singletons are present"""
            S_obs = len(self.data_no_singletons)
            abundance_counts = Counter(self.data_no_singletons)
            f1 = abundance_counts[1]  # Should be 0
            f2 = abundance_counts[2]
            result = chao1(S_obs, f1, f2)
            expected = S_obs  # No adjustment needed
            self.assertEqual(result, expected, "Chao1 should equal observed richness when no singletons exist")

        def test_chao1_singletons_only(self):
            """Chao1 when the dataset contains only singletons"""
            S_obs = len(self.data_singletons_only)
            abundance_counts = Counter(self.data_singletons_only)
            f1 = abundance_counts[1]  # Should equal S_obs
            f2 = abundance_counts[2]  # Should be 0
            result = chao1(S_obs, f1, f2)
            expected = S_obs + (f1 ** 2) / 2  # Fallback formula
            self.assertAlmostEqual(result, expected, places=5,
                                   msg="Chao1 result does not match expected value for singletons-only dataset")

        def test_chao1_empty_data(self):
            S_obs = 0
            f1 = 0
            f2 = 0
            result = chao1(S_obs, f1, f2)
            expected = 0
            self.assertEqual(result, expected, "Chao1 should return 0 for empty data")

    # iChao1 tests
    class TestIChao1Estimator(unittest.TestCase):
        def setUp(self):
            # Custom datasets for testing
            self.data_normal = [1, 1, 2, 2, 3, 3, 4, 4, 5, 10]  # General case
            self.data_no_quadrupletons = [1, 1, 2, 2, 3, 3, 3, 5]  # No quadrupletons
            self.data_no_tripletons = [1, 1, 2, 2, 4, 4, 5]  # No tripletons
            self.data_no_doubletons = [1, 1, 3, 3, 3, 4, 4]  # No doubletons
            self.data_singletons_only = [1, 1, 1]  # Only singletons
            self.data_empty = []  # Empty dataset

        def calculate_frequencies(self, data):
            counter = Counter(data)
            f1 = sum(1 for x in counter.values() if x == 1)
            f2 = sum(1 for x in counter.values() if x == 2)
            f3 = sum(1 for x in counter.values() if x == 3)
            f4 = sum(1 for x in counter.values() if x == 4)
            return f1, f2, f3, f4

        def test_ichao1_normal(self):
            """iChao1 with normal abundance-based data"""
            S_obs = len(self.data_normal)
            f1, f2, f3, f4 = self.calculate_frequencies(self.data_normal)
            S_chao1 = chao1(S_obs, f1, f2)
            result = iChao1(S_chao1, f1, f2, f3, f4)

            # Calculate expected result manually
            correction_term = (f3 / (4 * f4)) * max(f1 - (f2 * f3) / (2 * f4), 0) if f4 > 0 else 0
            expected = S_chao1 + correction_term
            self.assertAlmostEqual(result, expected, places=5,
                                   msg="iChao1 result does not match expected value for normal data")

        def test_ichao1_no_quadrupletons(self):
            S_obs = len(self.data_no_quadrupletons)
            f1, f2, f3, f4 = self.calculate_frequencies(self.data_no_quadrupletons)
            S_chao1 = chao1(S_obs, f1, f2)
            result = iChao1(S_chao1, f1, f2, f3, f4)

            # Fallback formula: Use (f3 + 1) in place of f4
            correction_term = (f3 / 4) * max(f1 - (f2 * f3) / (2 * (f3 + 1)), 0)
            expected = S_chao1 + correction_term
            self.assertAlmostEqual(result, expected, places=5,
                                   msg="iChao1 result does not match expected value when no quadrupletons exist")

        def test_ichao1_no_tripletons(self):
            S_obs = len(self.data_no_tripletons)
            f1, f2, f3, f4 = self.calculate_frequencies(self.data_no_tripletons)
            S_chao1 = chao1(S_obs, f1, f2)
            result = iChao1(S_chao1, f1, f2, f3, f4)
            expected = S_chao1  # No correction term as f3 = 0
            self.assertEqual(result, expected, "iChao1 should equal Chao1 when no tripletons exist")

        def test_ichao1_no_doubletons(self):
            S_obs = len(self.data_no_doubletons)
            f1, f2, f3, f4 = self.calculate_frequencies(self.data_no_doubletons)
            S_chao1 = chao1(S_obs, f1, f2)
            result = iChao1(S_chao1, f1, f2, f3, f4)
            expected = S_chao1  # No adjustment in Chao1 as f2 = 0
            self.assertEqual(result, expected, "iChao1 should equal Chao1 when no doubletons exist")

        def test_ichao1_singletons_only(self):
            S_obs = len(self.data_singletons_only)
            f1, f2, f3, f4 = self.calculate_frequencies(self.data_singletons_only)
            S_chao1 = chao1(S_obs, f1, f2)
            result = iChao1(S_chao1, f1, f2, f3, f4)
            expected = S_chao1  # No adjustment as there are no higher-order frequencies
            self.assertEqual(result, expected, "iChao1 should equal Chao1 when only singletons exist")

        def test_ichao1_empty_data(self):
            S_obs = 0
            f1, f2, f3, f4 = 0, 0, 0, 0
            S_chao1 = chao1(S_obs, f1, f2)
            result = iChao1(S_chao1, f1, f2, f3, f4)
            expected = S_chao1
            self.assertEqual(result, expected, "iChao1 should return 0 for empty data")

    # Chao2 tests
    class TestChao2Estimator(unittest.TestCase):
        def setUp(self):
            # Custom datasets for testing
            self.data_log_normal = [
                ["A", "B", "C", "D"],
                ["A", "B", "E", "F"],
                ["B", "G", "H"],
                ["A", "C", "I", "G"],
                ["A", "B", "C", "D"]
            ]  # Normal case
            self.data_log_no_doubletons = [
                ["A"],
                ["B"],
                ["C"],
                ["D"]
            ]  # No doubletons
            self.data_log_no_uniques = [
                ["A", "B"],
                ["A", "B"],
                ["C", "D"],
                ["C", "D"]
            ]  # No unique species
            self.data_log_empty = []  # Empty dataset

        def calculate_species_occurrences(self, data_log):
            S_obs = len(set(species for sample in data_log for species in sample))
            species_occurrences = Counter(species for sample in data_log for species in set(sample))
            Q1 = sum(1 for count in species_occurrences.values() if count == 1)
            Q2 = sum(1 for count in species_occurrences.values() if count == 2)
            return S_obs, Q1, Q2

        def test_chao2_normal(self):
            """Chao2 with normal incidence-based data"""
            S_obs, Q1, Q2 = self.calculate_species_occurrences(self.data_log_normal)
            result = chao2(S_obs, Q1, Q2)
            expected = S_obs + (Q1 ** 2) / (2 * Q2) if Q2 > 0 else S_obs + (Q1 ** 2) / 2
            self.assertAlmostEqual(result, expected, places=5,
                                   msg="Chao2 result does not match expected value for normal data")

        def test_chao2_no_doubletons(self):
            S_obs, Q1, Q2 = self.calculate_species_occurrences(self.data_log_no_doubletons)
            result = chao2(S_obs, Q1, Q2)
            expected = S_obs + (Q1 ** 2) / 2  # Fallback formula
            self.assertAlmostEqual(result, expected, places=5,
                                   msg="Chao2 result does not match expected value when no doubletons exist")

        def test_chao2_no_uniques(self):
            S_obs, Q1, Q2 = self.calculate_species_occurrences(self.data_log_no_uniques)
            result = chao2(S_obs, Q1, Q2)
            expected = S_obs  # No adjustment needed
            self.assertEqual(result, expected, "Chao2 should equal observed richness when no uniques exist")

        def test_chao2_empty_data(self):
            S_obs, Q1, Q2 = 0, 0, 0
            result = chao2(S_obs, Q1, Q2)
            expected = 0
            self.assertEqual(result, expected, "Chao2 should return 0 for empty data")

        def test_chao2_single_sample(self):
            """Chao2 when only one sample is present"""
            data_log_single_sample = [["A", "B", "C"]]  # Single sample
            S_obs, Q1, Q2 = self.calculate_species_occurrences(data_log_single_sample)
            result = chao2(S_obs, Q1, Q2)
            expected = S_obs + (Q1 ** 2) / 2 if Q1 > 0 else S_obs
            self.assertAlmostEqual(result, expected, places=5,
                                   msg="Chao2 result does not match expected value for a single sample")

    # iChao2 tests
class TestIChao2Estimator(unittest.TestCase):
    def setUp(self):
        # Custom datasets for testing
        self.data_log_normal = [
            ["A", "B", "C", "D"],
            ["A", "B", "E", "F"],
            ["B", "G", "H"],
            ["A", "C", "I", "G"],
            ["A", "B", "C", "D"]
        ]  # General case

        self.data_log_no_quadrupletons = [
            ["A", "B"],
            ["A", "C"],
            ["B", "D"],
            ["C", "E"],
            ["D", "F"]
        ]  # No quadrupletons

        self.data_log_no_tripletons = [
            ["A", "B"],
            ["C", "D"],
            ["E", "F"]
        ]  # No tripletons

        self.data_log_no_duplicates = [
            ["A"],
            ["B"],
            ["C"],
            ["D"]
        ]  # No duplicates

        self.data_log_empty = []  # Empty dataset

    def calculate_species_occurrences(self, data_log):
        return calculate_Q_frequencies(data_log)

    def test_ichao2_normal(self):
        """iChao2 with normal incidence-based data"""
        S_obs = len(set().union(*self.data_log_normal))
        Q1, Q2, Q3, Q4 = self.calculate_species_occurrences(self.data_log_normal)
        T = len(self.data_log_normal)
        S_chao2 = chao2(S_obs, Q1, Q2)
        result = iChao2(S_chao2, Q1, Q2, Q3, Q4, T)

        # Calculate expected result manually
        factor1 = (T - 3) / (4 * T)
        factor2 = Q3 / Q4 if Q4 > 0 else 0
        nested_term = ((T - 3) / (2 * (T - 1))) * (Q2 * Q3 / Q4) if Q4 > 0 and T > 1 else 0
        correction_term = factor1 * factor2 * max(Q1 - nested_term, 0) if Q4 > 0 and T > 3 else 0
        expected = S_chao2 + correction_term
        self.assertAlmostEqual(result, expected, places=5, msg="iChao2 result does not match expected value for normal data")

    def test_ichao2_no_quadrupletons(self):
        S_obs = len(set().union(*self.data_log_no_quadrupletons))
        Q1, Q2, Q3, Q4 = self.calculate_species_occurrences(self.data_log_no_quadrupletons)
        T = len(self.data_log_no_quadrupletons)
        S_chao2 = chao2(S_obs, Q1, Q2)
        result = iChao2(S_chao2, Q1, Q2, Q3, Q4, T)
        expected = S_chao2  # No correction as Q4 = 0
        self.assertEqual(result, expected, "iChao2 should equal Chao2 when no quadrupletons exist")

    def test_ichao2_no_tripletons(self):
        S_obs = len(set().union(*self.data_log_no_tripletons))
        Q1, Q2, Q3, Q4 = self.calculate_species_occurrences(self.data_log_no_tripletons)
        T = len(self.data_log_no_tripletons)
        S_chao2 = chao2(S_obs, Q1, Q2)
        result = iChao2(S_chao2, Q1, Q2, Q3, Q4, T)
        expected = S_chao2  # No correction as Q3 = 0
        self.assertEqual(result, expected, "iChao2 should equal Chao2 when no tripletons exist")

    def test_ichao2_no_duplicates(self):
        S_obs = len(set().union(*self.data_log_no_duplicates))
        Q1, Q2, Q3, Q4 = self.calculate_species_occurrences(self.data_log_no_duplicates)
        T = len(self.data_log_no_duplicates)
        S_chao2 = chao2(S_obs, Q1, Q2)
        result = iChao2(S_chao2, Q1, Q2, Q3, Q4, T)
        expected = S_chao2  # No correction as Q2 = 0
        self.assertEqual(result, expected, "iChao2 should equal Chao2 when no duplicates exist")

    def test_ichao2_empty_data(self):
        S_obs, Q1, Q2, Q3, Q4, T = 0, 0, 0, 0, 0, 0
        S_chao2 = chao2(S_obs, Q1, Q2)
        result = iChao2(S_chao2, Q1, Q2, Q3, Q4, T)
        expected = S_chao2  # Should be 0
        self.assertEqual(result, expected, "iChao2 should return 0 for empty data")


if __name__ == '__main__':
    unittest.main()

