"""
Provenance: [FORK NEW]

Purpose:
Unit test suite for src/cross_benchmark.py validating zero-shot transfer logic,
domain aggregation, table outputs, and error handling.
"""

import json
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.cross_benchmark import (
    BENCHMARK_DOMAINS,
    evaluate_partition,
    evaluate_queue_reduction,
    generate_table4_generalization,
    generate_table5_queue_reduction,
    load_calibration_model,
)


class TestCrossBenchmark(unittest.TestCase):
    """Tests the cross-benchmark evaluation module."""

    def setUp(self):
        self.calib_path = "outputs/tables/step4_calibration.json"
        self.eval_table_path = "outputs/tables/eval_table_heldout.csv"

        # Synthetic micro-dataframe with 4 questions mimicking eval_table format
        self.mock_df = pd.DataFrame([
            {
                "dataset": "singleop",
                "majority_ratio": 1.0,
                "entropy_confidence": 1.0,
                "cot_similarity": 0.5,
                "length_stability": 0.8,
                "hedging_score": 0.9,
                "y_maj": "42",
                "modal_count": 24,
                "num_models": 24,
                "parse_error_count": 0,
                "has_missing_cache": False,
                "legacy_target_json": '["42"]',
                "label_cleaning_status": "consensus",
                "label_is_correct": True,
            },
            {
                "dataset": "squad",
                "majority_ratio": 0.6,
                "entropy_confidence": 0.5,
                "cot_similarity": 0.3,
                "length_stability": 0.4,
                "hedging_score": 0.6,
                "y_maj": "Paris",
                "modal_count": 14,
                "num_models": 24,
                "parse_error_count": 0,
                "has_missing_cache": False,
                "legacy_target_json": '["London"]',
                "label_cleaning_status": "revised",
                "label_is_correct": False,
            },
            {
                "dataset": "tab_fact",
                "majority_ratio": 0.9,
                "entropy_confidence": 0.85,
                "cot_similarity": 0.4,
                "length_stability": 0.7,
                "hedging_score": 0.85,
                "y_maj": "1",
                "modal_count": 22,
                "num_models": 24,
                "parse_error_count": 0,
                "has_missing_cache": False,
                "legacy_target_json": '["1"]',
                "label_cleaning_status": "rejected",
                "label_is_correct": True,
            },
            {
                "dataset": "hotpotqa",
                "majority_ratio": 0.5,
                "entropy_confidence": 0.4,
                "cot_similarity": 0.2,
                "length_stability": 0.3,
                "hedging_score": 0.5,
                "y_maj": "PARSING_ERROR",
                "modal_count": 12,
                "num_models": 24,
                "parse_error_count": 1,
                "has_missing_cache": False,
                "legacy_target_json": '["Einstein"]',
                "label_cleaning_status": "verified",
                "label_is_correct": False,
            },
        ])

        if os.path.exists(self.calib_path):
            self.clf, self.calib_data = load_calibration_model(self.calib_path)
        else:
            # Fallback mock model if file missing during isolated unit testing
            self.clf = LogisticRegression()
            self.clf.coef_ = np.array([[2.0, 1.5, 0.5, 0.8, 1.3]])
            self.clf.intercept_ = np.array([-2.5])
            self.clf.classes_ = np.array([0, 1])
            self.calib_data = {
                "logistic_weights": [2.0, 1.5, 0.5, 0.8, 1.3],
                "logistic_intercept": -2.5,
                "taus": {
                    "S_maj": {"0.05": 0.75, "0.01": 1.0},
                    "S_ent": {"0.05": 0.80, "0.01": 1.0},
                    "S_comb_baseline": {"0.05": 0.70, "0.01": 0.85},
                    "S_comb_calib": {"0.05": 0.85, "0.01": 0.95},
                    "S_gate": {"0.05": 0.80, "0.01": 0.95},
                },
            }

    def test_domain_mapping_completeness(self):
        """Verifies that all 14 benchmarks are assigned to a known cognitive domain."""
        expected_benchmarks = [
            "gsm8k", "svamp", "drop", "singleop", "singleq", "multiarith", "mmlu_math",
            "bbh_logical_deduction_three_objects", "bbh_object_counting", "bbh_navigate",
            "tab_fact", "hotpotqa", "squad", "winograd_wsc"
        ]
        valid_domains = {"Math", "Reading / QA", "Symbolic & Logic", "Tabular"}
        for b in expected_benchmarks:
            self.assertIn(b, BENCHMARK_DOMAINS)
            self.assertIn(BENCHMARK_DOMAINS[b], valid_domains)

    def test_evaluate_partition_metrics(self):
        """Checks metric keys and bounds for evaluate_partition."""
        res = evaluate_partition(self.mock_df, self.clf, self.calib_data, "TestPartition")
        self.assertEqual(res["partition"], "TestPartition")
        self.assertEqual(res["N"], 4)
        self.assertGreaterEqual(res["base_error"], 0.0)
        self.assertLessEqual(res["base_error"], 1.0)
        self.assertIn("S_comb_calib_aurc", res)
        self.assertIn("S_comb_calib_cov_005", res)
        self.assertIn("S_comb_calib_risk_005", res)

    def test_evaluate_queue_reduction(self):
        """Checks queue reduction computation on mock data."""
        res = evaluate_queue_reduction(self.mock_df, self.clf, self.calib_data, "TestQueue")
        self.assertEqual(res["N_total"], 4)
        # In mock data, 3 items have cleaning_status != 'consensus' (revised, rejected, verified)
        self.assertEqual(res["N_queue"], 3)
        self.assertEqual(res["queue_pct"], 0.75)
        self.assertIn("queue_reduction", res)
        self.assertIn("queue_risk", res)

    def test_table4_and_table5_generation(self):
        """Verifies that full table generation functions produce valid DataFrames."""
        t4 = generate_table4_generalization(self.mock_df, self.clf, self.calib_data)
        self.assertIsInstance(t4, pd.DataFrame)
        self.assertGreater(len(t4), 0)
        self.assertIn("partition", t4.columns)

        t5 = generate_table5_queue_reduction(self.mock_df, self.clf, self.calib_data)
        self.assertIsInstance(t5, pd.DataFrame)
        self.assertGreater(len(t5), 0)
        self.assertIn("queue_reduction", t5.columns)


if __name__ == "__main__":
    unittest.main()
