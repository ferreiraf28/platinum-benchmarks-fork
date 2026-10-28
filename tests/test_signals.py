"""
Provenance: [FORK NEW]

Purpose:
Unit test suite for src/signals.py validating the 5 observable uncertainty signals
against mathematical boundary conditions, edge cases, and strict blind execution.
"""

import math
import unittest
import numpy as np

from src.signals import (
    compute_majority_ratio,
    compute_entropy_confidence,
    extract_reasoning_body,
    compute_cot_similarity,
    compute_length_stability,
    compute_hedging_score,
    extract_signals_for_ensemble,
)


class TestUncertaintySignals(unittest.TestCase):
    """Verifies mathematical correctness and boundary behavior of uncertainty signals."""

    def test_majority_ratio_unanimous(self):
        """24 identical predictions must yield M(x) = 1.0."""
        preds = ["18"] * 24
        maj_ratio, modal_pred, modal_count = compute_majority_ratio(preds)
        self.assertAlmostEqual(maj_ratio, 1.0)
        self.assertEqual(modal_pred, "18")
        self.assertEqual(modal_count, 24)

    def test_majority_ratio_50_50_tie(self):
        """12 votes for A and 12 votes for B must yield M(x) = 0.50."""
        preds = ["A"] * 12 + ["B"] * 12
        maj_ratio, modal_pred, modal_count = compute_majority_ratio(preds)
        self.assertAlmostEqual(maj_ratio, 0.50)
        self.assertEqual(modal_count, 12)

    def test_majority_ratio_23_1_split(self):
        """23 votes for A and 1 vote for B must yield M(x) = 23/24."""
        preds = ["A"] * 23 + ["B"] * 1
        maj_ratio, modal_pred, modal_count = compute_majority_ratio(preds)
        self.assertAlmostEqual(maj_ratio, 23.0 / 24.0)
        self.assertEqual(modal_pred, "A")
        self.assertEqual(modal_count, 23)

    def test_entropy_unanimous(self):
        """Unanimous distribution has H(x) = 0 bits and V(x) = 1.0."""
        preds = ["42"] * 24
        conf, raw_h = compute_entropy_confidence(preds)
        self.assertAlmostEqual(raw_h, 0.0)
        self.assertAlmostEqual(conf, 1.0)

    def test_entropy_maximum_dispersion(self):
        """24 unique predictions have H(x) = log2(24) and V(x) = 0.0."""
        preds = [str(i) for i in range(24)]
        conf, raw_h = compute_entropy_confidence(preds)
        expected_max_h = math.log2(24)
        self.assertAlmostEqual(raw_h, expected_max_h, places=5)
        self.assertAlmostEqual(conf, 0.0, places=5)

    def test_entropy_50_50_tie(self):
        """12-12 tie has H(x) = 1.0 bit, V(x) = 1 - 1/log2(24)."""
        preds = ["A"] * 12 + ["B"] * 12
        conf, raw_h = compute_entropy_confidence(preds)
        self.assertAlmostEqual(raw_h, 1.0, places=5)
        expected_conf = 1.0 - (1.0 / math.log2(24))
        self.assertAlmostEqual(conf, expected_conf, places=5)

    def test_entropy_23_1_split(self):
        """23-1 split has small non-zero entropy and V(x) ~ 0.945."""
        preds = ["A"] * 23 + ["B"] * 1
        conf, raw_h = compute_entropy_confidence(preds)
        p1 = 23 / 24
        p2 = 1 / 24
        expected_h = -(p1 * math.log2(p1) + p2 * math.log2(p2))
        self.assertAlmostEqual(raw_h, expected_h, places=5)
        expected_conf = 1.0 - (expected_h / math.log2(24))
        self.assertAlmostEqual(conf, expected_conf, places=5)
        self.assertTrue(0.94 < conf < 0.96)

    def test_extract_reasoning_body_cot_tag(self):
        """Extracts text inside <cot>...</cot> tags."""
        raw = "<cot>Step 1: 16 - 7 = 9.\nStep 2: 9 * 2 = 18.</cot>\nAnswer: 18"
        body = extract_reasoning_body(raw)
        self.assertEqual(body, "Step 1: 16 - 7 = 9.\nStep 2: 9 * 2 = 18.")

    def test_extract_reasoning_body_thinking_tag(self):
        """Extracts text inside <thinking>...</thinking> tags."""
        raw = "<thinking>Calculate remainder: 9. Revenue: $18.</thinking>\n\nAnswer: 18"
        body = extract_reasoning_body(raw)
        self.assertEqual(body, "Calculate remainder: 9. Revenue: $18.")

    def test_extract_reasoning_body_standard_answer(self):
        """Extracts text preceding the final 'Answer:' keyword."""
        raw = "Janet sells 9 eggs for $2 each.\nTotal revenue is 9 * 2 = 18 dollars.\nAnswer: 18"
        body = extract_reasoning_body(raw)
        self.assertIn("Janet sells 9 eggs", body)
        self.assertNotIn("Answer: 18", body)

    def test_cot_similarity_identical(self):
        """Identical reasoning texts must yield Sim(x) = 1.0."""
        text = "Janet starts with 16 eggs. She eats 3 and uses 4, leaving 9 eggs. Sells at $2 each: 9 * 2 = $18."
        texts = [text] * 10
        sim = compute_cot_similarity(texts)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cot_similarity_divergent(self):
        """Orthogonal reasoning texts must yield low Sim(x)."""
        texts = [
            "Janet calculates egg sales with ducks and muffins at the market.",
            "Quantum electrodynamics describes photon interactions in relativistic physics.",
            "The culinary recipe requires chopped onions simmered in olive oil.",
        ]
        sim = compute_cot_similarity(texts)
        self.assertTrue(sim < 0.25)

    def test_cot_similarity_filters_short_outputs(self):
        """Filters out non-reasoning 10-char outputs like 'Answer: 18'."""
        rich_text = "Step 1: Compute remaining duck eggs 16 - 7 = 9. Step 2: Multiply 9 by 2 dollars to get 18."
        texts = [rich_text] * 18 + ["Answer: 18"] * 6  # 6 closed models with redacted CoT
        sim = compute_cot_similarity(texts)
        # Should evaluate across the 18 rich texts and yield near 1.0
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_length_stability(self):
        """Identical lengths yield LS(x) = 1.0; extreme dispersion yields LS(x) near 0.0."""
        equal_texts = ["A" * 100] * 20
        scaled_ls, raw_ls = compute_length_stability(equal_texts, return_raw=True)
        self.assertAlmostEqual(scaled_ls, 1.0, places=4)
        self.assertAlmostEqual(raw_ls, 1.0, places=4)

        # High dispersion: half short, half long
        dispersed_texts = ["A" * 10] * 10 + ["B" * 2000] * 10
        scaled_ls, raw_ls = compute_length_stability(dispersed_texts, return_raw=True)
        self.assertTrue(0.0 <= scaled_ls <= 1.0)
        self.assertTrue(raw_ls < 0.5)

        # Extreme outlier dispersion: 23 short (10 chars), 1 huge outlier (15,000 chars)
        extreme_texts = ["A" * 10] * 23 + ["B" * 15000] * 1
        scaled_extreme = compute_length_stability(extreme_texts)
        self.assertTrue(scaled_extreme < 0.1)

    def test_hedging_score(self):
        """Zero hedging terms yield Hed(x) = 1.0; heavy hedging lowers score."""
        confident_texts = ["The exact solution is calculated as follows. The final value is 18."] * 10
        self.assertAlmostEqual(compute_hedging_score(confident_texts), 1.0, places=4)

        hedged_texts = [
            "Perhaps the answer is 18, assuming that the premise is not ambiguous. It is possible that..."
        ] * 10
        hed = compute_hedging_score(hedged_texts)
        self.assertTrue(hed < 0.6)

    def test_extract_signals_for_ensemble_blind_mode(self):
        """
        Extracts all 5 signals in strict blind mode without any target or curation metadata.
        """
        example = {
            "question": "Janet has 16 eggs. Eats 3, uses 4. Sells remaining for $2. Total?",
            "platinum_parsing_strategy": "math",
        }
        # Synthetic ensemble of 24 models: 22 correct (18), 1 typo (70), 1 arithmetic error (25)
        ensemble = {}
        for i in range(22):
            ensemble[f"model_{i}"] = (
                f"Step 1: 16 - 7 = 9 eggs remaining.\nStep 2: 9 * 2 = 18 dollars.\nAnswer: 18"
            )
        ensemble["model_typo"] = "Janet flips 16 eggs. Answer: 70"
        ensemble["model_err"] = "Janet makes a mistake. 9 * 2 = 25.\nAnswer: 25"

        signals = extract_signals_for_ensemble("gsm8k", example, ensemble)

        # Verify all signals exist and are bounded [0.0, 1.0]
        self.assertIn("majority_ratio", signals)
        self.assertIn("entropy_confidence", signals)
        self.assertIn("cot_similarity", signals)
        self.assertIn("length_stability", signals)
        self.assertIn("hedging_score", signals)
        self.assertIn("signal_vector", signals)

        vec = signals["signal_vector"]
        self.assertEqual(vec.shape, (5,))
        self.assertTrue(np.all(vec >= 0.0) and np.all(vec <= 1.0))

        # Check values match expectation
        self.assertEqual(signals["y_maj"], "18")
        self.assertEqual(signals["modal_count"], 22)
        self.assertAlmostEqual(signals["majority_ratio"], 22.0 / 24.0)
        self.assertTrue(signals["entropy_confidence"] > 0.85)


if __name__ == "__main__":
    unittest.main()
