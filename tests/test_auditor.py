"""
Provenance: [FORK NEW]

Purpose:
Unit test suite for src/auditor.py validating the Selective Auditor Engine, baseline scores,
operational decision rules, mandatory deferral guards, and the Safety Gate variant.
"""

import math
import unittest
import numpy as np

from src.auditor import (
    ACCEPT_PROVISIONAL,
    DEFER_TO_HUMAN,
    MODE_CALIBRATED_COMPOSITE,
    MODE_ENTROPY,
    MODE_LINEAR_COMPOSITE,
    MODE_MAJORITY,
    AuditorDecision,
    SafetyGateAuditor,
    SelectiveAuditor,
)


class TestSelectiveAuditor(unittest.TestCase):
    """Verifies scoring accuracy, decision thresholds, and safety policies of the auditor."""

    def setUp(self):
        self.auditor = SelectiveAuditor()

        # Synthetic standard high-consensus instance
        self.high_conf_signals = {
            "majority_ratio": 1.0,
            "entropy_confidence": 1.0,
            "cot_similarity": 0.80,
            "length_stability": 0.75,
            "hedging_score": 0.95,
            "signal_vector": np.array([1.0, 1.0, 0.80, 0.75, 0.95]),
            "y_maj": "42",
            "modal_count": 24,
            "num_models": 24,
            "parse_error_count": 0,
            "has_missing_cache": False,
            "parsed_predictions": {f"model_{i}": "42" for i in range(24)},
        }

        # Synthetic low-consensus contested instance (12-12 split)
        self.low_conf_signals = {
            "majority_ratio": 0.50,
            "entropy_confidence": 0.78,
            "cot_similarity": 0.20,
            "length_stability": 0.30,
            "hedging_score": 0.60,
            "signal_vector": np.array([0.50, 0.78, 0.20, 0.30, 0.60]),
            "y_maj": "A",
            "modal_count": 12,
            "num_models": 24,
            "parse_error_count": 0,
            "has_missing_cache": False,
            "parsed_predictions": {**{f"m_{i}": "A" for i in range(12)}, **{f"m_{i}": "B" for i in range(12, 24)}},
        }

    def test_score_majority_baseline(self):
        """Verifies S_maj(x) = M(x)."""
        score_high = self.auditor.score(self.high_conf_signals, mode=MODE_MAJORITY)
        self.assertAlmostEqual(score_high, 1.0)

        score_low = self.auditor.score(self.low_conf_signals, mode=MODE_MAJORITY)
        self.assertAlmostEqual(score_low, 0.50)

    def test_score_entropy_baseline(self):
        """Verifies S_ent(x) = V(x)."""
        score_high = self.auditor.score(self.high_conf_signals, mode=MODE_ENTROPY)
        self.assertAlmostEqual(score_high, 1.0)

        score_low = self.auditor.score(self.low_conf_signals, mode=MODE_ENTROPY)
        self.assertAlmostEqual(score_low, 0.78)

    def test_score_linear_composite(self):
        """
        Verifies S_comb_baseline = 0.40*M + 0.35*V + 0.15*Sim + 0.10*LS.
        """
        score = self.auditor.score(self.high_conf_signals, mode=MODE_LINEAR_COMPOSITE)
        # 0.40(1.0) + 0.35(1.0) + 0.15(0.80) + 0.10(0.75) = 0.40 + 0.35 + 0.12 + 0.075 = 0.945
        expected = 0.40 * 1.0 + 0.35 * 1.0 + 0.15 * 0.80 + 0.10 * 0.75
        self.assertAlmostEqual(score, expected, places=5)

    def test_score_calibrated_composite_manual_weights(self):
        """Verifies S_comb_calib = sigmoid(w^T s + b)."""
        w = np.array([2.0, 1.5, 0.5, 0.5, 0.5])
        b = -1.0
        auditor = SelectiveAuditor(calibrated_weights=w, calibrated_intercept=b)

        score = auditor.score(self.high_conf_signals, mode=MODE_CALIBRATED_COMPOSITE)
        z = np.dot(w, self.high_conf_signals["signal_vector"]) + b
        expected = 1.0 / (1.0 + math.exp(-z))
        self.assertAlmostEqual(score, expected, places=5)
        self.assertTrue(0.0 <= score <= 1.0)

    def test_fit_calibrated_logistic(self):
        """Verifies fitting logistic regression on development signal vectors."""
        # Create small synthetic dev set: 20 positive (high confidence), 20 negative (low confidence)
        dev_signals = [self.high_conf_signals] * 20 + [self.low_conf_signals] * 20
        dev_labels = [True] * 20 + [False] * 20

        auditor = SelectiveAuditor()
        auditor.fit_calibrated(dev_signals, dev_labels)

        score_pos = auditor.score(self.high_conf_signals, mode=MODE_CALIBRATED_COMPOSITE)
        score_neg = auditor.score(self.low_conf_signals, mode=MODE_CALIBRATED_COMPOSITE)

        self.assertTrue(score_pos > 0.75)
        self.assertTrue(score_neg < 0.35)
        self.assertTrue(score_pos - score_neg > 0.40)

    def test_decision_acceptance_above_threshold(self):
        """When S(x) >= tau and cache is complete, returns ACCEPT_PROVISIONAL."""
        decision = self.auditor.decide(self.high_conf_signals, tau=0.85, mode=MODE_LINEAR_COMPOSITE)
        self.assertEqual(decision.decision, ACCEPT_PROVISIONAL)
        self.assertTrue(decision.is_accepted)
        self.assertTrue(decision.score >= 0.85)
        self.assertEqual(decision.modal_prediction, "42")

    def test_decision_deferral_below_threshold(self):
        """When S(x) < tau, returns DEFER_TO_HUMAN."""
        decision = self.auditor.decide(self.low_conf_signals, tau=0.85, mode=MODE_LINEAR_COMPOSITE)
        self.assertEqual(decision.decision, DEFER_TO_HUMAN)
        self.assertFalse(decision.is_accepted)
        self.assertIn("DEFERRED_SCORE", decision.reason)

    def test_threshold_monotonicity(self):
        """If accepted at threshold tau_2, must be accepted at any looser threshold tau_1 < tau_2."""
        d_strict = self.auditor.decide(self.high_conf_signals, tau=0.90)
        d_loose = self.auditor.decide(self.high_conf_signals, tau=0.70)

        if d_strict.is_accepted:
            self.assertTrue(d_loose.is_accepted)

    def test_mandatory_deferral_incomplete_cache(self):
        """
        Incomplete cache (missing models) MUST trigger mandatory DEFER_TO_HUMAN,
        even if confidence score is 1.0!
        """
        missing_cache_signals = self.high_conf_signals.copy()
        missing_cache_signals["has_missing_cache"] = True
        missing_cache_signals["num_models"] = 21

        decision = self.auditor.decide(missing_cache_signals, tau=0.50, required_models=24)
        self.assertEqual(decision.decision, DEFER_TO_HUMAN)
        self.assertFalse(decision.is_cache_complete)
        self.assertIn("DEFERRED_MANDATORY", decision.reason)
        self.assertIn("Cache quorum incomplete", decision.reason)

    def test_mandatory_deferral_parse_error(self):
        """
        Model parsing errors MUST trigger mandatory DEFER_TO_HUMAN,
        even if confidence score is high.
        """
        parse_err_signals = self.high_conf_signals.copy()
        parse_err_signals["parse_error_count"] = 2

        decision = self.auditor.decide(parse_err_signals, tau=0.50)
        self.assertEqual(decision.decision, DEFER_TO_HUMAN)
        self.assertTrue(decision.has_parse_error)
        self.assertIn("parsing errors", decision.reason)

    def test_unknown_scoring_mode_raises(self):
        """Invalid scoring mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            self.auditor.score(self.high_conf_signals, mode="invalid_mode_name")


class TestSafetyGateAuditor(unittest.TestCase):
    """Verifies legacy label safety gate mechanics."""

    def setUp(self):
        self.gate_auditor = SafetyGateAuditor()

        # 24 models: 23 answered "42", 1 answered "70"
        self.split_predictions = {f"model_{i}": "42" for i in range(23)}
        self.split_predictions["model_23"] = "70"

        self.divergent_signals = {
            "majority_ratio": 23.0 / 24.0,
            "entropy_confidence": 0.95,
            "cot_similarity": 0.80,
            "length_stability": 0.70,
            "hedging_score": 0.95,
            "signal_vector": np.array([23.0 / 24.0, 0.95, 0.80, 0.70, 0.95]),
            "y_maj": "42",
            "modal_count": 23,
            "num_models": 24,
            "parse_error_count": 0,
            "has_missing_cache": False,
            "parsed_predictions": self.split_predictions,
        }

        # 24 models all answered "42"
        self.unanimous_predictions = {f"model_{i}": "42" for i in range(24)}
        self.unanimous_signals = {
            "majority_ratio": 1.0,
            "entropy_confidence": 1.0,
            "cot_similarity": 0.85,
            "length_stability": 0.80,
            "hedging_score": 1.0,
            "signal_vector": np.array([1.0, 1.0, 0.85, 0.80, 1.0]),
            "y_maj": "42",
            "modal_count": 24,
            "num_models": 24,
            "parse_error_count": 0,
            "has_missing_cache": False,
            "parsed_predictions": self.unanimous_predictions,
        }

    def test_safety_gate_default_majority_mode_accepts_minority_slip(self):
        """
        When 23 models say '42' and 1 model says '70', and legacy target is '42':
        Under default majority gate, modal prediction matches legacy label.
        Safety gate passes, allowing confidence thresholding to accept!
        """
        decision = self.gate_auditor.decide(
            signals=self.divergent_signals,
            legacy_target="42",
            tau=0.85,
            gate_mode="majority",
        )
        self.assertEqual(decision.decision, ACCEPT_PROVISIONAL)
        self.assertFalse(decision.legacy_disagreement)
        self.assertTrue(decision.is_accepted)

    def test_safety_gate_default_majority_mode_defers_when_majority_disagrees(self):
        """
        When modal prediction disagrees with legacy label (e.g. models overturned label),
        majority safety gate must intercept and trigger mandatory DEFER_TO_HUMAN.
        """
        overturned_signals = self.divergent_signals.copy()
        overturned_signals["y_maj"] = "70"  # Modal prediction is now 70, but legacy is 42

        decision = self.gate_auditor.decide(
            signals=overturned_signals,
            legacy_target="42",
            tau=0.50,  # Even with loose threshold
            gate_mode="majority",
        )
        self.assertEqual(decision.decision, DEFER_TO_HUMAN)
        self.assertTrue(decision.legacy_disagreement)
        self.assertIn("Safety gate triggered", decision.reason)
        self.assertIn("modal consensus '70' disagrees", decision.reason)

    def test_safety_gate_strict_unanimous_mode_defers_on_single_outlier(self):
        """
        Under paper baseline 'strict_unanimous', even a single dissenting model
        forces mandatory DEFER_TO_HUMAN.
        """
        decision = self.gate_auditor.decide(
            signals=self.divergent_signals,
            legacy_target="42",
            tau=0.50,
            gate_mode="strict_unanimous",
        )
        self.assertEqual(decision.decision, DEFER_TO_HUMAN)
        self.assertTrue(decision.legacy_disagreement)
        self.assertIn("Safety gate triggered (1 models disagreed", decision.reason)

    def test_safety_gate_passes_on_unanimous_legacy_agreement(self):
        """
        When all 24 models agree with legacy target '42',
        safety gate passes and accepts if score >= tau.
        """
        decision = self.gate_auditor.decide(
            signals=self.unanimous_signals,
            legacy_target="42",
            tau=0.85,
        )
        self.assertEqual(decision.decision, ACCEPT_PROVISIONAL)
        self.assertFalse(decision.legacy_disagreement)
        self.assertTrue(decision.is_accepted)

    def test_safety_gate_handles_list_legacy_targets(self):
        """
        Handles list of acceptable legacy labels (e.g. ['42', '42.0', '$42']).
        """
        decision = self.gate_auditor.decide(
            signals=self.unanimous_signals,
            legacy_target=["42", "42.0"],
            tau=0.85,
        )
        self.assertEqual(decision.decision, ACCEPT_PROVISIONAL)
        self.assertFalse(decision.legacy_disagreement)

    def test_safety_gate_bypassed_when_no_legacy_target(self):
        """
        When legacy_target is None, delegates purely to blind selective auditor.
        """
        decision = self.gate_auditor.decide(
            signals=self.unanimous_signals,
            legacy_target=None,
            tau=0.85,
        )
        self.assertEqual(decision.decision, ACCEPT_PROVISIONAL)
        self.assertIsNone(decision.legacy_disagreement)


if __name__ == "__main__":
    unittest.main()
