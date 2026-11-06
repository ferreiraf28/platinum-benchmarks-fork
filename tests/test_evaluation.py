"""
Provenance: [FORK NEW]

Purpose:
Unit test suite for src/evaluation.py validating the Step 4 protocol: blind stratified
split, error definitions, tie-aware risk-coverage curves, AURC, threshold calibration,
safety-gate normalization, mandatory deferrals and hermetic Dev/Eval separation.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation import (
    BLIND_COLUMNS,
    ERROR_DEF_CURATION,
    ERROR_DEF_LABEL,
    aurc,
    blind_view,
    build_error_labels,
    calibrate_on_dev,
    calibrate_tau,
    compute_scores,
    gate_triggered,
    load_eval_table,
    margin_bin,
    normalize_answer,
    operating_point,
    rc_curve,
    risk_at_coverage,
    run,
    stratified_split,
)


def make_table(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic per-question table where low consensus correlates with errors."""
    rng = np.random.default_rng(seed)
    datasets = np.array(["gsm8k", "svamp", "drop"])[rng.integers(0, 3, n)]
    m = np.clip(rng.choice([1.0, 0.9, 0.6], n, p=[0.5, 0.3, 0.2]) + rng.normal(0, 0.01, n), 0.3, 1.0)
    m[m > 0.995] = 1.0
    bad = rng.random(n) < (1.05 - m) * 0.6
    status = np.where(bad & (rng.random(n) < 0.7), "rejected", "consensus")
    is_correct = np.where(status == "rejected", None, ~bad)
    return pd.DataFrame({
        "dataset": datasets, "idx": np.arange(n),
        "majority_ratio": m, "entropy_confidence": m ** 2,
        "cot_similarity": rng.random(n), "length_stability": rng.random(n), "hedging_score": rng.random(n),
        "y_maj": ["42"] * n, "modal_count": np.round(m * 24).astype(int), "num_models": 24,
        "parse_error_count": 0, "has_missing_cache": False,
        "legacy_target_json": json.dumps(["42"]),
        "label_cleaning_status": status,
        "label_platinum_target_json": json.dumps(["42"]),
        "label_is_correct": is_correct,
    })


class TestSplit(unittest.TestCase):
    """Dev/Eval split must be balanced, disjoint, reproducible and label-free."""

    def setUp(self):
        self.df = make_table(300)

    def test_split_is_disjoint_and_exhaustive(self):
        dev, ev = stratified_split(self.df, seed=42)
        self.assertEqual(len(set(dev) & set(ev)), 0)
        self.assertEqual(len(dev) + len(ev), len(self.df))
        self.assertLessEqual(abs(len(dev) - len(ev)), 1)

    def test_split_is_reproducible(self):
        a, _ = stratified_split(self.df, seed=42)
        b, _ = stratified_split(self.df, seed=42)
        c, _ = stratified_split(self.df, seed=7)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_split_is_stratified_by_dataset_and_margin(self):
        dev, ev = stratified_split(self.df, seed=42)
        strata = self.df["dataset"] + "|" + self.df["majority_ratio"].map(margin_bin)
        for s in strata.unique():
            n_dev, n_ev = (strata.iloc[dev] == s).sum(), (strata.iloc[ev] == s).sum()
            self.assertLessEqual(abs(n_dev - n_ev), 1)

    def test_split_ignores_labels(self):
        """Flipping every label must not change the partition (split uses blind columns only)."""
        flipped = self.df.copy()
        flipped["label_cleaning_status"] = "rejected"
        np.testing.assert_array_equal(stratified_split(self.df)[0], stratified_split(flipped)[0])

    def test_margin_bins(self):
        self.assertEqual(margin_bin(1.0), "unanimous")
        self.assertEqual(margin_bin(0.875), "high")
        self.assertEqual(margin_bin(0.5), "low")


class TestErrorDefinitions(unittest.TestCase):
    """Curation definition counts rejected questions as errors; label definition excludes them."""

    def setUp(self):
        self.df = pd.DataFrame({
            "label_cleaning_status": ["consensus", "rejected", "revised", "consensus"],
            "label_is_correct": [True, None, False, True],
        })

    def test_curation_definition(self):
        pop, e = build_error_labels(self.df, ERROR_DEF_CURATION)
        self.assertEqual(len(pop), 4)
        np.testing.assert_array_equal(e, [0, 1, 1, 0])

    def test_label_definition(self):
        pop, e = build_error_labels(self.df, ERROR_DEF_LABEL)
        self.assertEqual(len(pop), 3)
        np.testing.assert_array_equal(e, [0, 1, 0])

    def test_unknown_definition_raises(self):
        with self.assertRaises(ValueError):
            build_error_labels(self.df, "other")


class TestRiskCoverage(unittest.TestCase):
    """Tie-aware RC curve, AURC and risk at a fixed coverage."""

    def test_perfect_ranking(self):
        cov, risk = rc_curve(np.array([0.9, 0.8, 0.7, 0.1]), np.array([0, 0, 0, 1]))
        np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(risk, [0, 0, 0, 0.25])

    def test_all_tied_scores_give_flat_base_rate(self):
        """One tie block = random order: expected risk equals the base rate at every coverage."""
        errors = np.array([1, 0, 0, 0])
        _, risk = rc_curve(np.zeros(4), errors)
        np.testing.assert_allclose(risk, 0.25)

    def test_partial_tie_block(self):
        """Top item is safe; the next two are tied with one error -> expected 0.5 error at k=2."""
        _, risk = rc_curve(np.array([1.0, 0.5, 0.5]), np.array([0, 1, 0]))
        np.testing.assert_allclose(risk, [0.0, 0.25, 1 / 3])

    def test_aurc_trapezoid(self):
        cov = np.array([0.5, 1.0])
        risk = np.array([0.0, 0.2])
        self.assertAlmostEqual(aurc(cov, risk), 0.05)

    def test_better_ranking_has_lower_aurc(self):
        errors = np.array([0, 0, 1, 1])
        good = aurc(*rc_curve(np.array([4, 3, 2, 1.0]), errors))
        bad = aurc(*rc_curve(np.array([1, 2, 3, 4.0]), errors))
        self.assertLess(good, bad)

    def test_risk_at_coverage(self):
        cov, risk = rc_curve(np.array([4, 3, 2, 1.0]), np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(risk_at_coverage(cov, risk, 0.75), 1 / 3)


class TestThresholds(unittest.TestCase):
    """Operating points and tau calibration."""

    def test_calibrate_tau_picks_largest_safe_coverage(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        errors = np.array([0, 0, 0, 1, 1])
        self.assertEqual(calibrate_tau(scores, errors, alpha=0.0), 0.7)
        self.assertEqual(calibrate_tau(scores, errors, alpha=0.25), 0.6)

    def test_calibrate_tau_returns_inf_when_impossible(self):
        self.assertEqual(calibrate_tau(np.array([0.9, 0.8]), np.array([1, 1]), alpha=0.05), float("inf"))

    def test_mandatory_deferrals_are_never_accepted(self):
        scores = np.array([0.9, -np.inf, 0.5])
        cov, risk, n = operating_point(scores, np.array([0, 0, 1]), tau=-1e9)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(cov, 2 / 3)


class TestGateAndScores(unittest.TestCase):
    """Safety-gate normalization, forced deferrals and hermetic scoring."""

    def test_normalize_answer_formats(self):
        self.assertEqual(normalize_answer("25 yards"), normalize_answer("25"))
        self.assertEqual(normalize_answer("two"), normalize_answer("2"))
        self.assertEqual(normalize_answer("697%"), normalize_answer("697"))
        self.assertEqual(normalize_answer("36.0"), normalize_answer("36"))
        self.assertEqual(normalize_answer("The Lascarins"), "lascarins")

    def test_gate_triggers_only_on_real_disagreement(self):
        self.assertFalse(gate_triggered("25 yards", json.dumps(["25"])))
        self.assertTrue(gate_triggered("11 days", json.dumps(["12"])))
        self.assertFalse(gate_triggered("anything", json.dumps([])))

    def test_parse_error_forces_deferral_in_every_score(self):
        df = make_table(10)
        df.loc[0, "parse_error_count"] = 1
        scores = compute_scores(blind_view(df), calib_override=np.full(10, 0.9))
        for name, s in scores.items():
            self.assertEqual(s[0], -np.inf, name)

    def test_scoring_rejects_label_columns(self):
        with self.assertRaises(ValueError):
            compute_scores(make_table(10))

    def test_blind_view_has_no_labels(self):
        cols = blind_view(make_table(10)).columns
        self.assertFalse(any(c.startswith("label_") for c in cols))
        self.assertTrue(set(BLIND_COLUMNS).issubset(cols))


class TestHermeticSeparation(unittest.TestCase):
    """Definition of Done: no Eval sample influences weights or thresholds."""

    def test_eval_labels_do_not_change_calibration(self):
        df = make_table(400)
        pop, errors = build_error_labels(df)
        dev_idx, eval_idx = stratified_split(pop)
        dev = pop.iloc[dev_idx].reset_index(drop=True)
        _, calib_a, _ = calibrate_on_dev(dev, errors[dev_idx])

        # Corrupt every Eval label: calibration only sees Dev, so nothing may change.
        corrupted = errors.copy()
        corrupted[eval_idx] = 1 - corrupted[eval_idx]
        _, calib_b, _ = calibrate_on_dev(dev, corrupted[dev_idx])
        self.assertEqual(calib_a.taus, calib_b.taus)
        self.assertEqual(calib_a.logistic_weights, calib_b.logistic_weights)


class TestEndToEnd(unittest.TestCase):
    """Full pipeline writes Table 1, the figure and the split file; headerless CSVs are recovered."""

    def test_run_produces_outputs(self):
        df = make_table(300)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "table.csv")
            df.to_csv(src, index=False)
            paths = {k: os.path.join(tmp, f"{k}") for k in ("t.csv", "f.png", "s.csv", "c.json")}
            table = run(src, ["gsm8k", "svamp", "drop"], n_boot=20, table_path=paths["t.csv"],
                        figure_path=paths["f.png"], split_path=paths["s.csv"], calib_path=paths["c.json"])
            for p in paths.values():
                self.assertTrue(os.path.exists(p), p)
            self.assertIn("S_comb_calib", set(table["method"]))
            split = pd.read_csv(paths["s.csv"])
            self.assertEqual(set(split["split"]), {"dev", "eval"})

    def test_headerless_csv_is_recovered(self):
        from src.evaluation import EXPORT_COLUMNS
        row = {c: 0 for c in EXPORT_COLUMNS}
        row.update({"dataset": "gsm8k", "y_maj": "7", "has_missing_cache": "False", "label_is_correct": "True"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nohdr.csv")
            pd.DataFrame([row])[EXPORT_COLUMNS].to_csv(path, index=False, header=False)
            loaded = load_eval_table(path)
            self.assertEqual(loaded.loc[0, "dataset"], "gsm8k")
            self.assertTrue(loaded.loc[0, "label_is_correct"])


if __name__ == "__main__":
    unittest.main()