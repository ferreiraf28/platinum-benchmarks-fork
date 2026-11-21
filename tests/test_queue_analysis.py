"""
Provenance: [FORK NEW]

Purpose:
Unit test suite for src/queue_analysis.py (Step 5): paper-queue reconstruction,
vote-margin bucketing and reuse of the frozen Step 4 Dev/Eval split.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation import build_error_labels, stratified_split
from src.evaluation import run as evaluation_run
from src.queue_analysis import (
    MARGIN_ORDER,
    TABLE2_METHODS,
    build_queue_frame,
    load_frozen_thresholds,
    margin_bucket,
    paper_flagged,
    per_dataset_queue_thresholds,
    queue_calibrated_thresholds,
    queue_summary,
    score_eval_queue,
    table2,
    wilson_ci,
    _votes,
    table3,
)
from tests.test_evaluation import make_table


class TestPaperQueue(unittest.TestCase):
    """The paper queue is every question a human reviewed (status != consensus)."""

    def test_consensus_is_not_flagged(self):
        df = pd.DataFrame({"label_cleaning_status": ["consensus", "verified", "revised", "rejected", "Consensus"]})
        np.testing.assert_array_equal(paper_flagged(df), [False, True, True, True, False])


class TestMarginBuckets(unittest.TestCase):
    """Buckets count dissenting models out of K = 24."""

    def test_buckets(self):
        self.assertEqual(margin_bucket(24, 24), "24-0")
        self.assertEqual(margin_bucket(23, 24), "23-1")
        self.assertEqual(margin_bucket(22, 24), "22-2")
        self.assertEqual(margin_bucket(21, 24), "21-3")
        self.assertEqual(margin_bucket(20, 24), "<=20")
        self.assertEqual(margin_bucket(7, 24), "<=20")

    def test_every_bucket_is_in_the_reporting_order(self):
        for m in range(1, 25):
            self.assertIn(margin_bucket(m, 24), MARGIN_ORDER)


class TestQueueFrame(unittest.TestCase):
    """The frame reuses the exact Step 4 split and carries the curation error."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "table.csv")
        df = make_table(300)
        df["label_cleaning_status"] = np.where(
            df["label_cleaning_status"] == "rejected", "rejected",
            np.where(df["majority_ratio"] < 1.0, "verified", "consensus"))
        df.to_csv(self.path, index=False)
        self.df = df

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_matches_step4(self):
        frame = build_queue_frame(self.path, ["gsm8k", "svamp", "drop"])
        pop, _ = build_error_labels(self.df.reset_index(drop=True))
        dev_idx, _ = stratified_split(pop, seed=42)
        np.testing.assert_array_equal(np.where(frame["split"] == "dev")[0], dev_idx)

    def test_columns_and_summary(self):
        frame = build_queue_frame(self.path, ["gsm8k", "svamp", "drop"])
        for col in ("split", "error", "paper_flagged", "margin"):
            self.assertIn(col, frame.columns)
        summary = queue_summary(frame)
        self.assertEqual(summary["n"].sum(), int(frame["paper_flagged"].sum()))


class TestWilson(unittest.TestCase):
    """Wilson interval: known values and edge cases."""

    def test_zero_errors(self):
        lo, hi = wilson_ci(0, 10)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)

    def test_contains_point_estimate(self):
        lo, hi = wilson_ci(3, 20)
        self.assertLess(lo, 0.15)
        self.assertGreater(hi, 0.15)

    def test_empty(self):
        self.assertTrue(all(np.isnan(v) for v in wilson_ci(0, 0)))


class TestTable2(unittest.TestCase):
    """Counting logic of Table 2 on a hand-made scored queue."""

    def setUp(self):
        n = 6
        self.scored = pd.DataFrame({
            "margin": ["23-1", "23-1", "23-1", "<=20", "<=20", "<=20"],
            "error": [0, 1, 0, 1, 1, 0],
            **{f"accept_{m}": [True, True, False, True, False, False] for m in TABLE2_METHODS},
            "idx": range(n),
        })

    def test_counts(self):
        t = table2(self.scored, "margin", ["23-1", "<=20"])
        row = t[(t["method"] == "S_comb_calib") & (t["margin"] == "23-1")].iloc[0]
        self.assertEqual((row.queue_n, row.accepted, row.deferred, row.false_accepts), (3, 2, 1, 1))
        self.assertAlmostEqual(row.risk_among_accepted, 0.5)
        self.assertAlmostEqual(row.queue_reduction, 2 / 3)
        total = t[(t["method"] == "S_comb_calib") & (t["margin"] == "Total")].iloc[0]
        self.assertEqual((total.queue_n, total.accepted, total.false_accepts), (6, 3, 2))


class TestFrozenAuditor(unittest.TestCase):
    """Step 5 reuses Step 4's parameters and scores only Eval questions in the paper queue."""

    def test_scores_only_eval_queue_with_stored_thresholds(self):
        df = make_table(300)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "table.csv")
            df.to_csv(src, index=False)
            calib = os.path.join(tmp, "calib.json")
            evaluation_run(src, ["gsm8k", "svamp", "drop"], n_boot=0,
                           table_path=os.path.join(tmp, "t.csv"), figure_path=os.path.join(tmp, "f.png"),
                           split_path=os.path.join(tmp, "s.csv"), calib_path=calib)
            frame = build_queue_frame(src, ["gsm8k", "svamp", "drop"])
            scored = score_eval_queue(frame, calib)
            self.assertTrue((scored["split"] == "eval").all())
            self.assertTrue(scored["paper_flagged"].all())
            taus = load_frozen_thresholds(calib)
            for m in TABLE2_METHODS:
                s = scored[f"score_{m}"]
                np.testing.assert_array_equal(scored[f"accept_{m}"], np.isfinite(s) & (s >= taus[m]))


class TestQueueThresholdsAreDevOnly(unittest.TestCase):
    """Queue-specific thresholds must not change when Eval labels change."""

    def test_eval_labels_do_not_move_queue_thresholds(self):
        df = make_table(400)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "table.csv")
            df.to_csv(src, index=False)
            frame = build_queue_frame(src, ["gsm8k", "svamp", "drop"])
        frame["paper_flagged"] = frame["majority_ratio"] < 1.0
        corrupted = frame.copy()
        is_eval = corrupted["split"] == "eval"
        corrupted.loc[is_eval, "error"] = 1 - corrupted.loc[is_eval, "error"]
        self.assertEqual(queue_calibrated_thresholds(frame), queue_calibrated_thresholds(corrupted))
        self.assertEqual(per_dataset_queue_thresholds(frame, ["gsm8k", "svamp", "drop"]),
                         per_dataset_queue_thresholds(corrupted, ["gsm8k", "svamp", "drop"]))


class TestTable3(unittest.TestCase):
    """Table 3 helpers: vote summary and graceful handling of missing examples."""

    def test_votes_summary(self):
        self.assertEqual(_votes('{"a": "64", "b": "64", "c": "56"}'), "64 ×2, 56 ×1")

    def test_missing_examples_give_header_only(self):
        df = make_table(300)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "table.csv")
            df["parsed_predictions_json"] = '{"m": "42"}'
            df.to_csv(src, index=False)
            calib = os.path.join(tmp, "calib.json")
            evaluation_run(src, ["gsm8k", "svamp", "drop"], n_boot=0,
                           table_path=os.path.join(tmp, "t.csv"), figure_path=os.path.join(tmp, "f.png"),
                           split_path=os.path.join(tmp, "s.csv"), calib_path=calib)
            frame = build_queue_frame(src, ["gsm8k", "svamp", "drop"])
            md = table3(frame[frame["idx"] > 10**6], calib, [], os.path.join(tmp, "none.json"))
        self.assertIn("| Regime |", md)
        self.assertEqual(md.count("\n| 4."), 0)


if __name__ == "__main__":
    unittest.main()