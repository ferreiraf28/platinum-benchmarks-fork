"""
Provenance: [FORK NEW]

Purpose:
Builds the per-question evaluation table consumed by Step 4 (src/evaluation.py).
For every test question of the selected benchmarks it (1) computes the blind
uncertainty signals from the offline cache, exactly as Steps 2-3 do, and
(2) appends, in clearly separated columns, the legacy label and the platinum
ground-truth fields needed only by the evaluation layer. Running this once
locally decouples the heavy cache/Hugging Face access from the evaluation code.

Column groups (hermetic separation is encoded in the column names):
- Identity:           dataset, idx
- Operational/blind:  signals, y_maj, vote counts, parse/cache flags, per-model predictions
- Legacy label:       legacy_*  (inherited original_target; allowed ONLY for the S_gate variant)
- Ground truth:       label_*   (cleaning_status, platinum_target, is_correct; evaluation layer ONLY)

Usage:
    python3 scripts/export_eval_table.py --datasets gsm8k svamp drop
    python3 scripts/export_eval_table.py --datasets gsm8k --limit 10   # quick smoke run
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from src.offline_loader import OFFICIAL_PAPER_MODELS, OfflineDataLoader
from src.signals import extract_signals_for_ensemble
from src.utils import check_prediction


CORE_BENCHMARKS: List[str] = ["gsm8k", "svamp", "drop"]
HF_DATASET_REPO = "madrylab/platinum-bench-paper-version"

# Sentinel predictions produced by src/signals.py that can never be correct
INVALID_PREDICTIONS = {"PARSING_ERROR", "MISSING_IN_CACHE", ""}


def safe_is_correct(prediction: str, platinum_target: Optional[List[str]], dataset_name: str) -> Optional[bool]:
    """
    Wraps src/utils.py::check_prediction (the paper's official correctness rule).

    - Math datasets: float(platinum_target[0]) == float(prediction).
    - Text datasets (DROP): prediction in platinum_target (list of accepted strings).

    Edge cases:
    - Missing/empty platinum_target (rejected questions) -> None (undefined, not False).
    - Sentinel or non-numeric predictions that make float() fail -> False.

    Returns:
        Optional[bool]: True/False correctness, or None when no ground truth exists.
    """
    if not platinum_target:
        return None
    if prediction in INVALID_PREDICTIONS:
        return False
    try:
        return bool(check_prediction(prediction, platinum_target, None, dataset_name))
    except (ValueError, TypeError):
        return False


def legacy_agrees(prediction: str, legacy_target: Optional[List[str]]) -> Optional[bool]:
    """
    Mirrors the comparison rule used by src/auditor.py::SafetyGateAuditor
    (case-insensitive exact string match against the set of legacy strings),
    so that S_gate computed downstream is consistent with the Step 3 engine.

    Returns:
        Optional[bool]: None if no legacy label exists.
    """
    if not legacy_target:
        return None
    legacy_set = {str(t).strip().lower() for t in legacy_target}
    return str(prediction).strip().lower() in legacy_set


def build_rows(
    dataset_name: str,
    examples: Iterable[Dict[str, Any]],
    loader: OfflineDataLoader,
    model_list: List[str],
) -> List[Dict[str, Any]]:
    """
    Builds one output row per question.

    The blind block is computed BEFORE any label field is read, and only from
    the example prompt fields plus the cached model responses.

    Args:
        dataset_name (str): Benchmark identifier.
        examples (Iterable[dict]): Hugging Face test records (all statuses, incl. rejected).
        loader (OfflineDataLoader): Offline cache reader.
        model_list (List[str]): Target ensemble (24 paper models).

    Returns:
        List[dict]: Flat records ready for a DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        # ---------------- Operational / blind block ----------------
        ensemble = loader.get_ensemble_responses(dataset_name, ex, model_list=model_list)
        sig = extract_signals_for_ensemble(dataset_name, ex, ensemble["responses"])

        preds = sig["parsed_predictions"]
        vote_counts = Counter(preds.values()).most_common()
        second_count = vote_counts[1][1] if len(vote_counts) > 1 else 0

        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "idx": idx,
            "majority_ratio": sig["majority_ratio"],
            "entropy_confidence": sig["entropy_confidence"],
            "raw_entropy": sig["raw_entropy"],
            "cot_similarity": sig["cot_similarity"],
            "raw_cot_similarity": sig["raw_cot_similarity"],
            "length_stability": sig["length_stability"],
            "raw_length_stability": sig["raw_length_stability"],
            "hedging_score": sig["hedging_score"],
            "y_maj": sig["y_maj"],
            "modal_count": sig["modal_count"],
            "second_count": second_count,
            # Vote margin = modal votes - runner-up votes (used to stratify the split)
            "vote_margin": sig["modal_count"] - second_count,
            "n_distinct_answers": len(vote_counts),
            "num_models": sig["num_models"],
            "num_available": ensemble["num_available"],
            "parse_error_count": sig["parse_error_count"],
            "has_missing_cache": sig["has_missing_cache"],
            "parsed_predictions_json": json.dumps(preds, ensure_ascii=False),
        }

        # ---------------- Legacy label block (S_gate only) ----------------
        legacy_target = list(ex.get("original_target") or [])
        row["legacy_target_json"] = json.dumps(legacy_target, ensure_ascii=False)
        row["legacy_agrees_modal"] = legacy_agrees(sig["y_maj"], legacy_target)
        row["legacy_n_models_disagree"] = (
            sum(1 for p in preds.values() if not legacy_agrees(p, legacy_target)) if legacy_target else None
        )

        # ---------------- Ground-truth block (evaluation layer only) ----------------
        platinum_target = list(ex.get("platinum_target") or [])
        row["label_cleaning_status"] = ex.get("cleaning_status")
        row["label_platinum_target_json"] = json.dumps(platinum_target, ensure_ascii=False)
        row["label_is_correct"] = safe_is_correct(sig["y_maj"], platinum_target, dataset_name)
        # Number of individual models wrong w.r.t. platinum target (used by Step 5 paper_flagged proxy)
        row["label_n_models_wrong"] = (
            sum(1 for p in preds.values() if not safe_is_correct(p, platinum_target, dataset_name))
            if platinum_target else None
        )

        rows.append(row)

    return rows


def print_sanity_summary(df: pd.DataFrame) -> None:
    """Prints per-dataset counts and the unrestricted-majority error on non-rejected questions."""
    valid = df[df["label_cleaning_status"] != "rejected"]
    summary = valid.groupby("dataset").agg(
        n_valid=("idx", "size"),
        majority_error=("label_is_correct", lambda s: 1.0 - s.astype(float).mean()),
        unanimous=("majority_ratio", lambda s: int((s == 1.0).sum())),
        parse_errors=("parse_error_count", lambda s: int((s > 0).sum())),
        missing_cache=("has_missing_cache", "sum"),
        gate_triggers=("legacy_agrees_modal", lambda s: int((s == False).sum())),  # noqa: E712
    )
    rejected = (df["label_cleaning_status"] == "rejected").groupby(df["dataset"]).sum()
    summary["n_rejected"] = rejected
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-question evaluation table for Step 4.")
    parser.add_argument("--datasets", nargs="+", default=CORE_BENCHMARKS)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output", type=str, default="./outputs/tables/eval_table_core.csv")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap per dataset (smoke runs).")
    args = parser.parse_args()

    from datasets import load_dataset  # imported here so build_rows stays testable without HF

    loader = OfflineDataLoader(cache_dir=args.cache_dir)
    all_rows: List[Dict[str, Any]] = []

    for name in args.datasets:
        ds = load_dataset(HF_DATASET_REPO, name, split="test")
        examples = list(ds) if args.limit is None else list(ds)[: args.limit]
        print(f"[Export] {name}: {len(examples)} questions")
        all_rows.extend(build_rows(name, examples, loader, OFFICIAL_PAPER_MODELS))

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\n[Export] Sanity summary (non-rejected questions):")
    print_sanity_summary(df)
    print(f"\n[Done] {len(df)} rows written to {args.output}")


if __name__ == "__main__":
    main()