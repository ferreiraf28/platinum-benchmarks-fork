"""
Provenance: [FORK NEW]

Purpose:
Validation script for Step 2 of the implementation playbook.
Computes and inspects the empirical distribution of the 5 uncertainty signals
(M, V, Sim, LS, Hed) on real benchmark examples from the local cache without
peeking at ground-truth targets. Generates summary statistics and correlations.
"""

import os
import sys
from typing import Dict, Any, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from datasets import load_dataset
import pandas as pd
import numpy as np

from src.offline_loader import OfflineDataLoader, OFFICIAL_PAPER_MODELS
from src.signals import extract_signals_for_ensemble


def run_signals_audit(dataset_name: str = "gsm8k", num_samples: int = 100) -> pd.DataFrame:
    """
    Extracts uncertainty signals for a sample of benchmark instances.

    Args:
        dataset_name (str): Benchmark dataset name ('gsm8k', 'svamp', 'drop').
        num_samples (int): Number of instances to evaluate.

    Returns:
        pd.DataFrame: DataFrame containing signal distributions and metadata.
    """
    print(f"\n[Step 2 Audit] Loading dataset '{dataset_name}' (test split) ...")
    dataset = load_dataset("madrylab/platinum-bench-paper-version", dataset_name, split="test")
    loader = OfflineDataLoader()

    n = min(num_samples, len(dataset))
    print(f"[Step 2 Audit] Computing 5 uncertainty signals for {n} instances across 24 models ...")

    records: List[Dict[str, Any]] = []

    for i in range(n):
        ex = dataset[i]
        ensemble = loader.get_ensemble_responses(dataset_name, ex, OFFICIAL_PAPER_MODELS)
        raw_responses = ensemble["responses"]

        # Strictly blind extraction: only problem metadata and raw model texts
        signals = extract_signals_for_ensemble(dataset_name, ex, raw_responses)

        records.append({
            "idx": i,
            "benchmark": dataset_name,
            "majority_ratio": signals["majority_ratio"],
            "entropy_confidence": signals["entropy_confidence"],
            "raw_entropy": signals["raw_entropy"],
            "cot_similarity": signals["cot_similarity"],
            "raw_cot_similarity": signals["raw_cot_similarity"],
            "length_stability": signals["length_stability"],
            "raw_length_stability": signals["raw_length_stability"],
            "hedging_score": signals["hedging_score"],
            "modal_count": signals["modal_count"],
            "num_models": signals["num_models"],
            "parse_error_count": signals["parse_error_count"],
            "cleaning_status": ex.get("cleaning_status"),  # Kept purely for post-hoc analysis
        })

    df = pd.DataFrame(records)

    print("\n" + "=" * 80)
    print(f"       EMPIRICAL SIGNAL DISTRIBUTION SUMMARY ({dataset_name.upper()} - N={n})")
    print("=" * 80)
    signal_cols = [
        "majority_ratio",
        "entropy_confidence",
        "cot_similarity",
        "raw_cot_similarity",
        "length_stability",
        "raw_length_stability",
        "hedging_score",
    ]
    stats_df = df[signal_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(stats_df.to_string())

    print("\n" + "-" * 80)
    print("       PAIRWISE PEARSON CORRELATION MATRIX BETWEEN SIGNALS")
    print("-" * 80)
    corr_cols = ["majority_ratio", "entropy_confidence", "cot_similarity", "length_stability", "hedging_score"]
    corr_matrix = df[corr_cols].corr()
    print(corr_matrix.round(3).to_string())

    # Stratified breakdown by paper cleaning status
    print("\n" + "-" * 80)
    print("       MEAN SIGNAL VALUES STRATIFIED BY PAPER CLEANING STATUS")
    print("-" * 80)
    status_group = df.groupby("cleaning_status")[corr_cols].mean()
    print(status_group.round(3).to_string())
    print("=" * 80 + "\n")

    # Export results
    os.makedirs("outputs/tables", exist_ok=True)
    out_path = f"outputs/tables/signals_{dataset_name}_sample_{n}.csv"
    df.to_csv(out_path, index=False)
    print(f"[Export] Saved signal records to: {out_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit Uncertainty Signals Distribution")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name ('gsm8k', 'drop', 'svamp', or 'all')",
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of instances to evaluate per benchmark")
    args = parser.parse_args()

    if args.dataset.lower() == "all":
        datasets = ["gsm8k", "svamp", "drop"]
        all_dfs = []
        for ds in datasets:
            df_ds = run_signals_audit(dataset_name=ds, num_samples=args.num_samples)
            all_dfs.append(df_ds)

        pooled_df = pd.concat(all_dfs, ignore_index=True)
        pooled_path = f"outputs/tables/signals_core_suite_sample_{len(pooled_df)}.csv"
        pooled_df.to_csv(pooled_path, index=False)
        print("\n" + "#" * 80)
        print(f"       POOLED CORE SUITE DISTRIBUTION SUMMARY (GSM8K + SVAMP + DROP - N={len(pooled_df)})")
        print("#" * 80)
        signal_cols = [
            "majority_ratio",
            "entropy_confidence",
            "cot_similarity",
            "raw_cot_similarity",
            "length_stability",
            "raw_length_stability",
            "hedging_score",
        ]
        print(pooled_df[signal_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())
        print("\n" + "-" * 80)
        print("       POOLED CORRELATION MATRIX")
        print("-" * 80)
        corr_cols = ["majority_ratio", "entropy_confidence", "cot_similarity", "length_stability", "hedging_score"]
        print(pooled_df[corr_cols].corr().round(3).to_string())
        print(f"\n[Export] Saved pooled records to: {pooled_path}\n")
    else:
        run_signals_audit(dataset_name=args.dataset, num_samples=args.num_samples)
