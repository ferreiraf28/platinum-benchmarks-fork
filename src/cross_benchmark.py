"""
Provenance: [FORK NEW]

Purpose:
Step 6: Cross-Benchmark Generalization of the Post-Hoc Selective Auditor.
Evaluates out-of-distribution zero-shot transfer of the calibrated auditor
(weights and thresholds frozen from Step 4 dev split) across all 11 held-out
textual benchmarks (and aggregated across all 14 benchmarks).

Evaluates:
- Layer 1 Selective Performance: Coverage and Selective Risk at frozen tau_0.05 and tau_0.01,
  Risk @ C=75%, AURC, and RC Curves per benchmark and domain.
- Layer 2 Review Queue Reduction: Reduction percentage and risk inside the paper queue
  (questions where cleaning_status != 'consensus').

Exports:
- outputs/tables/table4_cross_benchmark_generalization.csv
- outputs/tables/table5_cross_benchmark_queue.csv
- outputs/tables/table6_paper_head_to_head.csv
- outputs/figures/rc_curves_cross_benchmark.png

Usage:
    python3 src/cross_benchmark.py
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation import (
    ERROR_DEF_CURATION,
    METHOD_ORDER,
    aurc,
    blind_view,
    build_error_labels,
    compute_scores,
    load_eval_table,
    max_coverage,
    operating_point,
    rc_curve,
    risk_at_coverage,
)


# Domains grouping the 14 textual benchmarks
BENCHMARK_DOMAINS: Dict[str, str] = {
    "gsm8k": "Math",
    "svamp": "Math",
    "singleop": "Math",
    "singleq": "Math",
    "multiarith": "Math",
    "mmlu_math": "Math",
    "drop": "Reading / QA",
    "squad": "Reading / QA",
    "hotpotqa": "Reading / QA",
    "bbh_logical_deduction_three_objects": "Symbolic & Logic",
    "bbh_object_counting": "Symbolic & Logic",
    "bbh_navigate": "Symbolic & Logic",
    "winograd_wsc": "Symbolic & Logic",
    "tab_fact": "Tabular",
}

CORE_BENCHMARKS = ["gsm8k", "svamp", "drop"]


def load_calibration_model(calib_path: str) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Loads frozen logistic parameters and decision thresholds from Step 4 calibration file."""
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    clf = LogisticRegression()
    clf.coef_ = np.array([calib_data["logistic_weights"]])
    clf.intercept_ = np.array([calib_data["logistic_intercept"]])
    clf.classes_ = np.array([0, 1])
    return clf, calib_data


def evaluate_partition(
    df: pd.DataFrame,
    clf: LogisticRegression,
    calib_data: Dict[str, Any],
    partition_name: str,
) -> Dict[str, Any]:
    """
    Evaluates all scoring methods on a partition (e.g. benchmark or domain).
    Uses the curation error definition and frozen decision thresholds.
    """
    if len(df) == 0:
        res: Dict[str, Any] = {
            "partition": partition_name,
            "N": 0,
            "n_errors": 0,
            "base_error": 0.0,
            "Random_aurc": float("nan"),
        }
        for method in METHOD_ORDER:
            res[f"{method}_aurc"] = float("nan")
            res[f"{method}_risk_at_75"] = float("nan")
            res[f"{method}_cov_005"] = float("nan")
            res[f"{method}_risk_005"] = float("nan")
            res[f"{method}_acc_005"] = 0
            res[f"{method}_cov_001"] = float("nan")
            res[f"{method}_risk_001"] = float("nan")
            res[f"{method}_acc_001"] = 0
        return res

    pop, errors = build_error_labels(df, error_def=ERROR_DEF_CURATION)
    scores = compute_scores(blind_view(pop), calib_model=clf)

    n_total = len(pop)
    n_errors = int(errors.sum())
    base_error = n_errors / n_total if n_total > 0 else 0.0

    results: Dict[str, Any] = {
        "partition": partition_name,
        "N": n_total,
        "n_errors": n_errors,
        "base_error": base_error,
    }

    # Evaluate each scoring method
    for method in METHOD_ORDER:
        sc = scores[method]
        cov_curve, risk_curve = rc_curve(sc, errors)
        method_aurc = aurc(cov_curve, risk_curve)
        risk_75 = risk_at_coverage(cov_curve, risk_curve, 0.75)

        t_05 = calib_data["taus"][method]["0.05"]
        t_01 = calib_data["taus"][method]["0.01"]

        cov_05, risk_05, acc_05 = operating_point(sc, errors, t_05)
        cov_01, risk_01, acc_01 = operating_point(sc, errors, t_01)

        results[f"{method}_aurc"] = method_aurc
        results[f"{method}_risk_at_75"] = risk_75
        results[f"{method}_cov_005"] = cov_05
        results[f"{method}_risk_005"] = risk_05
        results[f"{method}_acc_005"] = acc_05
        results[f"{method}_cov_001"] = cov_01
        results[f"{method}_risk_001"] = risk_01
        results[f"{method}_acc_001"] = acc_01

    # Random abstention flat baseline
    rand_cov = np.linspace(0.0, 1.0, 101)
    rand_risk = np.full_like(rand_cov, base_error)
    results["Random_aurc"] = aurc(rand_cov, rand_risk)

    return results


def evaluate_queue_reduction(
    df: pd.DataFrame,
    clf: LogisticRegression,
    calib_data: Dict[str, Any],
    partition_name: str,
) -> Dict[str, Any]:
    """
    Evaluates human inspection queue reduction on the subset where cleaning_status != 'consensus'.
    """
    if len(df) == 0:
        return {
            "partition": partition_name,
            "N_total": 0,
            "N_queue": 0,
            "queue_pct": 0.0,
            "queue_base_error": 0.0,
            "accepted": 0,
            "deferred": 0,
            "queue_reduction": 0.0,
            "queue_risk": 0.0,
        }

    pop, errors = build_error_labels(df, error_def=ERROR_DEF_CURATION)
    scores = compute_scores(blind_view(pop), calib_model=clf)

    # Queue definition: instances requiring review under the paper's heuristic
    queue_mask = pop["label_cleaning_status"].astype(str).str.lower() != "consensus"
    n_pop = len(pop)
    n_queue = int(queue_mask.sum())

    if n_queue == 0:
        return {
            "partition": partition_name,
            "N_total": n_pop,
            "N_queue": 0,
            "queue_pct": 0.0,
            "queue_base_error": 0.0,
            "accepted": 0,
            "deferred": 0,
            "queue_reduction": 0.0,
            "queue_risk": 0.0,
        }

    sc_q = scores["S_comb_calib"][queue_mask]
    err_q = errors[queue_mask]
    t_05 = calib_data["taus"]["S_comb_calib"]["0.05"]

    cov_q, risk_q, n_acc = operating_point(sc_q, err_q, t_05)

    return {
        "partition": partition_name,
        "N_total": n_pop,
        "N_queue": n_queue,
        "queue_pct": n_queue / n_pop if n_pop > 0 else 0.0,
        "queue_base_error": float(err_q.mean()),
        "accepted": n_acc,
        "deferred": n_queue - n_acc,
        "queue_reduction": cov_q,
        "queue_risk": risk_q,
    }


def generate_table4_generalization(
    df_all: pd.DataFrame,
    clf: LogisticRegression,
    calib_data: Dict[str, Any],
) -> pd.DataFrame:
    """Generates Table 4: selective classification performance across datasets and domains."""
    rows = []

    # 1. Global partitions
    df_core = df_all[df_all["dataset"].isin(CORE_BENCHMARKS)]
    df_heldout = df_all[~df_all["dataset"].isin(CORE_BENCHMARKS)]

    rows.append(evaluate_partition(df_all, clf, calib_data, "All 14 Benchmarks (Pooled)"))
    rows.append(evaluate_partition(df_core, clf, calib_data, "Core Suite (Dev+Eval, N=850)"))
    rows.append(evaluate_partition(df_heldout, clf, calib_data, "Held-Out Suite (N=2212)"))

    # 2. Domain partitions
    df_all["domain"] = df_all["dataset"].map(BENCHMARK_DOMAINS)
    for domain in ["Math", "Reading / QA", "Symbolic & Logic", "Tabular"]:
        df_domain = df_all[df_all["domain"] == domain]
        rows.append(evaluate_partition(df_domain, clf, calib_data, f"Domain: {domain}"))

    # 3. Individual benchmark partitions
    for d in df_all["dataset"].unique():
        df_d = df_all[df_all["dataset"] == d]
        rows.append(evaluate_partition(df_d, clf, calib_data, f"Dataset: {d}"))

    return pd.DataFrame(rows)


def generate_table5_queue_reduction(
    df_all: pd.DataFrame,
    clf: LogisticRegression,
    calib_data: Dict[str, Any],
) -> pd.DataFrame:
    """Generates Table 5: review queue reduction across datasets and domains."""
    rows = []

    # 1. Global partitions
    df_core = df_all[df_all["dataset"].isin(CORE_BENCHMARKS)]
    df_heldout = df_all[~df_all["dataset"].isin(CORE_BENCHMARKS)]

    rows.append(evaluate_queue_reduction(df_all, clf, calib_data, "All 14 Benchmarks (Pooled)"))
    rows.append(evaluate_queue_reduction(df_core, clf, calib_data, "Core Suite (N=850)"))
    rows.append(evaluate_queue_reduction(df_heldout, clf, calib_data, "Held-Out Suite (N=2212)"))

    # 2. Domain partitions
    df_all["domain"] = df_all["dataset"].map(BENCHMARK_DOMAINS)
    for domain in ["Math", "Reading / QA", "Symbolic & Logic", "Tabular"]:
        df_domain = df_all[df_all["domain"] == domain]
        rows.append(evaluate_queue_reduction(df_domain, clf, calib_data, f"Domain: {domain}"))

    # 3. Individual benchmark partitions
    for d in df_all["dataset"].unique():
        df_d = df_all[df_all["dataset"] == d]
        rows.append(evaluate_queue_reduction(df_d, clf, calib_data, f"Dataset: {d}"))

    return pd.DataFrame(rows)


def _compute_policy_stats(
    accept_mask: np.ndarray,
    errors: np.ndarray,
    paper_deferred_count: int,
    partition_name: str,
    policy_name: str,
) -> Dict[str, Any]:
    """Computes head-to-head operational metrics for a single binary triage policy."""
    n_total = len(errors)
    n_errors = int(errors.sum())
    defer_mask = ~accept_mask

    n_acc = int(accept_mask.sum())
    n_def = int(defer_mask.sum())

    cov = n_acc / n_total if n_total > 0 else 0.0
    rev_rate = n_def / n_total if n_total > 0 else 0.0

    err_in_acc = int((accept_mask & errors).sum())
    sel_risk = err_in_acc / n_acc if n_acc > 0 else 0.0

    errors_caught = int((defer_mask & errors).sum())
    error_recall = errors_caught / n_errors if n_errors > 0 else 1.0

    false_alarms = int((defer_mask & (~errors)).sum())
    false_alarm_rate = false_alarms / n_def if n_def > 0 else 0.0

    workload_red = (
        (paper_deferred_count - n_def) / paper_deferred_count
        if paper_deferred_count > 0
        else 0.0
    )

    return {
        "partition": partition_name,
        "policy": policy_name,
        "N_total": n_total,
        "n_errors": n_errors,
        "accepted_count": n_acc,
        "coverage": cov,
        "deferred_count": n_def,
        "review_rate": rev_rate,
        "errors_in_accepted": err_in_acc,
        "selective_risk": sel_risk,
        "errors_caught": errors_caught,
        "error_recall": error_recall,
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarm_rate,
        "workload_reduction_vs_paper": workload_red,
    }


def evaluate_paper_head_to_head(
    df: pd.DataFrame,
    clf: LogisticRegression,
    calib_data: Dict[str, Any],
    partition_name: str,
) -> List[Dict[str, Any]]:
    """
    Compares Vendrow et al. (2025) paper heuristic head-to-head against
    Selective Auditor (S_comb_calib) and Safety-Gated Auditor (S_gate) at tau_0.05.
    """
    if len(df) == 0:
        return []

    pop, errors = build_error_labels(df, error_def=ERROR_DEF_CURATION)
    scores = compute_scores(blind_view(pop), calib_model=clf)
    errors_bool = errors.astype(bool)

    accept_paper = (
        pop["label_cleaning_status"].astype(str).str.lower() == "consensus"
    ).to_numpy()
    paper_deferred_count = int((~accept_paper).sum())

    t_calib = calib_data["taus"]["S_comb_calib"]["0.05"]
    accept_calib = (scores["S_comb_calib"] >= t_calib).astype(bool)

    t_gate = calib_data["taus"]["S_gate"]["0.05"]
    accept_gate = (scores["S_gate"] >= t_gate).astype(bool)

    return [
        _compute_policy_stats(
            accept_paper, errors_bool, paper_deferred_count, partition_name, "Paper Heuristic (Vendrow et al.)"
        ),
        _compute_policy_stats(
            accept_calib, errors_bool, paper_deferred_count, partition_name, "Selective Auditor (S_comb_calib)"
        ),
        _compute_policy_stats(
            accept_gate, errors_bool, paper_deferred_count, partition_name, "Safety-Gated Auditor (S_gate)"
        ),
    ]


def generate_table6_head_to_head(
    df_all: pd.DataFrame,
    clf: LogisticRegression,
    calib_data: Dict[str, Any],
) -> pd.DataFrame:
    """Generates Table 6: head-to-head comparison with the original paper across suites & domains."""
    rows: List[Dict[str, Any]] = []
    df_core = df_all[df_all["dataset"].isin(CORE_BENCHMARKS)]
    df_heldout = df_all[~df_all["dataset"].isin(CORE_BENCHMARKS)]

    rows.extend(evaluate_paper_head_to_head(df_all, clf, calib_data, "All 14 Benchmarks (Pooled)"))
    rows.extend(evaluate_paper_head_to_head(df_core, clf, calib_data, "Core Suite (N=850)"))
    rows.extend(evaluate_paper_head_to_head(df_heldout, clf, calib_data, "Held-Out Suite (N=2212)"))

    df_work = df_all.copy()
    df_work["domain"] = df_work["dataset"].map(BENCHMARK_DOMAINS)
    for domain in ["Math", "Reading / QA", "Symbolic & Logic", "Tabular"]:
        df_domain = df_work[df_work["domain"] == domain]
        rows.extend(evaluate_paper_head_to_head(df_domain, clf, calib_data, f"Domain: {domain}"))

    return pd.DataFrame(rows)


def plot_cross_benchmark_rc_curves(
    df_all: pd.DataFrame,
    clf: LogisticRegression,
    calib_data: Dict[str, Any],
    output_path: str,
) -> None:
    """Generates comparative Risk-Coverage curves across held-out domains."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left plot: Core vs Held-Out vs Full Suite
    ax1 = axes[0]
    pop_all, err_all = build_error_labels(df_all, error_def=ERROR_DEF_CURATION)
    sc_all = compute_scores(blind_view(pop_all), calib_model=clf)

    df_core = df_all[df_all["dataset"].isin(CORE_BENCHMARKS)]
    pop_core, err_core = build_error_labels(df_core, error_def=ERROR_DEF_CURATION)
    sc_core = compute_scores(blind_view(pop_core), calib_model=clf)

    df_heldout = df_all[~df_all["dataset"].isin(CORE_BENCHMARKS)]
    pop_heldout, err_heldout = build_error_labels(df_heldout, error_def=ERROR_DEF_CURATION)
    sc_heldout = compute_scores(blind_view(pop_heldout), calib_model=clf)

    c_c, r_c = rc_curve(sc_core["S_comb_calib"], err_core)
    c_h, r_h = rc_curve(sc_heldout["S_comb_calib"], err_heldout)
    c_a, r_a = rc_curve(sc_all["S_comb_calib"], err_all)

    ax1.plot(c_c, r_c, label=f"Core Suite (N=850, AURC={aurc(c_c, r_c):.3f})", color="#1f77b4", lw=2)
    ax1.plot(c_h, r_h, label=f"Held-Out Suite (N=2212, AURC={aurc(c_h, r_h):.3f})", color="#2ca02c", lw=2)
    ax1.plot(c_a, r_a, label=f"All 14 Benchmarks (N=3062, AURC={aurc(c_a, r_a):.3f})", color="#d62728", lw=2, ls="--")
    ax1.axhline(0.05, color="gray", ls=":", alpha=0.7, label="Risk Target (5%)")
    ax1.set_xlabel("Coverage (Proportion of Accepted Questions)")
    ax1.set_ylabel("Empirical Selective Risk (Error Rate)")
    ax1.set_title("RC Curves: Core vs Held-Out vs Pooled", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.01, 0.20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Right plot: Domain comparison
    ax2 = axes[1]
    df_all["domain"] = df_all["dataset"].map(BENCHMARK_DOMAINS)
    colors = {"Math": "#1f77b4", "Reading / QA": "#ff7f0e", "Symbolic & Logic": "#2ca02c", "Tabular": "#9467bd"}

    for domain in ["Math", "Reading / QA", "Symbolic & Logic", "Tabular"]:
        df_domain = df_all[df_all["domain"] == domain]
        pop_d, err_d = build_error_labels(df_domain, error_def=ERROR_DEF_CURATION)
        sc_d = compute_scores(blind_view(pop_d), calib_model=clf)
        c_d, r_d = rc_curve(sc_d["S_comb_calib"], err_d)
        ax2.plot(c_d, r_d, label=f"{domain} (N={len(pop_d)}, AURC={aurc(c_d, r_d):.3f})", color=colors[domain], lw=2)

    ax2.axhline(0.05, color="gray", ls=":", alpha=0.7, label="Risk Target (5%)")
    ax2.set_xlabel("Coverage (Proportion of Accepted Questions)")
    ax2.set_ylabel("Empirical Selective Risk (Error Rate)")
    ax2.set_title("RC Curves by Cognitive Domain", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.01, 0.25)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved cross-benchmark RC curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 6: Cross-Benchmark Generalization.")
    parser.add_argument("--eval-table-all", default="outputs/tables/eval_table_all.csv")
    parser.add_argument("--calibration", default="outputs/tables/step4_calibration.json")
    parser.add_argument("--output-table4", default="outputs/tables/table4_cross_benchmark_generalization.csv")
    parser.add_argument("--output-table5", default="outputs/tables/table5_cross_benchmark_queue.csv")
    parser.add_argument("--output-table6", default="outputs/tables/table6_paper_head_to_head.csv")
    parser.add_argument("--output-figure", default="outputs/figures/rc_curves_cross_benchmark.png")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 6: CROSS-BENCHMARK GENERALIZATION AUDIT")
    print("=" * 80)

    clf, calib_data = load_calibration_model(args.calibration)
    print(f"[Setup] Loaded frozen logistic model and decision thresholds from {args.calibration}")
    print(f"        Logistic weights: {calib_data['logistic_weights']}")
    print(f"        Intercept: {calib_data['logistic_intercept']:.4f}")
    print(f"        tau_0.05 (S_comb_calib): {calib_data['taus']['S_comb_calib']['0.05']:.4f}")

    df_all = load_eval_table(args.eval_table_all)
    print(f"[Data] Loaded {len(df_all)} instances across {df_all['dataset'].nunique()} benchmarks.")

    print("\n[Audit] Generating Table 4 (Layer 1 Selective Classification Generalization)...")
    t4 = generate_table4_generalization(df_all, clf, calib_data)
    t4.to_csv(args.output_table4, index=False)
    print(f"[Done] Table 4 exported to {args.output_table4}")

    print("\n[Audit] Generating Table 5 (Layer 2 Queue Reduction Generalization)...")
    t5 = generate_table5_queue_reduction(df_all, clf, calib_data)
    t5.to_csv(args.output_table5, index=False)
    print(f"[Done] Table 5 exported to {args.output_table5}")

    print("\n[Audit] Generating Table 6 (Head-to-Head vs Original Paper)...")
    t6 = generate_table6_head_to_head(df_all, clf, calib_data)
    t6.to_csv(args.output_table6, index=False)
    print(f"[Done] Table 6 exported to {args.output_table6}")

    print("\n[Audit] Plotting Cross-Benchmark Risk-Coverage Curves...")
    plot_cross_benchmark_rc_curves(df_all, clf, calib_data, args.output_figure)

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS: HELD-OUT SUITE VS CORE SUITE")
    print("=" * 80)
    summary_cols = ["partition", "N", "base_error", "S_comb_calib_cov_005", "S_comb_calib_risk_005", "S_comb_calib_aurc", "Random_aurc"]
    print(t4[summary_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS: REVIEW QUEUE REDUCTION")
    print("=" * 80)
    q_cols = ["partition", "N_total", "N_queue", "queue_pct", "queue_base_error", "queue_reduction", "queue_risk"]
    print(t5[q_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS: HEAD-TO-HEAD VS ORIGINAL PAPER")
    print("=" * 80)
    h2h_cols = [
        "partition", "policy", "coverage", "deferred_count", "review_rate",
        "selective_risk", "false_alarms", "false_alarm_rate", "workload_reduction_vs_paper"
    ]
    print(t6[h2h_cols].to_string(index=False))


if __name__ == "__main__":
    main()
