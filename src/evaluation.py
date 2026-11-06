"""
Provenance: [FORK NEW]

Purpose:
Step 4 Layer-1 evaluation of the Selective Auditor. Splits the per-question table
(scripts/export_eval_table.py) into a frozen 50/50 Dev/Eval partition, fits S_comb_calib
and calibrates operating thresholds on Dev only, and measures coverage, selective risk,
risk-coverage curves and AURC on the untouched Eval split. Produces Table 1
(outputs/tables/table1_selective_performance.csv) and the RC figure
(outputs/figures/rc_curves_core.png).

Hermetic separation:
- Scores are computed by `compute_scores`, which only receives BLIND_COLUMNS plus the
  legacy label (allowed for S_gate). It raises if any `label_*` column is passed.
- `label_*` columns are read only by `build_error_labels`, i.e. by the evaluation layer.

Error definitions (selected with --error-def):
- 'curation' (default): population = all questions; e(x) = 1 if the question was rejected
  by the paper's human review OR the majority answer differs from the platinum target.
  Rationale: an auto-accepted question is a curation failure if it needed a human,
  whether because its consensus label is wrong or because the question itself is flawed.
- 'label': population = non-rejected questions; e(x) = 1 iff y_maj != platinum_target
  (the original Step 4 definition; degenerate on the core suite: 3 errors in 756).

Usage:
    python3 src/evaluation.py --datasets gsm8k svamp drop --split_seed 42
"""

import argparse
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.auditor import DEFAULT_LINEAR_WEIGHTS


# ------------------------------------------------------------------------------
# Column contract
# ------------------------------------------------------------------------------
# Column order written by scripts/export_eval_table.py (used if a CSV lost its header)
EXPORT_COLUMNS: List[str] = [
    "dataset", "idx", "majority_ratio", "entropy_confidence", "raw_entropy", "cot_similarity",
    "raw_cot_similarity", "length_stability", "raw_length_stability", "hedging_score", "y_maj",
    "modal_count", "second_count", "vote_margin", "n_distinct_answers", "num_models",
    "num_available", "parse_error_count", "has_missing_cache", "parsed_predictions_json",
    "legacy_target_json", "legacy_agrees_modal", "legacy_n_models_disagree",
    "label_cleaning_status", "label_platinum_target_json", "label_is_correct", "label_n_models_wrong",
]

# Signals feeding the scores, in the order used by src/auditor.py: [M, V, Sim, LS, Hed]
SIGNAL_COLUMNS: List[str] = [
    "majority_ratio", "entropy_confidence", "cot_similarity", "length_stability", "hedging_score",
]

# Everything a score function may see (operational layer)
BLIND_COLUMNS: List[str] = SIGNAL_COLUMNS + [
    "dataset", "y_maj", "modal_count", "num_models", "parse_error_count", "has_missing_cache",
]
LEGACY_COLUMNS: List[str] = ["legacy_target_json"]

METHOD_ORDER: List[str] = ["S_maj", "S_ent", "S_comb_baseline", "S_comb_calib", "S_gate"]

ERROR_DEF_CURATION = "curation"
ERROR_DEF_LABEL = "label"


# ------------------------------------------------------------------------------
# Loading and labels (evaluation layer)
# ------------------------------------------------------------------------------
def _to_bool(value) -> Optional[bool]:
    """Parses CSV booleans ('True'/'False'/NaN) into Optional[bool]."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None


def load_eval_table(path: str, datasets: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads the exported per-question table. Recovers the column names if the header row
    is missing (e.g. after round-tripping through a spreadsheet application).
    """
    df = pd.read_csv(path, dtype={"y_maj": str}, keep_default_na=False, na_values=[""])
    if "dataset" not in df.columns:
        df = pd.read_csv(path, header=None, names=EXPORT_COLUMNS, dtype={"y_maj": str},
                         keep_default_na=False, na_values=[""])
    df["y_maj"] = df["y_maj"].fillna("").astype(str)
    for col in ("has_missing_cache", "label_is_correct"):
        df[col] = df[col].map(_to_bool)
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    return df.reset_index(drop=True)


def build_error_labels(df: pd.DataFrame, error_def: str = ERROR_DEF_CURATION) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Applies the chosen error definition (the ONLY function that reads label_* columns).

    Returns:
        (population_df, errors): the rows that belong to the evaluated population and
        a 0/1 array e(x) where 1 = accepting this question would be a curation error.
    """
    rejected = df["label_cleaning_status"].astype(str).str.lower().eq("rejected")
    majority_wrong = df["label_is_correct"].map(lambda v: v is False)

    if error_def == ERROR_DEF_CURATION:
        pop = df.copy()
        errors = (rejected | majority_wrong).to_numpy(dtype=int)
    elif error_def == ERROR_DEF_LABEL:
        pop = df[~rejected].copy()
        errors = majority_wrong[~rejected].to_numpy(dtype=int)
    else:
        raise ValueError(f"Unknown error_def '{error_def}'. Use 'curation' or 'label'.")
    return pop.reset_index(drop=True), errors


# ------------------------------------------------------------------------------
# Split (blind stratification)
# ------------------------------------------------------------------------------
def margin_bin(majority_ratio: float) -> str:
    """
    Blind vote-margin stratum: 'unanimous' (M=1), 'high' (0.75 <= M < 1), 'low' (M < 0.75).
    The 0.75 cut separates questions with at most a quarter of dissenting models.
    """
    if majority_ratio >= 1.0 - 1e-12:
        return "unanimous"
    if majority_ratio >= 0.75:
        return "high"
    return "low"


def stratified_split(df: pd.DataFrame, seed: int = 42, dev_fraction: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    50/50 Dev/Eval split stratified by dataset x margin bin. Uses only blind columns,
    so the partition itself cannot leak label information.

    Returns:
        (dev_idx, eval_idx): positional indices into df, sorted.
    """
    strata = df["dataset"].astype(str) + "|" + df["majority_ratio"].map(margin_bin)
    positions = np.arange(len(df))
    dev_idx, eval_idx = train_test_split(
        positions, train_size=dev_fraction, random_state=seed, stratify=strata,
    )
    return np.sort(dev_idx), np.sort(eval_idx)


# ------------------------------------------------------------------------------
# Scores (operational layer, blind)
# ------------------------------------------------------------------------------
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


def normalize_answer(text: str):
    """
    Canonical form used by the safety gate to compare y_maj with the legacy label.
    - If the answer contains a number (digits, or a leading number word), returns the first
      number as float, so '25 yards' == '25' and 'two' == '2' and '697%' == '697'.
    - Otherwise returns the lowercased string stripped of punctuation and articles.
    """
    s = str(text).strip().lower().replace(",", "")
    s = re.sub(r"[\"'*`]", "", s).strip().rstrip(".")
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    if match:
        return float(match.group())
    first = s.split(" ")[0] if s else ""
    if first in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[first])
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"\s+", " ", s)


def gate_triggered(y_maj: str, legacy_target_json: str) -> bool:
    """
    Safety gate (majority mode): True if the modal answer disagrees with every legacy
    label after normalization. No legacy label -> gate does not trigger.
    """
    try:
        legacy = json.loads(legacy_target_json) if isinstance(legacy_target_json, str) else []
    except json.JSONDecodeError:
        legacy = []
    if not legacy:
        return False
    y_norm = normalize_answer(y_maj)
    return all(normalize_answer(t) != y_norm for t in legacy)


def mandatory_deferral(df: pd.DataFrame) -> np.ndarray:
    """Questions that src/auditor.py always defers (parse error or incomplete cache)."""
    parse = df["parse_error_count"].fillna(0).astype(float) > 0
    missing = df["has_missing_cache"].map(lambda v: bool(v))
    incomplete = df["num_models"].astype(int) < df["num_models"].astype(int).max()
    return (parse | missing | incomplete).to_numpy()


def _check_blind(df: pd.DataFrame) -> None:
    leaked = [c for c in df.columns if c.startswith("label_")]
    if leaked:
        raise ValueError(f"Hermetic violation: label columns passed to the scoring layer: {leaked}")


def linear_baseline(df: pd.DataFrame) -> np.ndarray:
    """S_comb_baseline = 0.40 M + 0.35 V + 0.15 Sim + 0.10 LS (weights from src/auditor.py)."""
    w = DEFAULT_LINEAR_WEIGHTS
    s = (w["majority_ratio"] * df["majority_ratio"] + w["entropy_confidence"] * df["entropy_confidence"]
         + w["cot_similarity"] * df["cot_similarity"] + w["length_stability"] * df["length_stability"])
    return np.clip(s.to_numpy(dtype=float), 0.0, 1.0)


def fit_logistic(X: np.ndarray, y_ok: np.ndarray, c_reg: float = 1.0) -> LogisticRegression:
    """Same estimator as SelectiveAuditor.fit_calibrated (liblinear, L2, C=1)."""
    clf = LogisticRegression(C=c_reg, solver="liblinear", max_iter=1000)
    clf.fit(X, y_ok)
    return clf


def out_of_fold_scores(X: np.ndarray, y_ok: np.ndarray, seed: int = 42, n_splits: int = 5) -> np.ndarray:
    """
    Cross-fitted Dev scores: each Dev question is scored by a model that did not see it.
    Thresholds chosen on these scores are not optimistically biased by the logistic fit.
    """
    minority = int(min(y_ok.sum(), len(y_ok) - y_ok.sum()))
    k = min(n_splits, minority)
    if k < 2:
        raise ValueError(f"Too few errors on Dev ({minority}) to fit S_comb_calib.")
    oof = np.zeros(len(y_ok), dtype=float)
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y_ok):
        oof[te] = fit_logistic(X[tr], y_ok[tr]).predict_proba(X[te])[:, 1]
    return oof


@dataclass
class CalibrationResult:
    """Everything learned on Dev (weights and thresholds). Nothing here touches Eval labels."""
    logistic_weights: Optional[List[float]] = None
    logistic_intercept: Optional[float] = None
    taus: Dict[str, Dict[str, float]] = field(default_factory=dict)  # method -> {alpha: tau}
    notes: List[str] = field(default_factory=list)


def compute_scores(
    df: pd.DataFrame,
    calib_model: Optional[LogisticRegression] = None,
    calib_override: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes every S(x) from blind columns (+ legacy label for S_gate).
    Mandatory deferrals and gate triggers get score -inf, i.e. they are never accepted
    and are ranked last in the RC curve.

    Args:
        df: blind view of the questions (must not contain label_* columns).
        calib_model: fitted logistic for S_comb_calib (None -> S_comb_calib/S_gate are NaN).
        calib_override: precomputed calibrated scores (used for out-of-fold Dev scores).
    """
    _check_blind(df)
    forced = mandatory_deferral(df)
    X = df[SIGNAL_COLUMNS].to_numpy(dtype=float)

    scores = {
        "S_maj": df["majority_ratio"].to_numpy(dtype=float).copy(),
        "S_ent": df["entropy_confidence"].to_numpy(dtype=float).copy(),
        "S_comb_baseline": linear_baseline(df),
    }
    if calib_override is not None:
        calib = np.asarray(calib_override, dtype=float).copy()
    elif calib_model is not None:
        calib = calib_model.predict_proba(X)[:, 1]
    else:
        calib = np.full(len(df), np.nan)
    scores["S_comb_calib"] = calib

    gate = np.array([gate_triggered(y, l) for y, l in zip(df["y_maj"], df["legacy_target_json"])])
    scores["S_gate"] = np.where(gate, -np.inf, calib)

    for name in scores:
        if np.all(np.isnan(scores[name])):
            continue  # score unavailable (e.g. calibration skipped): keep it all-NaN so it is dropped
        scores[name] = np.where(forced, -np.inf, scores[name])
    return scores


def blind_view(df: pd.DataFrame) -> pd.DataFrame:
    """Projection of the table onto the columns the operational layer may read."""
    return df[BLIND_COLUMNS + LEGACY_COLUMNS].copy()


# ------------------------------------------------------------------------------
# Risk-coverage metrics (evaluation layer)
# ------------------------------------------------------------------------------
def rc_curve(scores: np.ndarray, errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Risk-coverage curve with exact tie handling.

    Questions are accepted in decreasing score order. Within a block of tied scores the
    order is undefined, so we use the EXPECTED risk under uniformly random tie-breaking:
    accepting j of the n_g tied items adds j * e_g / n_g expected errors. This equals the
    average over infinitely many random tie-breaks, without any sampling noise.

    Returns:
        coverage[k-1] = k/N and risk[k-1] = E[errors among top k] / k, for k = 1..N.
    """
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=float)
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-scores, kind="mergesort")
    s_sorted, e_sorted = scores[order], errors[order]

    expected_cum = np.empty(n, dtype=float)
    start, done_err = 0, 0.0
    while start < n:
        end = start
        while end + 1 < n and (s_sorted[end + 1] == s_sorted[start]):
            end += 1
        n_g = end - start + 1
        e_g = e_sorted[start:end + 1].sum()
        expected_cum[start:end + 1] = done_err + np.arange(1, n_g + 1) * (e_g / n_g)
        done_err += e_g
        start = end + 1

    k = np.arange(1, n + 1)
    return k / n, expected_cum / k


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """AURC = integral of R(C) dC by the trapezoidal rule over C in [1/N, 1]. Lower is better."""
    if len(coverage) < 2:
        return float("nan")
    return float(np.trapezoid(risk, coverage)) if hasattr(np, "trapezoid") else float(np.trapz(risk, coverage))


def risk_at_coverage(coverage: np.ndarray, risk: np.ndarray, target: float) -> float:
    """Expected selective risk when accepting round(target * N) top-ranked questions."""
    n = len(coverage)
    k = int(round(target * n))
    return float(risk[k - 1]) if 1 <= k <= n else float("nan")


def max_coverage(scores: np.ndarray) -> float:
    """Highest coverage achievable without accepting a mandatory deferral (-inf score)."""
    return float(np.isfinite(scores).mean()) if len(scores) else float("nan")


def operating_point(scores: np.ndarray, errors: np.ndarray, tau: float) -> Tuple[float, float, int]:
    """Deterministic rule ACCEPT iff S(x) >= tau. Returns (coverage, risk, n_accepted)."""
    accepted = np.isfinite(scores) & (scores >= tau)
    n_acc = int(accepted.sum())
    cov = n_acc / len(scores) if len(scores) else float("nan")
    risk = float(errors[accepted].mean()) if n_acc else float("nan")
    return cov, risk, n_acc


def calibrate_tau(scores: np.ndarray, errors: np.ndarray, alpha: float) -> float:
    """
    Largest-coverage threshold on Dev with empirical selective risk <= alpha.
    Candidates are the distinct finite scores; ties are accepted together, as the real
    rule S(x) >= tau would do. Returns +inf (accept nothing) if no threshold qualifies.
    """
    finite = np.isfinite(scores)
    best_tau, best_cov = float("inf"), -1.0
    for tau in np.unique(scores[finite])[::-1]:
        cov, risk, _ = operating_point(scores, errors, tau)
        if risk <= alpha + 1e-12 and cov > best_cov:
            best_tau, best_cov = float(tau), cov
    return best_tau


def bootstrap_ci(
    scores_by_method: Dict[str, np.ndarray],
    errors: np.ndarray,
    target_coverage: float = 0.75,
    n_boot: int = 1000,
    seed: int = 42,
    reference: str = "S_maj",
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Percentile 95% bootstrap CIs on Eval (questions resampled with replacement; the fitted
    models and thresholds are held fixed). Also returns the paired difference
    AURC(method) - AURC(reference): an interval containing 0 means no detectable difference.
    """
    rng = np.random.default_rng(seed)
    n = len(errors)
    samples: Dict[str, Dict[str, List[float]]] = {m: {"aurc": [], "risk": [], "diff": []} for m in scores_by_method}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        e = errors[idx]
        ref_cov, ref_risk = rc_curve(scores_by_method[reference][idx], e)
        ref_aurc = aurc(ref_cov, ref_risk)
        for m, s in scores_by_method.items():
            cov, risk = rc_curve(s[idx], e)
            a = aurc(cov, risk)
            samples[m]["aurc"].append(a)
            samples[m]["risk"].append(risk_at_coverage(cov, risk, target_coverage))
            samples[m]["diff"].append(a - ref_aurc)

    def pct(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) else (np.nan, np.nan)

    return {m: {k: pct(v) for k, v in d.items()} for m, d in samples.items()}


# ------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------
def calibrate_on_dev(dev: pd.DataFrame, dev_errors: np.ndarray, alphas=(0.05, 0.01), seed: int = 42):
    """
    Fits S_comb_calib and selects tau_alpha for every method using ONLY Dev rows.

    Returns:
        (final_logistic_model or None, CalibrationResult, dev_scores)
    """
    result = CalibrationResult()
    view = blind_view(dev)
    X = view[SIGNAL_COLUMNS].to_numpy(dtype=float)
    y_ok = 1 - dev_errors  # logistic target: 1 = safe to accept

    model, oof = None, None
    try:
        oof = out_of_fold_scores(X, y_ok, seed=seed)
        model = fit_logistic(X, y_ok)
        result.logistic_weights = [float(w) for w in model.coef_[0]]
        result.logistic_intercept = float(model.intercept_[0])
    except ValueError as exc:
        result.notes.append(f"S_comb_calib and S_gate skipped: {exc}")
        warnings.warn(str(exc))

    dev_scores = compute_scores(view, calib_override=oof)
    for method in METHOD_ORDER:
        s = dev_scores[method]
        if np.all(np.isnan(s)):
            continue
        result.taus[method] = {f"{a:.2f}": calibrate_tau(s, dev_errors, a) for a in alphas}
    return model, result, dev_scores


def evaluate_methods(
    scores: Dict[str, np.ndarray],
    errors: np.ndarray,
    calib: CalibrationResult,
    target_coverage: float = 0.75,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Builds the Table 1 rows (reference rows + every scored method) for one population."""
    n = len(errors)
    base = float(errors.mean()) if n else float("nan")
    rows = [
        {"method": "Full Human Review", "coverage_max": 0.0, "risk_at_C75": np.nan, "aurc": np.nan,
         "review_reduction_at_C75": 0.0},
        {"method": "Unrestricted Majority", "coverage_max": 1.0, "risk_at_C75": np.nan, "risk_full": base,
         "aurc": np.nan, "review_reduction_at_C75": np.nan, "review_reduction_full": 1.0},
    ]

    usable = {m: s for m, s in scores.items() if not np.all(np.isnan(s))}
    # Random abstention: a constant score makes every question one tie block -> flat risk = base rate.
    usable_with_random = {"Random Abstention": np.zeros(n), **usable}
    # Oracle: perfect ranking (errors last). Lower bound on AURC for this error rate.
    usable_with_random["Oracle (lower bound)"] = 1.0 - errors.astype(float)

    ci = bootstrap_ci(usable_with_random, errors, target_coverage, n_boot=n_boot, seed=seed) if n_boot else {}

    for m, s in usable_with_random.items():
        cov, risk = rc_curve(s, errors)
        row = {
            "method": m,
            "coverage_max": max_coverage(s),
            "risk_at_C75": risk_at_coverage(cov, risk, target_coverage),
            "aurc": aurc(cov, risk),
            "review_reduction_at_C75": target_coverage if max_coverage(s) >= target_coverage else np.nan,
        }
        if m in ci:
            row["risk_at_C75_ci"] = ci[m]["risk"]
            row["aurc_ci"] = ci[m]["aurc"]
            row["aurc_minus_S_maj_ci"] = ci[m]["diff"]
        for alpha, tau in calib.taus.get(m, {}).items():
            c_op, r_op, _ = operating_point(s, errors, tau)
            row[f"tau_{alpha}"] = tau
            row[f"eval_coverage_at_tau_{alpha}"] = c_op
            row[f"eval_risk_at_tau_{alpha}"] = r_op
        rows.append(row)

    table = pd.DataFrame(rows)
    table.insert(1, "n_eval", n)
    table.insert(2, "base_error_rate", base)
    return table


def _fmt_ci(ci) -> str:
    if not isinstance(ci, tuple) or any(pd.isna(v) for v in ci):
        return ""
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def plot_rc_curves(curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], path: str, error_def: str) -> None:
    """One panel per population (pooled + each dataset); one line per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Categorical slots 1-5 of a CVD-validated palette, assigned in fixed method order;
    # reference rows (random, oracle) are neutral grey with distinct dash patterns.
    styles = {
        "Random Abstention": dict(color="#8a8a86", linestyle="--", linewidth=1.2),
        "Oracle (lower bound)": dict(color="#8a8a86", linestyle=":", linewidth=1.4),
        "S_maj": dict(color="#2a78d6", linewidth=1.6),
        "S_ent": dict(color="#eb6834", linewidth=1.6),
        "S_comb_baseline": dict(color="#1baf7a", linewidth=1.6),
        "S_comb_calib": dict(color="#eda100", linewidth=2.4),
        "S_gate": dict(color="#e87ba4", linewidth=1.6),
    }
    panels = list(curves.keys())
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        for method, (cov, risk) in curves[panel].items():
            ax.plot(cov, risk, label=method, **styles.get(method, {}))
        ax.axvline(0.75, color="#cccccc", linewidth=0.8, zorder=0)
        ax.set_title(panel, fontsize=11)
        ax.set_xlabel("Coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee")
    axes[0].set_ylabel(f"Selective risk ({error_def})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(
    input_path: str,
    datasets: List[str],
    split_seed: int = 42,
    error_def: str = ERROR_DEF_CURATION,
    n_boot: int = 1000,
    table_path: str = "outputs/tables/table1_selective_performance.csv",
    figure_path: str = "outputs/figures/rc_curves_core.png",
    split_path: str = "outputs/tables/step4_split_assignment.csv",
    calib_path: str = "outputs/tables/step4_calibration.json",
) -> pd.DataFrame:
    """End-to-end Step 4: split -> calibrate on Dev -> evaluate on Eval -> export."""
    raw = load_eval_table(input_path, datasets)
    pop, errors = build_error_labels(raw, error_def)

    dev_idx, eval_idx = stratified_split(pop, seed=split_seed)
    dev, ev = pop.iloc[dev_idx].reset_index(drop=True), pop.iloc[eval_idx].reset_index(drop=True)
    e_dev, e_eval = errors[dev_idx], errors[eval_idx]

    model, calib, _ = calibrate_on_dev(dev, e_dev, seed=split_seed)

    # ---- Eval is touched only from here on, with every parameter frozen ----
    eval_scores = compute_scores(blind_view(ev), calib_model=model)

    tables, curves = [], {}
    for name, mask in [("pooled", np.ones(len(ev), bool))] + [(d, (ev["dataset"] == d).to_numpy()) for d in datasets]:
        sub_scores = {m: s[mask] for m, s in eval_scores.items()}
        t = evaluate_methods(sub_scores, e_eval[mask], calib, n_boot=n_boot, seed=split_seed)
        t.insert(0, "population", name)
        tables.append(t)
        curves[name] = {}
        for m in ["Random Abstention", "Oracle (lower bound)"] + METHOD_ORDER:
            if m == "Random Abstention":
                s = np.zeros(mask.sum())
            elif m.startswith("Oracle"):
                s = 1.0 - e_eval[mask].astype(float)
            else:
                s = sub_scores.get(m)
                if s is None or np.all(np.isnan(s)):
                    continue
            curves[name][m] = rc_curve(s, e_eval[mask])

    table = pd.concat(tables, ignore_index=True)
    for col in [c for c in table.columns if c.endswith("_ci")]:
        table[col] = table[col].map(_fmt_ci)
    table.insert(1, "error_def", error_def)

    os.makedirs(os.path.dirname(os.path.abspath(table_path)), exist_ok=True)
    table.to_csv(table_path, index=False)
    plot_rc_curves(curves, figure_path, error_def)

    split = pd.DataFrame({
        "dataset": pop["dataset"], "idx": pop["idx"],
        "split": np.where(np.isin(np.arange(len(pop)), dev_idx), "dev", "eval"),
        "margin_bin": pop["majority_ratio"].map(margin_bin),
    })
    split.to_csv(split_path, index=False)
    with open(calib_path, "w") as f:
        json.dump({
            "error_def": error_def, "split_seed": split_seed,
            "n_dev": int(len(dev)), "n_eval": int(len(ev)),
            "dev_errors": int(e_dev.sum()), "eval_errors": int(e_eval.sum()),
            "signal_order": SIGNAL_COLUMNS,
            "logistic_weights": calib.logistic_weights, "logistic_intercept": calib.logistic_intercept,
            "taus": calib.taus, "notes": calib.notes,
        }, f, indent=2)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Dev/Eval calibration and Layer-1 evaluation.")
    parser.add_argument("--input", default="outputs/tables/eval_table_core.csv")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "svamp", "drop"])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--error-def", default=ERROR_DEF_CURATION, choices=[ERROR_DEF_CURATION, ERROR_DEF_LABEL])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--table", default="outputs/tables/table1_selective_performance.csv")
    parser.add_argument("--figure", default="outputs/figures/rc_curves_core.png")
    args = parser.parse_args()

    table = run(args.input, args.datasets, args.split_seed, args.error_def, args.n_boot, args.table, args.figure)
    cols = ["population", "method", "n_eval", "base_error_rate", "coverage_max", "risk_at_C75",
            "aurc", "aurc_ci", "tau_0.05", "eval_coverage_at_tau_0.05", "eval_risk_at_tau_0.05"]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    print(f"\n[Done] Table 1 -> {args.table}\n[Done] Figure  -> {args.figure}")


if __name__ == "__main__":
    main()