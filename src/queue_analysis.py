"""
Provenance: [FORK NEW]

Purpose:
Step 5 Layer-2 evaluation. Reconstructs the paper's human-review queue, applies the
auditor calibrated in Step 4 (frozen weights and thresholds, Dev-only) to the queued
questions of the Eval split, and reports how much of that queue could be removed and at
what risk, broken down by vote margin (Table 2), plus a qualitative error taxonomy (Table 3).

Paper queue definition (decision, Step 5):
- A question entered the paper's manual review iff at least one model disagreed with the
  ORIGINAL (legacy) label. The released `cleaning_status` records exactly this outcome:
  `consensus` = never reviewed; `verified` / `revised` / `rejected` = reviewed.
- We therefore use paper_flagged(x) = (cleaning_status != "consensus"). This is the queue
  humans actually processed, not an approximation.
- The guide's formula (exists m: y_m != platinum_target) under-counts DROP by 130 questions:
  after revision, the platinum target was expanded to accept the models' answers
  (e.g. '25 yards'), so those reviewed questions no longer look disputed.

Usage:
    python3 src/queue_analysis.py --datasets gsm8k svamp drop
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd

from src.evaluation import (
    blind_view,
    build_error_labels,
    calibrate_on_dev,
    calibrate_tau,
    compute_scores,
    load_eval_table,
    stratified_split,
)


# Vote-margin buckets: number of models that did NOT give the modal answer (K = 24)
MARGIN_ORDER: List[str] = ["24-0", "23-1", "22-2", "21-3", "<=20"]


def paper_flagged(df: pd.DataFrame) -> np.ndarray:
    """
    True for questions the paper sent to human review (any status other than 'consensus').
    Reads a label column, so it belongs to the evaluation layer only.
    """
    return df["label_cleaning_status"].astype(str).str.lower().ne("consensus").to_numpy()


def margin_bucket(modal_count: int, num_models: int) -> str:
    """
    Vote-margin bucket from the modal vote count, e.g. 23 of 24 -> '23-1'.
    '24-0' = all models agree with each other (the question can still be in the paper
    queue if that shared answer differs from the legacy label).
    """
    dissent = int(num_models) - int(modal_count)
    if dissent <= 0:
        return "24-0"
    if dissent <= 3:
        return f"{int(num_models) - dissent}-{dissent}"
    return "<=20"


def build_queue_frame(
    input_path: str,
    datasets: Optional[List[str]] = None,
    split_seed: int = 42,
    error_def: str = "curation",
) -> pd.DataFrame:
    """
    Loads the Step 4 export, reproduces the Step 4 Dev/Eval split exactly, and annotates
    each question with its split, curation error, paper-queue membership and margin bucket.

    Returns:
        DataFrame with one row per question of the evaluated population and the extra
        columns: split, error, paper_flagged, margin.
    """
    raw = load_eval_table(input_path, datasets)
    pop, errors = build_error_labels(raw, error_def)
    dev_idx, _ = stratified_split(pop, seed=split_seed)

    out = pop.copy()
    out["split"] = np.where(np.isin(np.arange(len(pop)), dev_idx), "dev", "eval")
    out["error"] = errors
    out["paper_flagged"] = paper_flagged(pop)
    out["margin"] = [margin_bucket(m, k) for m, k in zip(pop["modal_count"], pop["num_models"])]
    return out


def queue_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Size of the paper queue per split, dataset and margin bucket (descriptive, no scores)."""
    q = frame[frame["paper_flagged"]]
    table = q.groupby(["split", "dataset", "margin"]).agg(n=("idx", "size"), errors=("error", "sum"))
    return table.reset_index()


# ------------------------------------------------------------------------------
# Part 2: frozen Step 4 auditor applied to the Eval paper queue (Table 2)
# ------------------------------------------------------------------------------
TABLE2_METHODS: List[str] = ["S_maj", "S_comb_calib", "S_gate"]


def wilson_ci(errors: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson 95% interval for a proportion errors/n. Preferred over the normal approximation
    because the buckets are small (10-100 questions) and risks are close to 0.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = errors / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_frozen_thresholds(calib_path: str, alpha: str = "0.05") -> Dict[str, float]:
    """Thresholds tau_alpha exactly as stored by Step 4 (selected on Dev only)."""
    with open(calib_path) as f:
        stored = json.load(f)
    return {m: float(v[alpha]) for m, v in stored["taus"].items()}


def score_eval_queue(
    frame: pd.DataFrame,
    calib_path: str,
    seed: int = 42,
    alpha: str = "0.05",
) -> pd.DataFrame:
    """
    Applies the frozen Step 4 auditor to the Eval questions of the paper queue.

    - The logistic model is refit on Dev with the same deterministic procedure as Step 4
      (liblinear is deterministic, so the weights are identical; this is checked against
      the stored weights).
    - Thresholds are read from Step 4's calibration file, not re-selected.
    - Nothing from the Eval split influences any parameter.

    Returns:
        The Eval paper-queue rows with one score column and one accept column per method.
    """
    with open(calib_path) as f:
        stored = json.load(f)
    dev = frame[frame["split"] == "dev"].reset_index(drop=True)
    model, calib, _ = calibrate_on_dev(dev, dev["error"].to_numpy(), seed=seed)
    if not np.allclose(calib.logistic_weights, stored["logistic_weights"]):
        raise ValueError("Refit logistic weights differ from Step 4: check the input table and seed.")

    taus = load_frozen_thresholds(calib_path, alpha)
    queue = frame[(frame["split"] == "eval") & frame["paper_flagged"]].reset_index(drop=True)
    scores = compute_scores(blind_view(queue), calib_model=model)

    out = queue.copy()
    for m in TABLE2_METHODS:
        s = scores[m]
        out[f"score_{m}"] = s
        out[f"accept_{m}"] = np.isfinite(s) & (s >= taus[m])
    return out


def queue_calibrated_thresholds(frame: pd.DataFrame, seed: int = 42, alpha: float = 0.05) -> Dict[str, float]:
    """
    Sensitivity analysis: thresholds chosen for the paper queue itself.

    Step 4's tau_0.05 controls risk over the WHOLE benchmark, where unreviewed `consensus`
    questions (no observable errors) dilute the risk. Inside the paper queue errors are
    concentrated, so that tau overshoots 5%. Here tau is re-selected to reach risk <= alpha
    among accepted QUEUE questions, using only Dev-queue questions and Step 4's cross-fitted
    (out-of-fold) Dev scores. Eval is not touched.
    """
    dev = frame[frame["split"] == "dev"].reset_index(drop=True)
    _, _, dev_scores = calibrate_on_dev(dev, dev["error"].to_numpy(), seed=seed)
    in_queue = dev["paper_flagged"].to_numpy()
    errors = dev["error"].to_numpy()[in_queue]
    return {m: calibrate_tau(dev_scores[m][in_queue], errors, alpha) for m in TABLE2_METHODS}


def per_dataset_queue_thresholds(
    frame: pd.DataFrame, datasets: List[str], seed: int = 42, alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Same as `queue_calibrated_thresholds`, but one threshold per benchmark (Dev queue of that
    benchmark only, out-of-fold scores). Small Dev queues (~34 questions for GSM8K / SVAMP)
    make these thresholds noisy; reported as a sensitivity analysis, not a headline result.

    Returns:
        {dataset: {method: tau}}
    """
    dev = frame[frame["split"] == "dev"].reset_index(drop=True)
    _, _, dev_scores = calibrate_on_dev(dev, dev["error"].to_numpy(), seed=seed)
    errors = dev["error"].to_numpy()
    out: Dict[str, Dict[str, float]] = {}
    for d in datasets:
        mask = (dev["paper_flagged"] & (dev["dataset"] == d)).to_numpy()
        out[d] = {m: calibrate_tau(dev_scores[m][mask], errors[mask], alpha) for m in TABLE2_METHODS}
    return out


def apply_per_dataset_thresholds(scored: pd.DataFrame, taus: Dict[str, Dict[str, float]], suffix: str) -> pd.DataFrame:
    """Accept columns where each question uses the threshold of its own benchmark."""
    out = scored.copy()
    for m in TABLE2_METHODS:
        s = out[f"score_{m}"].to_numpy()
        tau = out["dataset"].map(lambda d: taus[d][m]).to_numpy(dtype=float)
        out[f"accept_{m}{suffix}"] = np.isfinite(s) & (s >= tau)
    return out


def apply_thresholds(scored: pd.DataFrame, taus: Dict[str, float], suffix: str) -> pd.DataFrame:
    """Adds accept columns `accept_<method><suffix>` for an alternative set of thresholds."""
    out = scored.copy()
    for m in TABLE2_METHODS:
        s = out[f"score_{m}"].to_numpy()
        out[f"accept_{m}{suffix}"] = np.isfinite(s) & (s >= taus[m])
    return out


def table2(
    scored: pd.DataFrame,
    group_col: str = "margin",
    order: Optional[List[str]] = None,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Table 2: for every method and group (margin bucket or dataset), the queue size, how many
    questions the auditor accepts / defers, how many accepted questions are curation errors
    (false accepts), the empirical risk among accepted with a Wilson CI, and the share of the
    paper's queue removed (accepted / queue size).
    """
    rows = []
    groups = order if order is not None else sorted(scored[group_col].unique())
    for m in TABLE2_METHODS:
        acc = f"accept_{m}{suffix}"
        for g in groups + ["Total"]:
            sub = scored if g == "Total" else scored[scored[group_col] == g]
            n = len(sub)
            if n == 0:
                continue
            a = int(sub[acc].sum())
            fa = int(sub.loc[sub[acc], "error"].sum())
            lo, hi = wilson_ci(fa, a)
            rows.append({
                "method": m, group_col: g, "queue_n": n, "accepted": a, "deferred": n - a,
                "false_accepts": fa,
                "risk_among_accepted": fa / a if a else float("nan"),
                "risk_ci_low": lo, "risk_ci_high": hi,
                "queue_reduction": a / n,
                "errors_in_group": int(sub["error"].sum()),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Part 3: qualitative error taxonomy (Table 3)
# ------------------------------------------------------------------------------
# Curated examples (one per operational error regime of the proposal, Section 6.3).
# Selected from the paper queue by rule, then reviewed by hand; root causes written by the team.
TABLE3_EXAMPLES: List[Dict[str, str]] = [
    {"regime": "4.1 Isolated model error", "dataset": "gsm8k", "idx": "5",
     "root_cause": "One model mis-applied the discount; 23/24 agree with the correct label. "
                   "The paper's rule sent it to review; the auditor accepts it correctly."},
    {"regime": "4.2 Mislabelled legacy label", "dataset": "drop", "idx": "148",
     "root_cause": "All 24 models answer 'rob bironas'; the legacy label only accepted 'bironas'. "
                   "The models expose a brittle label. The blind auditor accepts (correct); "
                   "the safety gate defers by design, so no label is overturned silently."},
    {"regime": "4.3 Correlated model error", "dataset": "gsm8k", "idx": "288",
     "root_cause": "21/24 models and the legacy label divide total weight (6000/2000 = 3) and ignore "
                   "that whole flagstones must fit per truck (26 per truck -> 4 trucks). "
                   "Shared shortcut: invisible to any consensus-based auditor."},
    {"regime": "4.4 Shared ambiguity", "dataset": "gsm8k", "idx": "146",
     "root_cause": "'3 times more pieces than 500' reads as 1500 or 2000. 23/24 models pick the "
                   "label's reading; the single dissenter (2625) takes the other one. "
                   "Rejected by the paper as ambiguous, but consensus looks confident."},
]


def _votes(parsed_predictions_json: str, top: int = 3) -> str:
    """Top answers with their vote counts, e.g. '64 x23, 56 x1'."""
    counts = Counter(json.loads(parsed_predictions_json).values()).most_common(top)
    return ", ".join(f"{a} ×{n}" for a, n in counts)


def _load_questions(datasets: List[str], cache_path: str) -> Dict[str, str]:
    """
    Question texts for the curated examples: from Hugging Face when available, otherwise
    from a small cached JSON (offline environments). Missing texts are left empty.
    """
    texts: Dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            texts.update(json.load(f))
    try:
        from datasets import load_dataset
        for ex in TABLE3_EXAMPLES:
            key = f"{ex['dataset']}|{ex['idx']}"
            if key not in texts and ex["dataset"] in datasets:
                row = load_dataset("madrylab/platinum-bench-paper-version", ex["dataset"], split="test")[int(ex["idx"])]
                texts[key] = row.get("question", "")
    except Exception:
        pass
    return texts


def table3(
    frame: pd.DataFrame,
    calib_path: str,
    datasets: List[str],
    questions_path: str,
    seed: int = 42,
) -> str:
    """
    Builds Table 3 (markdown). Scores come from the frozen Step 4 model; they are shown for
    illustration only (an example may come from the Dev split, which is flagged), so Table 3
    never feeds any metric.
    """
    dev = frame[frame["split"] == "dev"].reset_index(drop=True)
    model, _, _ = calibrate_on_dev(dev, dev["error"].to_numpy(), seed=seed)
    taus = load_frozen_thresholds(calib_path)
    texts = _load_questions(datasets, questions_path)

    lines = [
        "# Table 3: Qualitative Error Taxonomy (Paper Review Queue)",
        "",
        "One curated example per operational error regime. Scores use the frozen Step 4 auditor "
        f"(tau_0.05: S_comb_calib = {taus['S_comb_calib']:.3f}, S_gate = {taus['S_gate']:.3f}).",
        "",
        "| Regime | Question | Votes (top 3) | Legacy label | Platinum target | Status | S_comb_calib | Blind decision | Gated decision | Root cause |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ex in TABLE3_EXAMPLES:
        row = frame[(frame["dataset"] == ex["dataset"]) & (frame["idx"].astype(str) == ex["idx"])]
        if row.empty:
            continue
        scores = compute_scores(blind_view(row.reset_index(drop=True)), calib_model=model)
        s_calib, s_gate = float(scores["S_comb_calib"][0]), float(scores["S_gate"][0])
        r = row.iloc[0]
        blind = "ACCEPT" if np.isfinite(s_calib) and s_calib >= taus["S_comb_calib"] else "DEFER"
        gated = "ACCEPT" if np.isfinite(s_gate) and s_gate >= taus["S_gate"] else "DEFER"
        question = texts.get(f"{ex['dataset']}|{ex['idx']}", "").replace("|", "/")
        where = f"{ex['dataset'].upper()} #{ex['idx']} ({r['split']})"
        lines.append(
            f"| {ex['regime']} | {where}: {question} | {_votes(r['parsed_predictions_json'])} | "
            f"{', '.join(json.loads(r['legacy_target_json']))} | {', '.join(json.loads(r['label_platinum_target_json']))} | "
            f"{r['label_cleaning_status']} | {s_calib:.3f} | {blind} | {gated} | {ex['root_cause']} |"
        )
    return "\n".join(lines) + "\n"


def run(
    input_path: str,
    datasets: List[str],
    split_seed: int = 42,
    calib_path: str = "outputs/tables/step4_calibration.json",
    table2_path: str = "outputs/tables/table2_queue_reduction.csv",
    table3_path: str = "outputs/tables/table3_error_taxonomy.md",
    questions_path: str = "outputs/tables/step5_example_questions.json",
) -> pd.DataFrame:
    """Step 5 end to end (Tables 2 and 3): queue -> frozen auditor on Eval queue -> margin and dataset tables."""
    frame = build_queue_frame(input_path, datasets, split_seed)
    scored = score_eval_queue(frame, calib_path, seed=split_seed)
    queue_taus = queue_calibrated_thresholds(frame, seed=split_seed)
    scored = apply_thresholds(scored, queue_taus, "_queue")
    ds_taus = per_dataset_queue_thresholds(frame, datasets, seed=split_seed)
    scored = apply_per_dataset_thresholds(scored, ds_taus, "_queue_ds")
    step4_taus = load_frozen_thresholds(calib_path)

    parts = []
    for tau_source, suffix, breakdowns in [
        ("step4_tau_0.05", "", [("margin", MARGIN_ORDER), ("dataset", datasets)]),
        ("queue_tau_0.05", "_queue", [("margin", MARGIN_ORDER), ("dataset", datasets)]),
        ("queue_tau_0.05_per_dataset", "_queue_ds", [("dataset", datasets)]),
    ]:
        for col, order in breakdowns:
            t = table2(scored, col, order, suffix).rename(columns={col: "group"})
            t.insert(0, "breakdown", col)
            t.insert(0, "threshold", tau_source)
            if suffix == "_queue_ds":
                t["tau"] = [ds_taus[g][m] if g in ds_taus else np.nan for g, m in zip(t["group"], t["method"])]
            else:
                t["tau"] = t["method"].map(step4_taus if suffix == "" else queue_taus)
            parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(table2_path)), exist_ok=True)
    out.to_csv(table2_path, index=False)
    with open(table3_path, "w") as f:
        f.write(table3(frame, calib_path, datasets, questions_path, seed=split_seed))
    scored.to_csv(os.path.join(os.path.dirname(os.path.abspath(table2_path)), "step5_scored_eval_queue.csv"), index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: paper review-queue triage (Layer 2).")
    parser.add_argument("--input", default="outputs/tables/eval_table_core.csv")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "svamp", "drop"])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--calib", default="outputs/tables/step4_calibration.json")
    parser.add_argument("--table2", default="outputs/tables/table2_queue_reduction.csv")
    parser.add_argument("--table3", default="outputs/tables/table3_error_taxonomy.md")
    args = parser.parse_args()
    out = run(args.input, args.datasets, args.split_seed, args.calib, args.table2, args.table3)
    cols = ["threshold", "breakdown", "method", "group", "queue_n", "accepted", "deferred", "false_accepts",
            "risk_among_accepted", "risk_ci_low", "risk_ci_high", "queue_reduction", "tau"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(out[cols].to_string(index=False))
    print(f"\n[Done] Table 2 -> {args.table2}\n[Done] Table 3 -> {args.table3}")


if __name__ == "__main__":
    main()