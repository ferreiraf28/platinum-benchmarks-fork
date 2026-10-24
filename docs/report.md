# Findings & Implementation Takeaways

Centralized log of empirical findings, dataset stats, and conclusions across all implementation steps.

---

## Step 1: Data Infrastructure & Cache Inventory

### Dataset Anatomy & Schema
Each benchmark pairs Hugging Face metadata with a local 24-model raw output cache:
- **Core fields:** `question`, `platinum_prompt` (with CoT), `platinum_prompt_no_cot` (for reasoning models), `original_target` (legacy), `platinum_target` (curated, hidden from auditor), `cleaning_status`, and `platinum_parsing_strategy` (`math` vs `text`).
- **Cleaning statuses & empirical distribution:**
  - `consensus` (no human review needed): GSM8K 78.0% (234), SVAMP 80.3% (241), DROP 12.4% (31).
  - `verified` (divergence occurred, but legacy target was correct; model errors): GSM8K 13.0% (39), SVAMP 9.7% (29), DROP 0.8% (2).
  - `revised` (legacy target was wrong/incomplete; models revealed ground truth): GSM8K 0.3% (1), SVAMP 1.0% (3), DROP 70.4% (176).
  - `rejected` (ambiguous/broken prompt; discarded): GSM8K 8.7% (26), SVAMP 9.0% (27), DROP 16.4% (41).

### Cache Inventory Findings
Audit conducted via [inventory.py](/src/inventory.py) across all 24 frontier models:
- **Cache Coverage:** Exactly 100.0% coverage across all 24 models on all valid questions (GSM8K: 274/274, SVAMP: 273/273, DROP: 209/209; $\bar{K} = 24.0$).
- **Parser Reliability:** Standard regex parsers extracted answers successfully in >99.97% of outputs (only 3 total edge parsing failures out of 18,144 generation strings).
- **Footprint:** All 15 paper benchmark caches take under 100 MB total.

### Offline Loader Design
Implemented in [offline_loader.py](/src/offline_loader.py):
- Reconstructs canonical 4-tuple cache keys: `(prompt, temperature, run_id=0, model_name)`.
- Strictly offline: binary pickle lookup only, returning `MISSING_IN_CACHE` and flagging `incomplete_cache = True` to force human review if a key is absent.
- Verified by unit tests in [test_offline_loader.py](/tests/test_offline_loader.py) (7/7 passed).

### Key Empirical Takeaways from Cache Inspection
1. **Unnecessary Human Review on Isolated Formatting Slips (GSM8K #2):**
   - Question asks for profit after house flip.
   - 21 / 24 models (87.5%) agreed on `70000`. Only `gpt-4o` dropped trailing zeros (`70`) and `o1-mini` made an arithmetic error.
   - The original paper heuristic flagged this for costly human review (`status: verified`). Our selective auditor can safely auto-accept high-consensus questions, directly reducing manual audit costs.
2. **Models Smarter than Legacy Labels (DROP #0):**
   - Question: *"How many yards was the second longest touchdown pass?"* Legacy target was integer `['25']`.
   - 18 / 24 models (75.0%) answered `25 yards`.
   - Legacy label penalized correct descriptive models. Human review had to expand the target to `['25', '25 yards', '25-yard']` (`status: revised`). Demonstrates how model consensus exposes brittle legacy ground-truth labels.

---

## Step 2: Prediction Parsers & Uncertainty Signals
- *(Conclusions on signal distributions, pairwise correlations, and CoT TF-IDF similarity to be recorded upon completion)*

---

## Step 3: Selective Auditor Engine & Baselines
- *(Conclusions on threshold calibration and risk-coverage tradeoffs to be recorded upon completion)*

---

## Step 4: Layer 1 Evaluation (Risk-Coverage & AURC)
- *(Empirical coverage, selective risk, and Table 1 results on Eval Split to be recorded upon completion)*

---

## Step 5: Layer 2 Evaluation (Paper Queue Triage & Failure Modes)
- *(Review queue reduction rates at R <= 0.05 and failure diagnoses to be recorded upon completion)*

---

## Step 6: Arm 2 (DROP Stress Test & Temporal Robustness)
- *(Consensus vulnerability and perturbation results to be recorded upon completion)*

---

## Step 7: Cross-Benchmark Generalization
- *(Cross-dataset transfer and final audit conclusions to be recorded upon completion)*
