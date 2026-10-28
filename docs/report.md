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

## Step 2: Prediction Parsers & Uncertainty Signals

### Signal Architecture & Blind Extraction Protocol
Implemented in [src/signals.py](/src/signals.py) operating in strict blind mode (zero access to targets or curation status). Extracts 5 observable signals across $K=24$ models, verified by 16/16 unit tests in [tests/test_signals.py](/tests/test_signals.py):
1. **Majority Ratio $M(x) \in [1/K, 1.0]$:** Frequency of modal prediction: $M(x) = \max_c N_c / K$.
2. **Entropy Confidence $V(x) \in [0.0, 1.0]$:** Normalized Shannon entropy complement: $V(x) = 1 - H(x) / \log_2(K)$.
3. **CoT Similarity $Sim(x) \in [0.0, 1.0]$:** Pairwise TF-IDF cosine similarity, scaled: $Sim_{\text{scaled}} = \text{clip}\left(\frac{Sim_{\text{raw}} - 0.15}{0.45}, 0, 1\right)$.
4. **Length Stability $LS(x) \in [0.0, 1.0]$:** Exponential length variation: $LS_{\text{scaled}} = \text{clip}\left(\frac{\exp(-CV_L) - 0.10}{0.50}, 0, 1\right)$.
5. **Hedging Score $Hed(x) \in [0.0, 1.0]$:** Epistemic marker density (`perhaps`, `unclear`, `cannot determine`).

---

### Empirical Signal Distributions (Core Development Suite, $N=300$)
Audited via [scripts/verify_step2_signals.py](/scripts/verify_step2_signals.py) on 100-sample test splits across 24 models:

| Benchmark ($N=100$) | $M(x)$ | $V(x)$ | $Sim_{\text{scaled}}(x)$ | $Sim_{\text{raw}}(x)$ | $LS_{\text{scaled}}(x)$ | $LS_{\text{raw}}(x)$ | $Hed(x)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | $0.958 \pm 0.081$ | $0.958 \pm 0.066$ | $0.627 \pm 0.161$ | $0.432 \pm 0.072$ | $0.677 \pm 0.169$ | $0.439 \pm 0.085$ | $0.938 \pm 0.165$ |
| **SVAMP** | $0.956 \pm 0.069$ | $0.951 \pm 0.061$ | $0.621 \pm 0.149$ | $0.430 \pm 0.067$ | $0.588 \pm 0.204$ | $0.393 \pm 0.105$ | $0.875 \pm 0.235$ |
| **DROP** | $0.749 \pm 0.208$ | $0.778 \pm 0.180$ | $0.409 \pm 0.181$ | $0.334 \pm 0.082$ | $0.601 \pm 0.217$ | $0.400 \pm 0.111$ | $0.927 \pm 0.154$ |
| **Pooled ($N=300$)** | **$0.888 \pm 0.166$** | **$0.896 \pm 0.143$** | **$0.552 \pm 0.193$** | $0.399 \pm 0.088$ | **$0.622 \pm 0.201$** | $0.410 \pm 0.103$ | **$0.913 \pm 0.190$** |

*Dynamic range across pooled suite: $Sim_{\text{scaled}} \in [0.11, 1.00]$, $LS_{\text{scaled}} \in [0.00, 0.91]$, $M \in [0.12, 1.00]$, $V \in [0.13, 1.00]$, $Hed \in [0.00, 1.00]$.*

#### Pooled Pairwise Correlation Matrix ($N=300$)
| Signal Dimension | $M(x)$ | $V(x)$ | $Sim_{\text{scaled}}(x)$ | $LS_{\text{scaled}}(x)$ | $Hed(x)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Ratio $M(x)$** | $1.000$ | $0.971$ | $0.477$ | $0.202$ | $0.189$ |
| **Entropy Confidence $V(x)$** | $0.971$ | $1.000$ | $0.465$ | $0.228$ | $0.219$ |
| **CoT Similarity $Sim_{\text{scaled}}(x)$** | $0.477$ | $0.465$ | $1.000$ | $0.068$ | $0.049$ |
| **Length Stability $LS_{\text{scaled}}(x)$** | $0.202$ | $0.228$ | $0.068$ | $1.000$ | $0.763$ |
| **Hedging Score $Hed(x)$** | $0.189$ | $0.219$ | $0.049$ | $0.763$ | $1.000$ |

#### Stratified Means by Curation Status
| Benchmark | Cleaning Status ($N$) | $M(x)$ | $V(x)$ | $Sim_{\text{scaled}}(x)$ | $LS_{\text{scaled}}(x)$ | $Hed(x)$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | `consensus` (78) / `verified` (13) / `rejected` (9) | $0.984$ / $0.847$ / $0.845$ | $0.981$ / $0.858$ / $0.861$ | $0.647$ / $0.534$ / $0.553$ | $0.694$ / $0.645$ / $0.543$ | $0.953$ / $0.929$ / $0.777$ |
| **SVAMP** | `consensus` (80) / `verified` (10) / `rejected` (9) | $0.973$ / $0.870$ / $0.819$ | $0.967$ / $0.857$ / $0.860$ | $0.633$ / $0.561$ / $0.547$ | $0.612$ / $0.476$ / **$0.311$** | $0.916$ / $0.703$ / **$0.319$** |
| **DROP** | `consensus` (12) / `revised` (71) / `rejected` (16) | $1.000$ / $0.757$ / $0.517$ | $1.000$ / $0.795$ / $0.532$ | $0.370$ / **$0.451$** / $0.270$ | $0.732$ / $0.611$ / $0.453$ | $0.981$ / $0.951$ / $0.787$ |

---

### Core Findings & Signal Robustness Justifications

1. **$M(x)$ and $V(x)$ are Orthogonally Complementary (Modal Mass vs. Distribution Disorder):**
   - **Why Pure Majority $M(x)$ is Insufficient:** $M(x)$ only tracks the single highest vote share ($p_{\max}$) and is blind to the structure of the remaining $(1 - p_{\max})$ dissenting votes. For example, with 12 votes out of 24 ($M = 0.500$), $M(x)$ treats a 12-vs-12 deadlock between two rival answers identically to a 12-vs-1x12 split where 12 models suffered isolated random errors. Entropy adds vital diagnostic value by distinguishing the organized deadlock from high-disorder confusion ($V = 0.782$ vs. $V = 0.455$).
   - **Why Pure Entropy $V(x)$ is Insufficient (The Low-Cardinality Blind Spot):** $V(x) = 1 - H(x)/\log_2(K)$ measures distribution dispersion, but is mathematically capped when predictions collapse into few distinct answers. In a razor-thin 14-vs-10 tie, $V = 0.786$ (only 2 classes, $H \le 1.0$ bit). In an 18-vs-3-vs-3 supermajority, $V = 0.768$ (3 classes). Pure entropy alone mistakenly rates the dangerous 14-10 tie as safer than a 75% decisive consensus simply because 2 classes have lower theoretical entropy than 3.
   - **The Multi-Signal Synergy:** Neither signal is safe in isolation. $M(x)$ anchors the absolute volume of agreement to prevent binary ties from passing, while $V(x)$ penalizes high-disorder distribution scattering. Combining both in $S_{\text{comb}}$ ensures that an instance is only accepted when it has both high winning mass and low disagreement disorder.
2. **CoT Similarity $Sim(x)$ Guards Against Spurious Consensus & Exposes Flawed Targets:**
   - Moderate correlation with majority ($r = 0.477$) confirms non-redundancy.
   - *DROP Revelation:* $Sim_{\text{scaled}}$ is higher on `revised` ($0.451$) than `consensus` ($0.370$) because multi-step sports arithmetic forced models to quote identical play-by-play numbers. Models were rock-solid, revealing the legacy target was incomplete. On defective prompts (`rejected`), $Sim_{\text{scaled}}$ collapses to $0.270$.
3. **Length Stability $LS(x)$ Flags Prompt Ambiguity & Model Panic:**
   - Independent of reasoning similarity ($r = 0.068$) and majority ($r = 0.202$).
   - On SVAMP `rejected` items, $LS$ collapses from $0.612$ to **$0.311$**: broken prompts cause reasoning models to loop extensively while standard models output brief guesses, driving length dispersion ($CV_L$) up.
4. **Epistemic Hedging $Hed(x)$ Provides Direct Verbalized Calibration:**
   - On SVAMP `rejected` items, $Hed(x)$ crashes from $0.916$ to **$0.319$**. Models explicitly output verbal hesitation (*"assuming no contradiction"*, *"cannot determine"*) without requiring ground-truth access.
5. **Calibrated Scaling Strategy ($0.15 \to 0.60$):**
   - Accounts for the cross-architecture ceiling ($\sim 0.60$) and residual prompt-entity overlap floor ($\sim 0.15$).
   - Prevents artificial downward drag on linear baseline combinations ($S_{\text{comb}}^{\text{baseline}}$), while universal dev-suite calibration ($N=850$) protects the 12 held-out test benchmarks for leak-free zero-shot evaluation in Step 7.

---


## Step 3: Selective Auditor Engine & Baselines

### Engine Architecture & Decision Rules
Implemented in [src/auditor.py](/src/auditor.py) and validated by 17/17 unit tests in [tests/test_auditor.py](/tests/test_auditor.py) (40/40 repository-wide). Converts observable signals into continuous confidence $S(x)$ and outputs binary decisions $D_\tau(x) \in \{\text{ACCEPT\_PROVISIONAL}, \text{DEFER\_TO\_HUMAN}\}$:
- **Baseline 1 ($S_{\text{maj}}$):** $S(x) = M(x)$ (Majority ratio).
- **Baseline 2 ($S_{\text{ent}}$):** $S(x) = V(x)$ (Entropy confidence).
- **Baseline 3 ($S_{\text{comb}}^{\text{baseline}}$):** $0.40 M + 0.35 V + 0.15 Sim_{\text{scaled}} + 0.10 LS_{\text{scaled}}$ (Transparent linear composite).
- **Baseline 4 ($S_{\text{comb}}^{\text{calib}}$):** $\sigma(\mathbf{w}^T \mathbf{s}(x) + b)$ (Calibrated logistic probability interface `.fit_calibrated()`; parameter fitting and feature ablation reserved for Step 4 on the frozen Dev split).
- **Safety Gate Variant ($S_{\text{gate}}$):** External control-flow filter intercepting cases where modal consensus $\hat{y}_{\text{maj}}$ disagrees with legacy target $y_{\text{legacy}}$ prior to thresholding (`gate_mode="majority"` by default; `strict_unanimous` optional).
- **Mandatory Deferral Guards:** If cache quorum is incomplete ($K_i < K_{\text{target}}$) or parsing errors occur, `DEFER_TO_HUMAN` is enforced unconditionally, regardless of $S(x)$.

---

### Empirical Coverage Sweep across Thresholds and Modes ($N=300$ Pooled Suite)
Audited via [scripts/verify_step3_auditor.py](/scripts/verify_step3_auditor.py) across $\tau \in [0.70, 0.95]$:

| Threshold $\tau$ | $S_{\text{maj}}$ Coverage | $S_{\text{ent}}$ Coverage | $S_{\text{comb}}^{\text{baseline}}$ Coverage |
| :---: | :---: | :---: | :---: |
| **0.70** | $85.0\%$ | $89.3\%$ | $82.0\%$ |
| **0.80** | $78.3\%$ | $80.0\%$ | $71.0\%$ |
| **0.85** | $75.3\%$ | $75.0\%$ | $60.7\%$ |
| **0.90** | $70.3\%$ | $67.0\%$ | $25.7\%$ |
| **0.95** | $60.3\%$ | $40.0\%$ | $2.0\%$ |

#### Benchmark Breakdown at Illustrative Threshold $\tau = 0.80$ vs. Paper Heuristic
*Note on baseline reference and threshold selection:* The breakdown below uses operational midpoint $\tau = 0.80$ (yielding $71.0\%$ pooled blind coverage) to contrast the paper's heuristic with the auditor in both operational settings:
1. **Greenfield / Blind Curation ($S_{\text{comb}}^{\text{baseline}}$):** Triage performed in strict blind mode without any reference label (applicable when curating new datasets from scratch).
2. **Legacy Benchmark Curation ($S_{\text{gate}}$, Majority Gate):** Realistic benchmark curation where historical labels $y_{\text{legacy}}$ exist. The Safety Gate automatically defers instances to human inspection if modal consensus disagrees with the legacy target ($\hat{y}_{\text{maj}} \neq y_{\text{legacy}}$), ensuring no historical label is overturned without human review.

The optimal scoring formulation (baseline vs. calibrated) and the definitive operational threshold ($\tau_{0.05}$, targeting selective risk $R \le 0.05$) will be empirically selected on the Development split in the **following steps**.

| Benchmark | Total ($N$) | Paper Heuristic (24/24 Unanimous with $y_{\text{legacy}}$) | Blind Auditor $C(0.80)$ ($S_{\text{comb}}^{\text{baseline}}$, Greenfield) | Safety-Gated Auditor $C(0.80)$ ($S_{\text{gate}}$, Legacy Curation) | Operational Triage Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **GSM8K** | 100 | $81.0\%$ (19 flagged) | **$91.0\%$** (9 deferred) | **$91.0\%$** (9 deferred) | Both auditor modes safely absorb isolated 23–1 and 22–2 model typos (GSM8K #2) where paper triggered manual review. |
| **SVAMP** | 100 | $87.0\%$ (13 flagged) | **$93.0\%$** (7 deferred) | **$93.0\%$** (7 deferred) | Auto-accepts sound reasoning paths with high consensus, improving review throughput by $+6.0\%$ over the paper heuristic. |
| **DROP** | 100 | $13.0\%$ (87 flagged) | **$29.0\%$** (71 deferred) | **$17.0\%$** (83 deferred) | Blind auto-accepts unanimous model revisions ($29\%$); Safety Gate intercepts them ($17\%$), enforcing human review whenever $y_{\text{legacy}}$ is overturned. |
| **Pooled Suite** | 300 | $60.3\%$ (119 flagged) | **$71.0\%$** (87 deferred) | **$67.0\%$** (99 deferred) | Substantially outperforms paper throughput ($+10.7\%$ blind, $+6.7\%$ gated) by distinguishing real errors from minor model slips. |

---

### Key Takeaways
1. **Strict Monotonicity & Safety Interception:** Threshold filtering behaves strictly monotonically. All simulated cache dropouts and parsing errors were intercepted 100% of the time, guaranteeing zero silent passes on compromised data.
2. **Selective Domain Differentiation:** At $\tau = 0.80$, $S_{\text{comb}}^{\text{baseline}}$ accepts 91% of clean GSM8K math problems and 93% of SVAMP while automatically routing 71% (blind) / 83% (gated) of difficult DROP instances to human review, proving that multi-signal consensus naturally adapts triage intensity to benchmark difficulty.
3. **Engine Architectural Readiness:** The modular `.fit_calibrated()` interface is implemented and verified for supervised calibration. In accordance with strict experimental hygiene against data snooping, model parameter training and feature ablation are withheld from Step 3 and reserved for the frozen 50% Dev Split in Step 4.
4. **Realistic Dual-Mode Curation vs. Paper Heuristic:**
   - *Legacy Benchmark Curation ($S_{\text{gate}}$ with $y_{\text{legacy}}$):* In an existing benchmark cleaning pipeline, allowing an AI to silently overturn textbook answers without human verification is unacceptable. The Safety Gate solves this by checking modal consensus against $y_{\text{legacy}}$. On clean math (GSM8K & SVAMP), it matches blind coverage ($91\%$ and $93\%$), proving it safely ignores isolated typos (e.g. 21-vs-1 in GSM8K #2) while validating agreed-upon math labels. On DROP, where 70% of legacy labels were flawed, the gate catches the overturned labels and routes them to human inspection ($17\%$ coverage vs. $29\%$ blind), beating the paper's $13\%$ coverage while guaranteeing human sign-off on all label changes.
   - *Greenfield / Unlabeled Tasks ($S_{\text{comb}}$ without $y_{\text{legacy}}$):* When building new benchmarks where no legacy label exists, the paper's heuristic cannot operate. The blind auditor triangulates observable dynamics ($M, V, Sim, LS$) into continuous confidence, achieving $71.0\%$ throughput across the pooled suite purely from ensemble consensus.

---

### Next Steps (Roadmap to Final Evaluation)
- **Calibrate weights and pick the optimal threshold (Step 4 - Dev Split):** Fit logistic regression on the Dev split to learn data-driven signal weights, select the best scoring method (heuristic baseline vs. calibrated logistic regression), and determine the optimal confidence threshold to guarantee an error rate under 5% ($\tau_{0.05}$).
- **Evaluate on unseen data (Step 4 - Eval Split):** Benchmark all scoring methods out-of-sample on the untouched Eval split using Coverage, Selective Risk, and Risk-Coverage curves (AURC) to produce Table 1.
- **Measure human review savings and diagnose failures (Step 5):** Quantify how much manual review effort is saved compared to the paper's original triage heuristic, and analyze why the ensemble fails when it does (e.g., benchmark prompt ambiguity vs. shared model mistakes).
- **Test generalization across benchmarks (Step 7):** Verify whether the learned weights and optimal threshold transfer effectively to new, unseen benchmarks without retraining.

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
