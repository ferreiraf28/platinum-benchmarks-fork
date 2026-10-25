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

1. **$M(x)$ and $V(x)$ are Orthogonally Complementary (Modal Mass vs. Dissent Dispersion):**
   - *Proof:* In a 14-vs-10 tie, $M = 0.583$ and $V = 0.786$. In an 18-vs-3-vs-3 supermajority, $M = 0.750$ and $V = 0.768$. Pure entropy mistakenly ranks a razor-thin 14-10 tie as safer than a 75% decisive consensus. Conversely, pure majority cannot distinguish concentrated opposition (18-6) from fragmented noise (18-1-1-1-1-1-1). Both are mandatory to pin modal confidence while penalizing organized dissent.
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
