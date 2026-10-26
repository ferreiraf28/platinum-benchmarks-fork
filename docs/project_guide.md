# Engineering Playbook & Step-by-Step Implementation Guide
## Selective Human Review & Post-Hoc Auditing for Scalable LLM Benchmark Curation

> **Course:** Advanced Topics in Machine Learning (TAAC / M.IA009) — MSc in Artificial Intelligence, Faculty of Engineering / Faculty of Sciences, University of Porto  
> **Base Paper:** *Do Large Language Model Benchmarks Test Reliability?* (Vendrow, Vendrow, Beery & Mądry; MIT / arXiv:2502.03461, 2025)  
> **Repository:** `platinum-benchmarks-fork`  
> **Document Status:** Operational Living Guide & Engineering Protocol (Companion to [docs/project_proposal.md](file:///Users/franciscoferreira/Library/CloudStorage/GoogleDrive-klferreira28@gmail.com/My%20Drive/MSc%20AI/TAAC/Project/platinum-benchmarks-fork/docs/project_proposal.md))

---

## 1. Purpose and Philosophy of this Guide

This document is the **operational execution guide** for implementing the research project specified in [docs/project_proposal.md](file:///Users/franciscoferreira/Library/CloudStorage/GoogleDrive-klferreira28@gmail.com/My%20Drive/MSc%20AI/TAAC/Project/platinum-benchmarks-fork/docs/project_proposal.md). 

While the proposal details the academic motivation, literature review, theoretical questions (RQ1–RQ4), and hypotheses (H1–H4), this guide provides:
1. **Incremental, Step-by-Step Roadmap:** The implementation is decomposed into 7 distinct, manageable steps.
2. **Deep Technical & Didactic Explanations:** For every module, algorithm, and equation, the rationale is documented so the team understands every design choice.
3. **Verification & Test-Driven Validation:** Every step defines explicit sanity checks, validation tests, and "Definition of Done" (DoD) criteria before proceeding to the next stage.
4. **Zero-Cost / Anti-API Protocol:** Strict offline execution guarantees that prevent unexpected API charges or reliance on active third-party endpoints.
5. **Living Log & Decision Record:** A tracked record of technical hurdles, empirical findings, and design pivots encountered during the project.

---

## 2. System Architecture & Hermetic Isolation

The project comprises two complementary investigative arms:

```
+---------------------------------------------------------------------------------------------------+
|                                      RESEARCH ARCHITECTURE                                        |
+---------------------------------------------------------------------------------------------------+
|  ARM 1 (Core Contribution - 100% Offline):                                                        |
|  Post-Hoc Selective Auditor for Benchmark Curation                                                |
|  - Operates strictly on precomputed model outputs from official cache (24 LLMs)                   |
|  - Extracts multi-signal uncertainty (Majority, Shannon Entropy, TF-IDF CoT similarity, Length)  |
|  - Calibrates continuous confidence score S(x) in [0, 1] on Dev Split                             |
|  - Makes binary triage decision: ACCEPT_PROVISIONAL vs. DEFER_TO_HUMAN                            |
|  - Evaluates Risk-Coverage curves (Layer 1) & Human Review Queue Reduction (Layer 2)              |
+---------------------------------------------------------------------------------------------------+
|  ARM 2 (Secondary Experiment - Robustness Analysis):                                              |
|  Controlled Stress Test on Consensus Failures in DROP                                             |
|  - Targets instances with 100% unanimous agreement and S(x) >= 0.95                              |
|  - Compares Temporal Perturbation T_temp vs. Placebo Invariant T_placebo                          |
|  - Measures Consensus-Break Rate to detect shared first-event biases                              |
+---------------------------------------------------------------------------------------------------+
```

### The Hermetic Separation Protocol (Preventing Label Leakage)

To maintain absolute scientific validity, the code is strictly separated into two isolated layers:

```
[Cache .pkl + Prompts] 
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. OPERATIONAL LAYER (Blind Auditor)                        │
│ - Inputs: Prompts, Raw Model Responses, Model IDs           │
│ - FORBIDDEN: platinum_target, cleaning_status, is_correct   │
│ - Outputs: Uncertainty Signals, Score S(x), Decision D(x)   │
└─────────────────────────────────────────────────────────────┘
          │ Decisions D(x) & S(x)
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. EVALUATION LAYER (Retrospective Meta-Evaluation)         │
│ - Inputs: Decisions D(x), Targets (platinum_target)         │
│ - Metrics: Empirical Selective Risk R(tau), Coverage C(tau),│
│            AURC, Queue Reduction, Error Regime Diagnostics  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Roadmap (Steps Overview)

| Step | Focus Area | Key Deliverables | Validation Target | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Step 1** | Environment & Offline Loader | Cache downloaded; `src/offline_loader.py`; `src/inventory.py` | 0 API calls; complete cache inventory report | `[x]` |
| **Step 2** | Prediction Parsers & Uncertainty Signals | `src/signals.py` (Majority, Entropy, TF-IDF CoT, Length, Hedging) | Unit tests pass; all signals bounded in $[0, 1]$ | `[x]` |
| **Step 3** | Selective Auditor Engine & Baselines | `src/auditor.py` ($S_{\text{maj}}, S_{\text{ent}}, S_{\text{comb}}, S_{\text{gate}}$) | Decision policy verified; mandatory deferral on incomplete cache | `[x]` |
| **Step 4** | Dev/Eval Split Protocol & Layer 1 Evaluation | `src/evaluation.py`; threshold $\tau$ calibration; RC Curves & AURC | **Table 1** generated for core benchmarks (`gsm8k`, `svamp`, `drop`) | `[ ]` |
| **Step 5** | Layer 2: Paper Queue Triage & Error Taxonomy | Proxy `paper_flagged`; margin analysis; error regimes | **Table 2** (queue reduction) & **Table 3** (qualitative taxonomy) | `[ ]` |
| **Step 6** | Arm 2: Controlled Stress Test in DROP | `src/stress_test.py`; 20 DROP instances; $\mathcal{T}_{\text{temp}}$ vs $\mathcal{T}_{\text{placebo}}$ | Consensus-break rate computed on local model / test harness | `[ ]` |
| **Step 7** | Full Dataset Extension & End-to-End Notebook | `notebook.ipynb`; all 14 text datasets; artifact exports | Clean "Run All Cells" in virgin environment | `[ ]` |

---

## 4. Detailed Step-by-Step Execution Plan

### Step 1: Environment Setup & Anti-API Offline Data Infrastructure

#### 1.1 Objective & Scientific Rationale
Vendrow et al. (2025) released cached inference outputs for 24 leading LLMs across 15 benchmarks. However, the original `src/run_benchmark.py` script attempts to instantiate live API clients (OpenAI, Anthropic, Google) if a cache key is not matched or formatted incorrectly.
In Step 1, we build a bulletproof, network-isolated data loader (`OfflineDataLoader`) that reconstructs official cache keys deterministically and loads `.pkl` files without importing or touching external API SDKs.

#### 1.2 Specifications & Files
- **Files to Create/Modify:**
  - `src/offline_loader.py`: The pure Python class to load and index binary `.pkl` caches.
  - `src/inventory.py`: Script to generate descriptive statistics of cache coverage.
  - `scripts/download_paper_cache.sh`: Validate download execution.
- **Cache Key Signature:**
  $$\text{Key: } (prompt, temperature, run\_id, model\_name) \implies response\_text$$
  Where:
  - `prompt`: Constructed via `src/utils.py::get_prompt(example, model_name)`.
  - `temperature`: Determined via `src/models.py::ModelEngineFactory.get_temperature(model_name)` (0.5 for standard models, 1.0 for reasoning models like `o1`/`o3`).
  - `run_id`: `0` for all paper runs.
- **Incomplete Cache Policy:**
  If an example does not have answers for all target models ($K_i < K_{\text{target}}$), flag `incomplete_cache = True`. It must **never** be silently treated as a consensus.

#### 1.3 Step-by-Step Action Items
- [x] 1. Verify virtual environment and install dependencies (`datasets`, `huggingface_hub`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`).
- [x] 2. Run `bash scripts/download_paper_cache.sh` to populate `./cache/` with `.pkl` files.
- [x] 3. Implement `src/offline_loader.py`:
  - [x] Strict binary loading (`open(..., 'rb')`).
  - [x] Canonical key generation matching `src/utils.py` and `src/models.py`.
  - [x] No API client imports (`openai`, `anthropic`, `google.genai`).
  - [x] Clear error and status tracking (`FOUND`, `MISSING_IN_CACHE`).
- [x] 4. Implement `src/inventory.py`:
  - [x] Load Hugging Face dataset `madrylab/platinum-bench-paper-version`.
  - [x] Iterate through primary benchmarks: `gsm8k`, `svamp`, `drop`.
  - [x] Produce coverage matrix: total questions, valid questions (`cleaning_status != 'rejected'`), models available per question ($K_i$).
- [x] 5. Run inventory and save report to `outputs/tables/cache_inventory.csv`.

#### 1.4 Verification Tests
```bash
# Test 1: Ensure no internet connection is required to load cache
python3 -c "from src.offline_loader import OfflineDataLoader; print('Loader successfully imported with zero API dependencies')"

# Test 2: Run inventory script on primary datasets
python3 src/inventory.py --datasets gsm8k svamp drop
```
*Expected Output:* An inventory table showing >95% cache coverage across the 24 models without any network requests or exceptions.

#### 1.5 Definition of Done (DoD)
- `./cache/` contains verified `.pkl` files.
- `src/offline_loader.py` can retrieve raw responses for arbitrary (example, model) pairs.
- Cache coverage report saved and reviewed by the team.

---

### Step 2: Canonical Prediction Parsing & Uncertainty Signals Extraction

#### 2.1 Objective & Scientific Rationale
Raw model outputs contain reasoning text, introductory remarks, and final answers. To audit consensus without looking at ground truth, we must:
1. Parse the explicit prediction $\hat{y}_m$ using the benchmark's canonical parsing strategy (`math`, `text`, `multiple_choice`, `bbh_multiple_choice`, `squad`).
2. Extract the preceding reasoning chain (Chain-of-Thought / explanation).
3. Compute 5 mathematically rigorous signals of observable uncertainty.

#### 2.2 Mathematical Signal Definitions & Calibrated Scaling Protocol
For each question $x$ evaluated across $K$ available models:
1. **Majority Ratio $M(x) \in [1/K, 1.0]$:**
   $$M(x) = \frac{\max_{c} N_c(x)}{K}, \quad N_c(x) = \sum_{k=1}^K \mathbb{I}(\hat{y}_k = c)$$
2. **Normalized Vote Confidence $V(x) \in [0.0, 1.0]$:**
   $$H(x) = - \sum_{c \in \mathcal{C}(x)} p(c|x) \log_2 p(c|x), \quad p(c|x) = \frac{N_c(x)}{K}$$
   $$V(x) = 1 - \frac{H(x)}{\log_2(K)}$$
3. **Textual Semantic Coherence $Sim(x) \in [0.0, 1.0]$ (TF-IDF Cosine Similarity):**
   Extract reasoning bodies $R_1, \dots, R_K$. Compute pairwise cosine similarities over dynamically fitted TF-IDF vectors $\mathbf{v}_k$:
   $$Sim_{\text{raw}}(x) = \frac{2}{K(K-1)} \sum_{1 \le i < j \le K} \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$
4. **Length Stability $LS(x) \in [0.0, 1.0]$ (Deliberation Length Dispersion):**
   Compute coefficient of variation of character lengths: $CV_L(x) = \frac{\sigma_L(x)}{\mu_L(x) + \epsilon}$, $LS_{\text{raw}}(x) = \exp(-CV_L(x))$.
5. **Hedging Score $Hed(x) \in [0.0, 1.0]$:**
   Fraction of hedging terms (`perhaps`, `unclear`, `ambiguous`, `assume`, etc.) normalized across models:
   $$Hed(x) = 1.0 - \min\left(1.0, \frac{\sum_{k=1}^K \text{count}(w \in \mathcal{H} \text{ in } R_k)}{K \cdot \theta}\right)$$

#### 2.3 Calibrated Scaling Protocol for $Sim(x)$ and $LS(x)$
- **Empirical Rationale:** Because the 24 frontier models come from fundamentally distinct model families (OpenAI, Anthropic, Google, Meta, Mistral, xAI, DeepSeek, Qwen) with differing prompt wrappers, systemic verbosity, and stylistic conventions, raw $Sim(x)$ naturally peaks at $\sim 0.60$ and raw $LS(x)$ at $\sim 0.56$ even when all models agree perfectly on the reasoning steps. Unscaled, both signals float around $0.35 - 0.45$.
- **The Downward Drag Problem:** If unscaled $Sim(x)$ and $LS(x)$ are fed directly into linear composites like $S_{\text{comb}}^{\text{baseline}} = 0.40 M + 0.35 V + 0.15 Sim + 0.10 LS$, they artificially suppress composite confidence (capping $S(x)$ at $\approx 0.85$ even under unanimous agreement). This distorts threshold calibration and biases ablation comparisons against calibrated models.
- **Methodologically Sound Solution:** We calibrate fixed, universal affine reference bounds strictly on the pooled Core Development Suite ($N=850$ across `gsm8k`, `svamp`, and `drop`):
  - **CoT Similarity Scaling:**
    $$Sim_{\text{scaled}}(x) = \text{clip}\left(\frac{Sim_{\text{raw}}(x) - F_{\text{sim}}}{C_{\text{sim}} - F_{\text{sim}}}, 0.0, 1.0\right), \quad F_{\text{sim}} = 0.15, \; C_{\text{sim}} = 0.60$$
  - **Length Stability Scaling:**
    $$LS_{\text{scaled}}(x) = \text{clip}\left(\frac{LS_{\text{raw}}(x) - F_{\text{LS}}}{C_{\text{LS}} - F_{\text{LS}}}, 0.0, 1.0\right), \quad F_{\text{LS}} = 0.10, \; C_{\text{LS}} = 0.60$$
- **Cross-Benchmark Integrity Guarantee:** These reference bounds are frozen and applied globally. They prevent data snooping and protect the purity of the 12 held-out benchmarks evaluated in Step 7, ensuring true out-of-distribution zero-shot validation. Both raw and scaled values are retained in signal bundles for full provenance.

#### 2.4 Step-by-Step Action Items
- [x] 1. Create `src/signals.py`.
- [x] 2. Integrate extraction parsers from `src/utils.py::get_parse_fn(strategy)`.
- [x] 3. Implement separate parsing handler for CoT extraction (extract `<cot>`, `<thinking>`, or text preceding `Answer:`).
- [x] 4. Implement vector calculation functions for $M(x)$, $V(x)$, $Sim(x)$, $LS(x)$, and $Hed(x)$.
- [x] 5. Write unit tests in `tests/test_signals.py` covering boundary cases:
  - Unanimous votes ($K$ identical predictions) $\implies M=1.0, V=1.0$.
  - Maximum dispersion (all $K$ predictions different) $\implies V=0.0$.
  - 50-50 binary tie $\implies M=0.5, V=1 - \frac{1}{\log_2(K)}$.
- [x] 6. Compute signals for a sample of 100 questions from `gsm8k` and inspect distribution.

#### 2.5 Verification Tests
```bash
# Run unit tests for signals
python3 -m unittest tests/test_signals.py
```
*Expected Output:* All assertion checks pass; signals are strictly normalized in $[0, 1]$.

#### 2.6 Definition of Done (DoD)
- `src/signals.py` executes in operational blind mode (no ground truth parameter allowed).
- All 5 signals computed deterministically with unit tests passing.

---

### Step 3: Selective Auditor Engine, Decision Rule & Baselines

#### 3.1 Objective & Scientific Rationale
The selective auditor transforms observable uncertainty signals into a unified confidence metric $S(x)$ and outputs a binary operational decision:
- $\text{ACCEPT\_PROVISIONAL}$: Sufficient observable agreement; keep in benchmark without immediate manual review.
- $\text{DEFER\_TO\_HUMAN}$: High uncertainty, model disagreement, parsing failure, or missing cache; dispatch to human inspection queue.

We implement 4 score formulations to isolate the contribution of each signal:
1. **Majority Baseline ($S_{\text{maj}}$):** $S(x) = M(x)$
2. **Entropy Baseline ($S_{\text{ent}}$):** $S(x) = V(x)$
3. **Transparent Linear Composite ($S_{\text{comb}}^{\text{baseline}}$):**
   $$S_{\text{comb}}^{\text{baseline}}(x) = 0.40 \cdot M(x) + 0.35 \cdot V(x) + 0.15 \cdot Sim(x) + 0.10 \cdot LS(x)$$
4. **Calibrated Logistic Composite ($S_{\text{comb}}^{\text{calib}}$):**
   $$S_{\text{comb}}^{\text{calib}}(x) = \sigma\left(\mathbf{w}^T \mathbf{s}(x) + b\right)$$
   (Weights fitted strictly on the Dev Split).
5. **Safety Gate Variant ($S_{\text{gate}}$):**
   If legacy label $y_{\text{legacy}}$ is available, triggers mandatory review if the ensemble modal consensus disagrees with $y_{\text{legacy}}$ (`gate_mode="majority"`, preventing single-model typos from blocking consensus), with an optional `strict_unanimous` mode to replicate the paper's original filter.

#### 3.2 Decision Policy
For threshold $\tau \in [0, 1]$:
$$D_\tau(x) = \begin{cases}
\text{ACCEPT\_PROVISIONAL}, & \text{if } S(x) \ge \tau \text{ and } \text{cache\_complete}(x) \text{ and no parsing error} \\
\text{DEFER\_TO\_HUMAN}, & \text{if } S(x) < \tau \text{ or } \neg \text{cache\_complete}(x) \text{ or parsing error}
\end{cases}$$

#### 3.3 Step-by-Step Action Items
- [x] 1. Create `src/auditor.py`.
- [x] 2. Implement class `SelectiveAuditor` supporting methods `score(signals, mode)` and `decide(score, tau, metadata)`.
- [x] 3. Implement `SafetyGateAuditor` wrapping the selective auditor with legacy label checks.
- [x] 4. Implement mandatory deferral rules for incomplete cache and parse errors.
- [x] 5. Write unit tests in `tests/test_auditor.py` to verify:
  - Cache miss always yields `DEFER_TO_HUMAN` regardless of score.
  - Threshold filtering acts monotonically on acceptance.
  - Safety gate flags legacy disagreements immediately.

#### 3.4 Verification Tests
```bash
python3 -m unittest tests/test_auditor.py
```
*Expected Output:* All decision rules, gates, and mandatory deferral checks pass.

#### 3.5 Definition of Done (DoD)
- `src/auditor.py` operational and tested against edge cases.

---

### Step 4: Calibration Protocol (Dev/Eval Split) & Layer 1 Global Evaluation

#### 4.1 Objective & Scientific Rationale
To measure how well $S(x)$ distinguishes correct from incorrect consensus without data snooping, we freeze a **50% Dev Split** and a **50% Eval Split** (stratified by dataset and vote margin, `seed=42`).
On Dev Split:
- Fit logistic regression weights for $S_{\text{comb}}^{\text{calib}}$.
- Calibrate operational thresholds: find $\tau_{0.05}$ ensuring selective risk $R \le 5\%$, and $\tau_{0.01}$ ensuring $R \le 1\%$.
On Eval Split (untouched until final testing):
- Measure Coverage $C(\tau)$, Selective Risk $R(\tau)$, Risk-Coverage (RC) Curve, and AURC.

#### 4.2 Mathematical Metrics (Layer 1)
- **Coverage $C(\tau)$:** Fraction of examples accepted automatically:
  $$C(\tau) = \frac{|\{x \in \mathcal{D}_{\text{eval}} : D_\tau(x) = \text{ACCEPT}\}|}{|\mathcal{D}_{\text{eval}}|}$$
- **Selective Risk $R(\tau)$:** Empirical error rate among accepted examples:
  $$R(\tau) = \frac{\sum_{x \in \mathcal{D}_{\text{eval}}} \mathbb{I}(D_\tau(x) = \text{ACCEPT}) \cdot \mathbb{I}(\hat{y}_{\text{maj}}(x) \neq y_{\text{target}}^*(x))}{\sum_{x \in \mathcal{D}_{\text{eval}}} \mathbb{I}(D_\tau(x) = \text{ACCEPT})}$$
- **AURC (Area Under Risk-Coverage Curve):**
  $$\text{AURC} = \int_0^1 R(C) \, dC$$
  (Lower is strictly better; reflects low risk maintained across higher coverage).

#### 4.3 Deliverable: Formal Table 1
Generate and format **Table 1** for core benchmarks (`gsm8k`, `svamp`, `drop`):
```text
+------------------------+-----------+---------------+---------------+-----------+
| Triage Method          | Coverage  | Risk (Error)  | AURC          | Review    |
|                        | (C)       | at C = 75%    | (Area under RC| Reduction |
+------------------------+-----------+---------------+---------------+-----------+
| Full Human Review      | 0.0%      | 0.0%          | 0.000         | 0.0%      |
| Unrestricted Majority  | 100.0%    | [Base Error]  | [Flat AURC]   | 100.0%    |
| Random Abstention      | 75.0%     | [Base Error]  | [Rand AURC]   | 25.0%     |
| S_maj (Majority Ratio) | 75.0%     | R_maj%        | AURC_maj      | 25.0%     |
| S_ent (Vote Entropy)   | 75.0%     | R_ent%        | AURC_ent      | 25.0%     |
| S_comb (Proposed)      | 75.0%     | R_comb%       | AURC_comb     | 25.0%     |
| S_gate (With Label-Gate| 75.0%     | R_gate%       | AURC_gate     | 25.0%     |
+------------------------+-----------+---------------+---------------+-----------+
```

#### 4.4 Step-by-Step Action Items
- [ ] 1. Create `src/evaluation.py`.
- [ ] 2. Implement stratified 50/50 split generator (`seed=42`).
- [ ] 3. Implement calibration routine on Dev Split to optimize weights and derive $\tau_{\text{target}}$.
- [ ] 4. Implement blind evaluator on Eval Split using `src/utils.py::check_prediction`.
- [ ] 5. Implement numerical trapezoidal integration for AURC.
- [ ] 6. Generate and save RC Curve plots comparing all baselines to `outputs/figures/rc_curves_core.png`.
- [ ] 7. Export Table 1 to `outputs/tables/table1_selective_performance.csv`.

#### 4.5 Verification Tests
```bash
python3 src/evaluation.py --datasets gsm8k svamp drop --split_seed 42
```
*Expected Output:* Risk-Coverage curves monotonically decreasing as threshold increases; Table 1 printed and saved.

#### 4.6 Definition of Done (DoD)
- Strict separation confirmed (no Eval Split samples used in threshold or weight selection).
- Table 1 generated and saved.
- Figures generated in `outputs/figures/`.

---

### Step 5: Layer 2 Evaluation: Paper Review Queue Triage & Error Taxonomy

#### 5.1 Objective & Scientific Rationale
In Vendrow et al. (2025), a question is sent to human review whenever at least one model diverges from the target:
$$\text{paper\_flagged}(x) = \text{True} \iff \exists m \in \mathcal{M} : \hat{y}_m(x) \neq y_{\text{target}}^*(x)$$
In Layer 2, we evaluate: *How much of this flagged queue can our selective auditor safely accept without manual review, under a controlled risk ceiling ($R \le 0.05$)?*
We also test **Hypothesis H3** via margin stratification ($[23-1], [22-2], [21-3], \dots$) and diagnose failure modes according to the 4 Operational Error Regimes.

#### 5.2 Deliverables: Table 2 & Table 3
- **Table 2 (Paper Queue Reduction across Vote Margins):**
```text
+-----------------------+-------------+-------------+---------------+----------------+
| Margin Subset         | Total Flagged| Automatically| Deferred to   | False Accepts  |
| (e.g., K = 24)        | Paper Queue | Accepted    | Human Review  | (Empirical Risk|
+-----------------------+-------------+-------------+---------------+----------------+
| Margin 23-1 (1 diverg)| N_23        | A_23        | D_23          | E_23 (R_23%)   |
| Margin 22-2 (2 diverg)| N_22        | A_22        | D_22          | E_22 (R_22%)   |
| Margin 21-3 (3 diverg)| N_21        | A_21        | D_21          | E_21 (R_21%)   |
| Mixed Margins (<= 20) | N_low       | A_low       | D_low         | E_low (R_low%) |
+-----------------------+-------------+-------------+---------------+----------------+
| Total Aggregated      | Total_Queue | Total_Accept| Total_Human   | Global_Risk%   |
+-----------------------+-------------+-------------+---------------+----------------+
```
- **Table 3 (Qualitative Error Taxonomy):**
  Sample instances representing:
  1. *Regime 4.1 (Isolated Model Error):* Single weak model failed; auditor accepts safely.
  2. *Regime 4.2 (Mislabelled Legacy Label):* Models converged on correct solution, correcting original dataset error!
  3. *Regime 4.3 (Correlated Error):* False consensus on incorrect answer; blind auditor blind spot.
  4. *Regime 4.4 (Shared Ambiguity / First-Event Bias):* Consensus driven by prompt heuristics; studied in Arm 2.

#### 5.3 Step-by-Step Action Items
- [ ] 1. Create `src/queue_analysis.py`.
- [ ] 2. Filter dataset to subset where `paper_flagged == True`.
- [ ] 3. Apply calibrated auditor ($S_{\text{comb}}^{\text{calib}} \ge \tau_{0.05}$) to this subset.
- [ ] 4. Stratify results into vote margin buckets: $[23-1], [22-2], [21-3], \le 20$.
- [ ] 5. Compute review queue reduction percentage:
  $$\text{Queue\_Reduction} = \frac{|\text{Accepted by Auditor}|}{|\mathcal{Q}_{\text{paper}}|}$$
- [ ] 6. Sample 4 illustrative cases for Table 3, documenting prompt, votes, target, and root cause.
- [ ] 7. Export Table 2 to `outputs/tables/table2_queue_reduction.csv` and Table 3 to `outputs/tables/table3_error_taxonomy.md`.

#### 5.4 Verification Tests
```bash
python3 src/queue_analysis.py --datasets gsm8k svamp drop
```
*Expected Output:* Clear margin breakdown; confirmation of whether $[23-1]$ margin has risk $\le 5\%$.

#### 5.5 Definition of Done (DoD)
- Table 2 and Table 3 generated and verified.
- Empirical test of Hypothesis H3 concluded.

---

### Step 6: Arm 2 — Controlled Stress Test on Consensus in DROP

#### 6.1 Objective & Scientific Rationale
Vendrow et al. observed that reading comprehension models on `DROP` exhibit a *first-event bias* (blindly selecting the earliest mentioned entity or date). If all models share this heuristic, they form a false unanimous consensus ($S(x) \approx 1.0$), creating false accepts (Regime 4.4).
Arm 2 tests this vulnerability by applying two controlled transformations to 20 unanimous consensus instances from DROP:
1. **Temporal Perturbation ($\mathcal{T}_{\text{temp}}$):** Reverses temporal logic (e.g. asking for the second event instead of the first), with a newly verified target $y_{\text{temp}}^*$.
2. **Placebo Invariant Transformation ($\mathcal{T}_{\text{placebo}}$):** Inserts neutral, factually invariant sentences; target $y_{\text{placebo}}^* = y_{\text{original}}^*$.

#### 6.2 Metric: Consensus-Break Rate
$$\text{Break\_Rate}(\mathcal{T}) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}(\mathcal{T}(x_i)) \neq y_{\mathcal{T}}^*(x_i))$$
If $\text{Break\_Rate}(\mathcal{T}_{\text{temp}}) \gg \text{Break\_Rate}(\mathcal{T}_{\text{placebo}})$, the initial consensus was driven by positional shortcuts rather than robust temporal reasoning.

#### 6.3 Step-by-Step Action Items
- [ ] 1. Create `src/stress_test.py`.
- [ ] 2. Filter `drop` dataset for candidate instances:
  - $M(x) = 1.0$ (unanimous agreement in official cache).
  - $S(x) \ge 0.95$.
  - Narrative contains $\ge 2$ verifiable temporal events.
- [ ] 3. Sample 20 target questions and export to `data/stress_test_drop_sample.json`.
- [ ] 4. Encode $\mathcal{T}_{\text{temp}}$ and $\mathcal{T}_{\text{placebo}}$ variants with manually verified new targets.
- [ ] 5. Execute inference using local lightweight open-weights model (`Llama-3.2-1B-Instruct` or `Qwen2.5-1.5B-Instruct` via `transformers` on CPU/MPS; fallback test harness if hardware restricted).
- [ ] 6. Compute and report break rates for both transformations.
- [ ] 7. Save results table to `outputs/tables/table_arm2_stress_test.csv`.

#### 6.4 Verification Tests
```bash
python3 src/stress_test.py --sample_file data/stress_test_drop_sample.json
```
*Expected Output:* Clear comparison of consensus-break rates between temporal perturbation and placebo controls.

#### 6.5 Definition of Done (DoD)
- 20 DROP questions sampled and perturbed with verified targets.
- Break rates computed and analyzed.
- Hypothesis H4 evaluated.

---

### Step 7: Full Dataset Extension, End-to-End Notebook & Submission Packaging

#### 7.1 Objective & Scientific Rationale
With all core algorithms verified on `gsm8k`, `svamp`, and `drop`, we scale the evaluation across the remaining 11 textual benchmarks (`singleop`, `singleq`, `multiarith`, `mmlu_math`, `bbh_logical_deduction_three_objects`, `bbh_object_counting`, `bbh_navigate`, `tab_fact`, `hotpotqa`, `squad`, `winograd_wsc`).
We consolidate the entire pipeline into a clean, reproducible Jupyter notebook (`notebook.ipynb`) meeting the academic requirements of TAAC.

#### 7.2 Directory Structure for Final Submission
```text
submission_group_XX/
├── notebook.ipynb               <- Fully runnable notebook with precomputed outputs
├── src/
│   ├── offline_loader.py        <- Zero-network cache loader
│   ├── signals.py               <- Observable uncertainty signals
│   ├── auditor.py               <- Selective auditor and decision policy
│   ├── evaluation.py            <- Layer 1 evaluation and RC curves
│   ├── queue_analysis.py        <- Layer 2 paper review queue reduction
│   ├── stress_test.py           <- Arm 2 DROP perturbation analysis
│   └── utils.py                 <- Official parsers and evaluation logic
├── outputs/
│   ├── tables/                  <- CSV exports of Tables 1, 2, 3 and cache inventory
│   └── figures/                 <- High-resolution PNGs of RC curves and margin distributions
├── requirements.txt             <- Frozen dependencies
└── README.md                    <- Instructions for reproduction
```

#### 7.3 Step-by-Step Action Items
- [ ] 1. Run full evaluation across all 14 textual datasets.
- [ ] 2. Compile aggregated cross-benchmark tables and figures.
- [ ] 3. Construct `notebook.ipynb` with clear narrative headings, explanations of equations, execution blocks, and rendered charts.
- [ ] 4. Perform fresh-environment verification ("Run All Cells" in a clean virtual environment).
- [ ] 5. Verify that no cells require API keys or active internet connection.
- [ ] 6. Generate presentation figures and summary data for the final slides.

#### 7.4 Verification Tests
```bash
# Test complete notebook execution in headless mode
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook_executed.ipynb
```
*Expected Output:* Notebook executes completely from cell 1 to end with exit code 0.

#### 7.5 Definition of Done (DoD)
- All 14 text benchmarks processed.
- `notebook.ipynb` runs cleanly end-to-end.
- Output artifacts organized and reproducible.

---

## 5. Course Milestones & Deliverables Schedule

| Milestone | Weight | Timing | Primary Focus & Deliverables |
| :--- | :---: | :---: | :--- |
| **Checkpoint 1** | 10% | End of Week 1 | Problem statement, literature context, cache download & `OfflineDataLoader` inventory demonstration. |
| **Checkpoint 2** | 25% | End of Week 2 | Uncertainty signals formulation, auditor implementation, preliminary Layer 1 RC curves on GSM8K, SVAMP, DROP. |
| **Final Submission & Defense** | 65% (50% Group + 15% Individual) | End of Week 4 | Executable `notebook.ipynb`, PDF presentation slides (max 40 pages with appendices), final defense. |

---

## 6. Execution Log & Technical Decisions

*(This section is updated dynamically throughout implementation to record decisions, bugs resolved, and empirical observations).*

### Entry 001 — Protocol Initialization (2026-10-24)
- **Action:** Created `docs/project_guide.md` in English following complete methodological specification from `docs/project_proposal.md`.
- **Key Decision:** Confirmed zero-API design: all operations rely strictly on local `.pkl` caches from `madrylab/platinum-bench-paper-cache`.
- **Next Immediate Action:** Begin Step 1: verify environment dependencies and download paper cache.

### Entry 002 — Step 1 Completed: Offline Infrastructure & Sanity Audit (2026-10-24)
- **Action:** Downloaded all 15 cache `.pkl` files (99.9 MB) via enhanced `scripts/download_paper_cache.sh`. Implemented `src/offline_loader.py` with strict zero-network execution and canonical 4-tuple key reconstruction.
- **Verification:** Created and executed unit tests in `tests/test_offline_loader.py` (7/7 tests passed in 0.014s). Ran `src/inventory.py` on core datasets (`gsm8k`, `svamp`, `drop`).
- **Empirical Findings:**
  - Cache coverage across all 24 paper models is 100.0% for all three core benchmarks (`gsm8k`: 274/274 valid examples have all 24 models; `svamp`: 273/273; `drop`: 209/209).
  - Parsing success rate is >= 99.97% across available responses (only 2 parse errors in `gsm8k`, 1 in `svamp`, 0 in `drop`).
  - Report saved to `outputs/tables/cache_inventory.csv`.
- **Next Immediate Action:** Begin Step 2: implement observable uncertainty signals extraction in `src/signals.py`.

### Entry 003 — Step 2 Completed: Uncertainty Signals Extraction & Empirical Audit (2026-10-25)
- **Action:** Implemented `src/signals.py` operating in strict blind mode (zero label leakage). Extracts parsed predictions, CoT reasoning chains, and computes all 5 signals: $M(x)$, $V(x)$, $Sim(x)$ (via local TF-IDF cosine similarity), $LS(x)$, and $Hed(x)$.
- **Verification:** Created `tests/test_signals.py` with 16 automated unit tests validating boundary cases (unanimous agreement, maximum dispersion, 50-50 ties, 23-1 splits, reasoning extraction). All 16 tests passed in 0.007s.
- **Empirical Audit (GSM8K Sample N=100 via `scripts/verify_step2_signals.py`):**
  - Confirmed $Sim(x)$ has moderate correlation with $M(x)$ ($r = 0.291$), proving it captures orthogonal reasoning alignment rather than being a redundant proxy for vote counts.
  - Demonstrated that behavioral signals directly flag defective/broken prompts (`status: rejected`): on rejected instances, Hedging drops from $0.953 \to 0.777$ and Length Stability drops from $0.446 \to 0.365$.
  - Data exported to `outputs/tables/signals_gsm8k_sample_100.csv` and documented in `docs/report.md`.
- **Next Immediate Action:** Begin Step 3: Implement Selective Auditor Engine and Baselines ($S_{\text{maj}}, S_{\text{ent}}, S_{\text{comb}}, S_{\text{gate}}$) in `src/auditor.py`.

