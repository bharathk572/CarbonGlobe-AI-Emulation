# IEEE BigData Cup 2026: AI Emulation Challenge
## Champion Solution Documentation & Technical Report (Score: 0.072)

---

### Executive Summary

* **Competition:** *IEEE BigData Cup 2026: AI Emulation Challenge*
* **Core Objective:** Emulate the complex, process-based Ecosystem Demography (ED / CarbonGlobe) model predicting global forest canopy **Height** and **Aboveground Biomass (AGB)** across a 40-year projection horizon under two distinct future climate pathways:
  * **SSP1-2.6:** Low-emissions mitigation pathway.
  * **SSP5-8.5:** High-emissions extreme warming pathway.
* **Evaluation Metric:**
  $$\text{Competition Score} = 0.5 \times \left( \text{MSE}_{\text{height}} + \text{MSE}_{\text{agb}} \right)$$
  Evaluated across $6,171 \text{ global sites} \times 15 \text{ stand seed ages} \times 2 \text{ scenarios} = \mathbf{185,130} \text{ prediction rows}$.
* **Final Champion Score:** **`0.072`** (Achieved by `notebookv72` / `notebookv81`).

---

### Leaderboard Evolution: How We Reached 0.072

| Phase | Milestone Version | Leaderboard Score | Key Architecture & Method |
| :--- | :--- | :--- | :--- |
| **Phase 1: Baselines** | `notebookv1` | **`0.186`** | Unconstrained autoregressive sequence model; compounded step-by-step drift over 40 years. |
| | `notebook1` (EcoPrior) | **`0.092`** | Pure domain prior (logarithmic stand growth without ML). Proved physical anchoring is mandatory. |
| **Phase 2: Tree Models** | `notebookv26` | **`0.076`** | 20-Fold Bagged GBDT (LightGBM + CatBoost) with stand headroom features. Hit the '0.076 Tree Wall'. |
| **Phase 3: Deep Sequence** | **`notebookv68`** | **`0.072`** | **Breakthrough to 0.072!** Dual-branch Conv1D + BiGRU + Stand Attention sequence network. |
| **Phase 4: Clean Champion** | **`notebookv72`** | **`0.072`** | Clean, self-contained 10-fold deep neural emulator with sweet-spot scaling (alpha=0.350). |
| **Phase 5: 20-Model Ensemble** | **`notebookv81`** | **`0.072`** | 2-Seed 10-Fold Deep Neural Ensemble (20 models) with out-of-fold parabolic calibration (alpha=0.420). |

---

### Technical Architecture of the Champion Solution

#### 1. Input Processing & Temporal Decomposition (DLinear / TimeXer Paradigm)
Raw inputs consist of 40 years x 12 months x 136 features = 480 monthly climate steps. We decompose this high-dimensional sequence into two orthogonal physical components:
1. **Branch 1 (40-Year Climate Trend Sequence):**
   * Computes annual averages across the 12 months.
   * Extracts cumulative 40-year multi-decadal trends and rate of change.
   * Dimension: (N, 40, 408).
2. **Branch 2 (Monthly Seasonal Shock Sequence):**
   * Computes monthly seasonal anomalies relative to annual baselines.
   * Extracts summer heat shocks (months 5-10), winter freeze shocks, and annual extremes (min/max).
   * Dimension: (N, 40, 544).

#### 2. Physical Chronosequence Anchors (EcoPrior)
* Standard forests follow well-established logarithmic growth trajectories across their developmental chronosequence ([1, 10, 20, ..., 500] years).
* For each site and seed age, the network predicts the **climate-driven residual delta** relative to the physical chronosequence target:
  $$\hat{Y}_{40} = Y_{\text{EcoPrior}}(t+40) + \alpha \cdot \Delta_{\text{Neural}}$$

#### 3. Deep Neural Backbone Architecture
* **Trend Encoder:** 2-layer 1D-CNN (with dilated receptive field) followed by a Bidirectional GRU (BiGRU) and global temporal pooling.
* **Seasonal Shock Encoder:** Pointwise 1D-CNN projecting acute weather extremes to latent shock vectors.
* **Stand Chronosequence Attention:** Multi-Head Self-Attention (d=128, 4 heads) across the 15 stand ages, capturing structural competition and canopy closure.
* **Multi-Task Checkpoint Heads:** Evaluates 8 intermediate temporal checkpoints (Years 5, 10, 15, ..., 40) across all 7 ecological state variables (`height`, `agb`, `soil`, `lai`, `gpp`, `npp`, `rh`) to prevent overfitting.

#### 4. Post-Processing & Exact Dual Calibration
1. **Strict Tundra / Barren Ground Protection:**
   * 43.8% of test rows (81,038 rows) correspond to non-forest sites (tundra / desert where initial Y0 <= 0).
   * These rows are strictly clamped to zero-variance physical ground truth: no neural noise is added.
2. **Exact 5-Pass Iterative Mean Alignment:**
   * Aligns the global test predictions to the exact empirical target distribution means:
     * **Height Mean:** 0.899911
     * **AGB Mean:** 0.923806
   * Clips strictly at physical floors:
     * Height >= -0.082607
     * AGB >= -0.118213

---

### Champion Deliverables Directory (`CHAMPION_BEST_0.072/`)

* `notebook_champion_v72_0.072.ipynb`: Cleanest, recommended Champion Notebook for faculty presentation (runs in ~8 min on GPU).
* `submission_champion_v72_0.072.csv`: Official Champion Submission File (0.072 on Kaggle).
* `CARBON GLOBE CHALLENGE REPORT.pdf` / `ML_RESEARCH_PAPER_FINAL_SUBMISSION.pdf`: Complete 4-page IEEE formatted research paper report ready for submission.
* `CarbonGlobe_Champion_Presentation.pptx` / `CarbonGlobe_ColorTheme.pptx`: 11-slide faculty-ready presentation with dark theme, updated architecture diagrams, and empirical results.
* `notebook_champion_v81_0.072.ipynb` & `submission_champion_v81_0.072.csv`: Alternative 20-model ensemble version.
* `archive_older_attempts/`: All other intermediate experimental iterations cleanly archived.
