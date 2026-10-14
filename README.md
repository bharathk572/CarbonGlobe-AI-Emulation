# Physics-Guided Deep Sequence Emulation for Forest Growth Prediction (CarbonGlobe)
### IEEE BigData Cup 2026: AI Emulation Challenge · Champion Solution (Score: 0.072)

[![Competition](https://img.shields.io/badge/IEEE%20BigData%20Cup-2026-blue)](https://www.kaggle.com)
[![Score](https://img.shields.io/badge/Competition%20Score-0.072%20(Top%201%25)-brightgreen)](https://www.kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👨‍💻 Authors & Affiliation
* **Bharath K** ([bharath2410354@ssn.edu.in](mailto:bharath2410354@ssn.edu.in)) — *Lead Developer & Researcher*
* **Avantheka S, Darshan C L, Dharshan K**
* **Institution:** Department of Computer Science and Engineering, Sri Sivasubramaniya Nadar (SSN) College of Engineering, Chennai, Tamil Nadu, India

---

## 📄 Key Project Deliverables
* **Research Paper:** [`CARBON GLOBE CHALLENGE REPORT.pdf`](./CARBON%20GLOBE%20CHALLENGE%20REPORT.pdf)
* **Presentation Deck:** [`CarbonGlobe_Presentation.pptx`](./CarbonGlobe_Presentation.pptx)
* **Champion Clean Notebook:** [`notebook_champion_v72_0.072.ipynb`](./notebook_champion_v72_0.072.ipynb) (Runs in ~8 min on Kaggle GPU)
* **Champion 20-Model Ensemble Notebook:** [`notebook_champion_v81_0.072.ipynb`](./notebook_champion_v81_0.072.ipynb)
* **Official Kaggle Submission:** [`submission_champion_v72_0.072.csv`](./submission_champion_v72_0.072.csv)

---

## 📌 Problem Overview
The **IEEE BigData Cup 2026 AI Emulation Challenge** requires emulating the complex process-based Ecosystem Demography (ED / CarbonGlobe) model. The objective is to forecast global forest canopy **Height** and **Above-Ground Biomass (AGB)** across a **40-year climate horizon** under two future pathways:
* **SSP1-2.6:** Low-emissions sustainable mitigation pathway.
* **SSP5-8.5:** High-emissions extreme warming pathway.

### Evaluation Metric
$$\text{Competition Score} = 0.5 \times \left( \text{MSE}_{\text{height}} + \text{MSE}_{\text{agb}} \right)$$
Evaluated across **185,130 prediction rows** ($6,171 \text{ sites} \times 15 \text{ stand seed ages} \times 2 \text{ scenarios}$).

---

## 🏗️ System Architecture

```
[Raw Climate Sequences: 40 Yrs x 12 Mos x 136 Feats = 480 Steps]
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[Branch 1: Trend Encoder]        [Branch 2: Seasonal Shock]
(Dilated Conv1D + BiGRU, 408 ch) (Pointwise 1D-CNN, 544 ch)
         │                                 │
         └────────────────┬────────────────┘
                          ▼
        [Stand Chronosequence Attention]
           (4-Head Self-Attention over 15 Ages)
                          │
                          ▼
            [EcoPrior Residual Engine]
            Y = EcoPrior + α · Δ_Neural
            (g(a) = log(a+1) Anchor)
                          │
                          ▼
         [Clamping & 5-Pass Mean Calibration]
            (Tundra Floor Lock: 43.8% Rows)
            (μ_H = 0.899911, μ_AGB = 0.923806)
                          │
                          ▼
             ★ Champion Forecast (0.072)
```

1. **Temporal Decomposition:** Decouples 40-year multi-decadal climate trends from acute monthly weather extremes.
2. **Stand Chronosequence Attention:** Multi-Head Attention captures structural stand competition and canopy closure across 15 developmental ages.
3. **EcoPrior Physics Residual Engine:** Predicts bounded climate perturbations relative to a natural logarithmic stand-growth curve $g(a) = \log(a+1)$, eliminating 40-year autoregressive drift.
4. **Physical Clamping & 5-Pass Alignment:** Strictly locks 43.8% tundra sites to non-forest physical floors and aligns global prediction means to empirical ground-truth targets.

---

## 📊 Empirical Score Progression (Ablation Results)

| Architecture / Configuration | Competition Score (MSE) | Key Innovation / Rationale |
| :--- | :---: | :--- |
| **Autoregressive DL Baseline (LSTM)** | `0.186` | Unconstrained recurrent rollout; compounding step-by-step drift. |
| **EcoPrior Baseline Only** | `0.092` | Pure logarithmic stand growth; proves physical anchoring is vital. |
| **Trend Encoder Only** | `0.086` | Captures 40-year multi-decadal climate trajectories. |
| **Seasonal Encoder Only** | `0.089` | Captures acute weather shocks and extremes. |
| **Dual Temporal Encoder** | `0.081` | Combines trend and shock representations. |
| **Single Neural Model** | `0.079` | Full network without multi-seed ensemble averaging. |
| **Without Attention** | `0.076` | Disables chronosequence cross-age attention. |
| **Standard GBDT Tree Wall** | `0.076` | 20-fold bagged LightGBM + CatBoost tree ensemble. |
| **Without Physics Clamping** | `0.075` | Disables tundra floor clamping and calibration. |
| **Champion Full Pipeline** | **`0.072`** | **Dual-branch BiGRU + Attention + Bagging + Calibration.** |

---

## 🚀 Quick Start & Reproduction

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn
```

### Running the Champion Model
Open [`notebook_champion_v72_0.072.ipynb`](./notebook_champion_v72_0.072.ipynb) in Jupyter or Kaggle:
```python
# Runs in ~8 minutes on a standard T4 / P100 GPU
# Generates submission_champion_v72_0.072.csv scoring 0.072 on Kaggle Leaderboard
```

---

## 📜 Citation
If you use this work or codebase, please cite our research paper:
```bibtex
@inproceedings{bharath2026carbonglobe,
  title={Physics-Guided Temporal Decomposition and Chronosequence Attention for Climate-Driven Forest Growth Prediction},
  author={Bharath, K. and Avantheka, S. and Darshan, C. L. and Dharshan, K.},
  booktitle={IEEE BigData Cup 2026: AI Emulation Challenge},
  year={2026}
}
```
