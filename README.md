# CanopyIntel: Forest Loss Prediction & Deforestation Hotspot Prioritization System

CanopyIntel is an enterprise-grade environmental intelligence system engineered to predict subnational forest loss rates, classify deforestation severity, project multi-year canopy trajectories (2025–2030), and prioritize high-risk ecological intervention zones.

Built following the rigorous specifications of `PRD.md`, CanopyIntel enforces strict **zero-leakage temporal cross-validation**, residual target modeling, physical mass-balance conservation, and an editorial **Deep Dark Forest** visual design system.

---

## 🌲 Key Features

- **Residual Log-Ratio Learning:** Beats the strong persistence baseline ($r=0.925$) on unseen 2023–2024 holdout test data (Test MAE **0.0932** vs Baseline **0.0969**).
- **Physical Conservation of Mass:** Guarantees strict mass balance ($\text{Area}_{t+1} = \text{Area}_t - \text{Loss}_t + \text{Gain}_t$) across recursive 2025–2030 forecasts.
- **Zero Leakage:** Temporal holdout ($\le 2021$ train, 2022 val, 2023–2024 test), 6-fold expanding-window CV, and 5-fold spatial GroupKFold.
- **Explainable AI (XAI):** Global TreeSHAP summary, beeswarm, feature interaction analysis, and local single-region waterfall attributions.
- **Editorial Streamlit Dashboard:** 8 comprehensive pages styled with custom Google Fonts (*Fraunces*, *IBM Plex Sans*, *IBM Plex Mono*), a 10x5 schematic tile cartogram, real-time What-If policy simulator, and one-click Executive Brief HTML generator.

---

## 📊 Benchmark Results (Holdout Test 2023–2024)

| Model | Target Form | Test MAE (%) | 95% Bootstrap CI | Test RMSE | Test $R^2$ | Test sMAPE | Test MASE | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline ($y_{t+1}=y_t$)** | Direct | $0.0969$ | $[0.0843, 0.1105]$ | $0.1221$ | $0.9452$ | $12.74\%$ | $1.0000$ | Baseline |
| **Region Mean Baseline** | Direct | $0.4481$ | $[0.3920, 0.5110]$ | $0.5620$ | $-0.1624$ | $48.21\%$ | $4.6243$ | Baseline |
| **Ridge Regression** | Log-Ratio | $0.0984$ | $[0.0851, 0.1122]$ | $0.1248$ | $0.9427$ | $12.91\%$ | $1.0154$ | Candidate |
| **LightGBM Regressor** | Log-Ratio | $0.0954$ | $[0.0821, 0.1089]$ | $0.1205$ | $0.9466$ | $12.52\%$ | $0.9845$ | Candidate |
| **ExtraTrees Regressor** | Log-Ratio | $0.0948$ | $[0.0815, 0.1080]$ | $0.1192$ | $0.9478$ | $12.44\%$ | $0.9783$ | Candidate |
| **RandomForest Regressor (Tuned)** | **Log-Ratio** | **0.0932** | **[0.0800, 0.1064]** | **0.1176** | **0.9491** | **12.28%** | **0.9618** | **Production Winner** |
| **NNLS Weighted Blend** | Log-Ratio | $0.0934$ | $[0.0802, 0.1067]$ | $0.1179$ | $0.9488$ | $12.31\%$ | $0.9639$ | Evaluated (Pruned) |

> **Parsimony Decision:** The weighted blend did not beat the best single model (Random Forest) by $\ge 0.001$ with $95\%$ CI strictly below 0. Per PRD Section 7.7, the simpler single model (Random Forest) was shipped.

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
# Clone and create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
Run the full automated pipeline (ingestion, leakage checks, feature engineering, Optuna tuning, ensemble benchmarking, SHAP computation, recursive forecasting, and hotspot ranking):
```bash
# Fast mode (~4 minutes, 15 trials)
python -m src.train --fast

# Full exhaustive mode (60-100 trials)
python -m src.train
```

### 3. Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run Test Suite
```bash
python -m pytest -q
```
*Current test suite: 22 passed across data validation, temporal leakage, feature integrity, multi-year mass balance, and headless Streamlit page tests.*

---

## 🗂️ Project Architecture

```
├── app/                             # Editorial Streamlit Web Application
│   ├── components.py                # Reusable KPI cards, cartogram, Plotly theme
│   ├── style.css                    # Deep Dark Forest theme styling & typography
│   ├── streamlit_app.py             # Landing portal & architecture hub
│   └── pages/                       # Multi-page analytical views
│       ├── 1_Overview.py            # Executive summary & net forest balance
│       ├── 2_Data_Explorer.py       # Data distributions, correlations, audit
│       ├── 3_Hotspot_Analysis.py    # Ranked table, 10x5 cartogram, heatmap
│       ├── 4_Model_Performance.py   # Benchmark leaderboard & LaTeX methodology
│       ├── 5_Forecast.py            # 2025–2030 multi-scenario trajectory engine
│       ├── 6_What_If_Simulator.py   # Real-time policy & environmental simulation
│       ├── 7_Explainability.py      # TreeSHAP beeswarm, waterfall, permutations
│       └── 8_Executive_Brief.py     # Standalone HTML export generator
├── data/                            # Raw and processed datasets
│   └── forest_loss_trend.csv        # 50 subnational regions (2005–2024)
├── models/                          # Serialized trained model artifacts (.joblib)
├── reports/                         # Computed outputs, tables, and figures
│   ├── figures/                     # SHAP and EDA visual assets
│   ├── experiment_log.csv           # Iterative run tracking
│   ├── forecast_2025_2030.csv       # Multi-year recursive scenario projections
│   ├── hotspots.csv                 # Jurisdictional composite risk rankings
│   ├── metrics.json                 # Complete quantitative evaluation metrics
│   └── technical_report.md          # 14-section formal engineering report
├── src/                             # Core Python ML package
│   ├── config.py                    # Schema, paths, and hyperparameters
│   ├── data_loader.py               # 11 automated integrity & mass-balance checks
│   ├── ensemble.py                  # Stacking, NNLS blending, bootstrap CI test
│   ├── evaluate.py                  # Multi-metric regression & classification suite
│   ├── explain.py                   # TreeSHAP and permutation importance engine
│   ├── features.py                  # 73 causal lag, rolling, and slope features
│   ├── forecast.py                  # Autoregressive multi-year rollout engine
│   ├── hotspots.py                  # Multi-factor composite risk prioritization
│   ├── models.py                    # Model zoo factory & persistence baselines
│   ├── splits.py                    # Expanding-window, holdout & group CV
│   ├── targets.py                   # Residual log-ratio & tertile targets
│   ├── train.py                     # Master execution orchestrator
│   └── tuning.py                    # Optuna hyperparameter optimization
├── tests/                           # Pytest integration & leakage suite
├── DECISIONS.md                     # Engineering & architectural decision log
├── Makefile                         # Unified CLI task runner
└── PRD.md                           # Product requirements and evaluation rules
```

---

## 🎨 Dashboard Pages Overview

1. **Overview & Executive Summary:** Headline computed insight, KPI strip with sparklines, annual net balance bar chart, and schematic risk cartogram.
2. **Data Explorer:** Full searchable data table with column configuration, feature distributions by protected status, interactive correlation heatmap, and multi-region time-series comparisons.
3. **Hotspot Analysis:** 10x5 schematic cartogram, composite risk rankings, region $\times$ year risk heatmap with policy enactments marked, and regional risk archetypes.
4. **Model Performance:** Leaderboards with pinned persistence baseline, bootstrap confidence interval whiskers, confusion matrices, and formal LaTeX mathematical formulations.
5. **Multi-Year Forecast:** 2025–2030 recursive autoregressive trajectories across 5 scenarios, prediction intervals, and historical backtest verification.
6. **What-If Simulator:** Real-time parameter sliders (fire count, agricultural expansion, rainfall, policy toggle) with instant cached model inference, before-and-after delta, and severity probability shifts.
7. **Explainability & Attribution:** Global TreeSHAP summary beeswarm, permutation importance ranking, and local single-region waterfall case studies.
8. **Executive Brief:** Dynamic, publication-grade standalone HTML briefing generator with one-click download.

---

## ⚖️ License & Integrity
Seed 42 is enforced across all splits, models, and simulations. Zero data leakage is mathematically enforced and tested.

