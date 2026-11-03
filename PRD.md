# PRD: Forest Loss Prediction & Deforestation Hotspot System

> **Instruction to the AI code editor (Claude Code / OpenCode / Antigravity):**
> Read this entire document before writing any code. Build the project exactly as specified, in the phase order given in Section 18. Every number in Section 3 was measured from the real dataset, so treat it as ground truth. Where this PRD says MUST, it is mandatory. Where it says SHOULD, do it unless it is impossible. After each phase, run the acceptance checks in Section 19 before moving on.

---

## 1. Project Summary

| Item | Value |
|---|---|
| Goal | Analyze historical forest-cover data, predict future forest loss, and flag regions with significant forest degradation |
| Input | `forest_loss_trend.csv` (1,000 rows, 13 columns) |
| Judging | Competition judged mainly on **model accuracy**, plus data handling, preprocessing, model choice, ensembling, metrics (Accuracy, Macro-F1, Precision, Recall), technical approach and mathematics |
| Deliverables | Reproducible ML pipeline, trained models, evaluation report, 5-year forecast, Streamlit dashboard |
| Language / stack | Python 3.10+, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, Plotly, Streamlit |
| Dataset nature | **Synthetic.** Expect clean data, simple relationships, and some noise-only columns. |

### 1.1 Problem statement (from the competition)
Build a machine learning model to analyze historical forest-cover data and predict future forest loss based on environmental conditions, human activities, agricultural expansion, and regional development patterns. The solution should identify areas experiencing significant forest degradation and estimate future deforestation trends to support sustainable forest management.

### 1.2 Three tasks this system solves
1. **Task A: Regression.** Predict next-year forest loss (`loss_pct` and `loss_ha`) for each region.
2. **Task B: Classification.** Classify each region-year into a degradation severity class (Low / Medium / High) for next year. This feeds the Accuracy, Macro-F1, Precision and Recall metrics.
3. **Task C: Forecasting.** Project loss and remaining forest area for 2025 to 2030 per region, with uncertainty intervals, and rank regions by risk.

> **Why three tasks:** The competition text does not state an explicit target column, and the judging criteria list classification metrics (Accuracy, Macro-F1). Covering regression, classification and forecasting guarantees the solution is valid whichever target the judges have in mind. The target definition MUST be configurable in `config.yaml` (Section 5.4).

---

## 2. Direct Answers to the Product Owner's Questions

| Question | Answer |
|---|---|
| Does it need EDA? | **Yes, but short and targeted.** Synthetic data is clean, but EDA decides the split strategy, exposes leakage, and shows which features carry signal. It is also a judged part of the "technical approach". Keep it to roughly 12 plots plus one summary table. |
| Does it need one-hot encoding? | **Only for `region_id` in linear models.** `region_id` has 50 unique values. For tree models use native categorical support (LightGBM/CatBoost) or out-of-fold target encoding. `protected_area` and `policy_active` are already binary (0/1) and need no encoding. |
| Does it need feature engineering? | **Yes, and this is where most accuracy gain comes from.** Lag features, rolling means, trend slopes, policy timing and region-level aggregates (Section 6). Without lags the model ignores the strongest predictor (loss is 92% autocorrelated). |
| Does it need scaling? | Only for linear models and any neural baseline. Tree/boosting models do not need it. |
| Missing values / duplicates? | None found (0 missing, 0 duplicates). Still implement the checks so the pipeline is robust to a different test file. |
| What about NLP? | **NLP is not applicable.** The dataset has no text columns. Do not add any NLP model. In the report, state one sentence: "No free-text fields exist; the task is tabular panel/time-series prediction." |
| Single model or grouped models? | **Both.** Train several single models, compare them, then build a stacking ensemble and a weighted-average blend. Report both. Final submission uses whichever wins on the time-based validation. |
| What metrics? | Regression: MAE, RMSE, R², sMAPE. Classification: Accuracy, Macro-F1, Macro-Precision, Macro-Recall, Weighted-F1, MCC, Cohen's kappa, ROC-AUC (OvR), confusion matrix. Section 10. |

---

## 3. Dataset Specification (measured facts)

**File:** `data/forest_loss_trend.csv` | **Shape:** 1,000 rows x 13 columns | **Structure:** panel data, **50 regions (R01 to R50) x 20 years (2005 to 2024)**, exactly 20 rows per region, 50 rows per year. **Missing values:** 0. **Duplicate rows:** 0.

| Column | Type | Meaning | Notes |
|---|---|---|---|
| `region_id` | str | Region identifier (R01..R50) | Grouping key, not a numeric feature |
| `year` | int | 2005 to 2024 | Time index |
| `forest_area_ha` | int | Forest area at start of year (hectares) | Range 42,249 to 797,930 |
| `loss_ha` | int | Forest loss in the year (ha) | Range 262 to 19,118 |
| `gain_ha` | int | Forest regrowth/gain in the year (ha) | Range 24 to 2,280 |
| `loss_pct` | float | Loss as % of forest area | Range 0.05 to 2.86, mean 1.22 |
| `population_density` | float | People per km² | Grows steadily each year per region |
| `road_density_km_per_km2` | float | Road density | **Constant within each region** |
| `agri_expansion_pct` | float | Agricultural expansion % | Noisy year to year |
| `rainfall_mm` | int | Annual rainfall | Very weak signal |
| `fire_count` | int | Fires in the year | **Strongest environmental driver** |
| `protected_area` | int (0/1) | Region protected status | **Constant within each region**; 18% of rows = 1 |
| `policy_active` | int (0/1) | Conservation policy in force | 37.3% of rows = 1. In 39 of 50 regions it switches on once (a policy start year), 11 regions never change |

### 3.1 Critical findings the code MUST respect

1. **Identity leakage:** `loss_pct = loss_ha / forest_area_ha x 100` (max error 0.001). So `loss_ha`, `loss_pct` and `forest_area_ha` of the **same year** are deterministic functions of each other. Never use current-year `loss_ha` to predict current-year `loss_pct`.
2. **Mass balance:** `forest_area(t+1) = forest_area(t) - loss(t) + gain(t)` (mean error 0.31 ha, max 1 ha). Use this identity to convert predicted loss and gain into future forest area in the forecast (Section 9).
3. **Persistence is extremely strong:** lag-1 autocorrelation of `loss_pct` = **0.925**. A "next year = this year" baseline scores R² of about 0.948 and MAE of about 0.097 on 2022 to 2024. Quick checks showed that a plain Gradient Boosting model with lag features did **not** beat it. Therefore:
   - The persistence baseline MUST be reported as the number to beat.
   - Models MUST be tuned to predict the **change** from persistence (residual learning, Section 7.2), otherwise they will not beat the baseline.
4. **Static region traits:** `protected_area` and `road_density_km_per_km2` never change inside a region. They explain *between-region* differences (between-region std 0.53 vs within-region std 0.24 for `loss_pct`).
5. **Signal ranking (correlation with `loss_pct`):** `fire_count` +0.46, `protected_area` -0.44, `policy_active` -0.37, `year` -0.16, `road_density` +0.15, `agri_expansion_pct` +0.06, everything else (`rainfall_mm`, `population_density`, `gain_ha`) is near zero. The dataset is synthetic, so do not read causal meaning into these. Keep weak features in the baseline, then show via feature selection whether dropping them helps.
6. **Time trend:** mean loss falls from about 1.3% (2005 to 2011) to about 1.1% (2019 to 2024), consistent with policies switching on over time.
7. Mean `loss_pct` by protection: unprotected 1.33, protected 0.67. By policy: inactive 1.38, active 0.94.

---

## 4. Repository Structure

```
forest-loss-ml/
├── README.md
├── PRD.md
├── requirements.txt
├── config.yaml
├── Makefile                      # make train | make app | make test
├── data/
│   ├── forest_loss_trend.csv
│   └── processed/                # engineered feature tables (parquet)
├── src/
│   ├── __init__.py
│   ├── config.py                 # loads config.yaml
│   ├── data_loader.py            # load + validate + schema checks
│   ├── eda.py                    # EDA plots and summary stats
│   ├── features.py               # all feature engineering
│   ├── targets.py                # builds regression + classification targets
│   ├── splits.py                 # time-based and group-based CV
│   ├── models.py                 # model factory and wrappers
│   ├── tuning.py                 # Optuna search
│   ├── ensemble.py               # stacking and weighted blend
│   ├── evaluate.py               # all metrics and reports
│   ├── explain.py                # SHAP and permutation importance
│   ├── forecast.py               # recursive 2025 to 2030 forecast
│   ├── hotspots.py               # risk scoring and degradation flags
│   └── train.py                  # end-to-end CLI entrypoint
├── app/
│   ├── streamlit_app.py          # main entry
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Data_Explorer.py
│   │   ├── 3_Hotspot_Analysis.py
│   │   ├── 4_Model_Performance.py
│   │   ├── 5_Forecast.py
│   │   ├── 6_What_If_Simulator.py
│   │   └── 7_Explainability.py
│   ├── components.py             # reusable UI widgets
│   └── style.css
├── models/                       # saved .joblib artifacts
├── reports/
│   ├── figures/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── forecast_2025_2030.csv
│   ├── hotspots.csv
│   └── technical_report.md
├── notebooks/
│   └── 01_eda_and_experiments.ipynb
└── tests/
    ├── test_features.py
    ├── test_no_leakage.py
    ├── test_splits.py
    └── test_forecast.py
```

---

## 5. Targets and Task Definitions

### 5.1 Prediction framing
Every row in the training table represents **region r at year t** with features known at the end of year t, and the target describes **year t+1**.

- Rows for year 2024 have no t+1 target; they are used only for the forecast.
- Usable supervised rows: 950 (50 regions x 19 transitions), fewer after lag features drop the first years (see 6.6).

### 5.2 Task A: Regression targets
- **Primary:** `y_loss_pct_next` = `loss_pct` at t+1.
- **Secondary:** `y_loss_ha_next` = `loss_ha` at t+1. Obtain it as `y_loss_pct_next x forest_area_next / 100` or train directly. Report both and keep the better one.
- **Residual target (used internally):** `delta = y_loss_pct_next - loss_pct`. Final prediction = `loss_pct + predicted delta`.

### 5.3 Task B: Classification target (degradation severity)
Create `severity_next` from `y_loss_pct_next` with three classes:

| Class | Label | Rule (default, configurable) |
|---|---|---|
| 0 | Low | `< 0.96` (33rd percentile of training years) |
| 1 | Medium | `0.96 to 1.40` |
| 2 | High | `> 1.40` (67th percentile of training years) |

- Thresholds MUST be computed **on the training split only**, then applied to validation/test, to avoid leakage. Save them in `models/thresholds.json`.
- Also support an alternative **binary "significant degradation"** target: `loss_pct_next >= 1.5`, or top-quartile of the training distribution. Selected via `config.yaml: classification.mode: ["ternary", "binary"]`.
- Provide both a **direct classifier** (trained on classes) and a **regression-then-bin** classifier (predict `loss_pct`, then apply thresholds). Report both and choose the better on Macro-F1.

### 5.4 Configurable target (competition safety net)
`config.yaml` MUST expose:
```yaml
target:
  regression: loss_pct          # loss_pct | loss_ha
  horizon_years: 1              # also test 2 and 3
classification:
  mode: ternary                 # ternary | binary
  thresholds: auto              # auto = train quantiles, or give [0.96, 1.40]
```
If the judges' hidden test file contains a pre-defined label column, the pipeline MUST allow pointing to it with `target.custom_column`.

---

## 6. Feature Engineering Specification

All features are computed **per region, sorted by year**, using only information available at or before year t. Implement in `src/features.py` as pure functions with unit tests.

### 6.1 Lag features (the biggest single accuracy gain)
For `loss_pct`, `loss_ha`, `fire_count`, `agri_expansion_pct`, `rainfall_mm`, `gain_ha`: lags 1, 2, 3.

### 6.2 Rolling statistics
Rolling mean and rolling std over windows 3 and 5 for `loss_pct`, `fire_count`, `agri_expansion_pct`, `rainfall_mm`. Use `min_periods` so early years keep rows where possible.

### 6.3 Trend and change features
- `loss_pct_diff1` = `loss_pct(t) - loss_pct(t-1)`
- `loss_pct_diff2` (acceleration)
- `loss_pct_slope3`, `loss_pct_slope5`: slope of a linear fit over the last 3 or 5 years
- `fire_count_pct_change`, `pop_density_growth` (year over year growth rate)
- `forest_area_change_rate` = `(forest_area(t) - forest_area(t-1)) / forest_area(t-1)`
- `net_loss_ha` = `loss_ha - gain_ha`; `gain_to_loss_ratio` = `gain_ha / loss_ha`; `net_loss_pct` = `net_loss_ha / forest_area_ha x 100`

### 6.4 Human-pressure and interaction features
- `fire_x_agri` = `fire_count x agri_expansion_pct`
- `road_x_pop` = `road_density x population_density`
- `pressure_index` = z-score(`pop_density`) + z-score(`road_density`) + z-score(`agri_expansion_pct`) (z-scores computed on training data only)
- `fire_per_100k_ha` = `fire_count / forest_area_ha x 100000`
- `drought_flag`: `rainfall_mm` below the training 20th percentile; `rainfall_anomaly` = `rainfall_mm - region mean rainfall (training only)`
- `protected_x_policy` = `protected_area x policy_active`

### 6.5 Policy-timing features
- `years_since_policy`: years since `policy_active` first became 1 in that region (0 or -1 if never)
- `policy_ever_active`: region has any policy in history **up to year t** (never look into the future)
- `policy_just_started`: 1 in the first year of the policy

### 6.6 Region-level features
- Region baseline: `region_mean_loss_pct`, `region_std_loss_pct`, `region_mean_fire`, computed **using only years up to t** (expanding mean) for time-safe usage, or from the training period only.
- `forest_area_rank` and `size_bucket` (small / medium / large, by quantiles).
- Cumulative loss: `cum_loss_ha`, `cum_loss_share` = cumulative loss / the first-year forest area.
- `year_idx` = `year - 2005`, and `year_idx²` optionally.

### 6.7 Categorical handling of `region_id`
| Model family | Handling |
|---|---|
| Linear / Ridge / ElasticNet | One-hot encoding (50 columns, `drop_first=True`) |
| LightGBM / CatBoost | Native categorical feature |
| XGBoost / RandomForest / ExtraTrees | Out-of-fold target encoding with smoothing, or just rely on region-baseline numeric features and drop the raw ID |
- When evaluating on **unseen regions** (GroupKFold), `region_id` MUST be excluded because the model cannot know new regions. Use region-baseline features computed from history instead.

### 6.8 Feature selection
After building about 60 features, run: (a) correlation filter (drop |r| > 0.95 duplicates), (b) permutation importance on validation, (c) recursive feature elimination or SHAP-based pruning. Compare **full vs pruned** feature sets with the same CV and keep the best. Log the comparison in `reports/feature_selection.csv`.

### 6.9 Preprocessing pipeline rules
- Use `sklearn.pipeline.Pipeline` / `ColumnTransformer`. Fit transformers **only on the training fold**.
- Scaling (`StandardScaler`) only inside pipelines for linear models. Check skewness of `loss_ha`, `forest_area_ha`, `fire_count`; apply `log1p` if |skew| > 1 for linear models.
- Outliers: do **not** drop rows (synthetic data, limited size). Use Winsorizing at 1%/99% only inside the linear-model pipeline. Tree models are robust as-is.
- Data validation (`data_loader.py`): schema check, dtype check, ranges (`loss_pct` in [0, 100], flags in {0,1}), uniqueness of (`region_id`, `year`), identity check `loss_pct ≈ loss_ha/forest_area_ha`, and consecutive-year check per region. Fail loudly with clear messages.

---

## 7. Modeling Specification

### 7.1 Data splitting (critical for honest accuracy)
Random K-fold is **forbidden** here. It leaks future years into training and inflates scores.

| Split | Definition |
|---|---|
| **Primary: temporal holdout** | Train: target years ≤ 2021. Validation: 2022. Test: 2023 and 2024 targets. The test set is touched only once at the end. |
| **Model-selection CV** | Expanding-window time-series CV with at least 5 folds, e.g. train up to 2015, 2016, 2017, 2018, 2019 and validate on the next year or two. Implemented with a custom splitter grouped by year. |
| **Spatial generalization check** | `GroupKFold(n_splits=5)` by `region_id`: shows how the model behaves on regions never seen. Report separately. |
| **Final fit** | After choosing the model, refit on all data up to the latest available target year and use it for the 2025 to 2030 forecast. |

Also run a **final sanity check**: for each model, report score on the temporal holdout **and** on the spatial CV. A model good at both is trustworthy.

### 7.2 Residual / delta learning (to beat persistence)
Train each regressor on `delta = y_next - loss_pct_current`. Prediction = persistence + predicted delta. Compare three target formulations and keep the best by CV MAE:
1. Direct `y_next`
2. Delta `y_next - loss_pct`
3. Log-ratio `log(y_next / loss_pct)`

### 7.3 Model zoo (train all, compare all)

| # | Model | Role | Notes |
|---|---|---|---|
| 0 | **Persistence** (`y_next = loss_pct`) | Must-beat baseline | Also "region mean" and "linear trend per region" baselines |
| 1 | Ridge / ElasticNet | Linear baseline | One-hot region, scaled, shows interpretable coefficients |
| 2 | Random Forest | Bagging | Tune `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| 3 | Extra Trees | Bagging | Faster, lower variance |
| 4 | Gradient Boosting (sklearn) | Boosting baseline | Reference |
| 5 | **XGBoost** | Boosting | Tune `eta`, `max_depth`, `subsample`, `colsample_bytree`, `lambda`, `alpha`, `min_child_weight` |
| 6 | **LightGBM** | Boosting | Native categorical, `num_leaves`, `min_data_in_leaf`, `feature_fraction` |
| 7 | **CatBoost** | Boosting | Native categorical, strong on small data |
| 8 | SVR / kernel ridge (optional) | Non-tree diversity | Scaled features |
| 9 | Small MLP (optional) | Diversity | `sklearn.neural_network.MLPRegressor`, early stopping |
| 10 | **Stacking ensemble** | Meta-model | Ridge meta-learner over out-of-fold predictions of models 1 to 9 |
| 11 | **Weighted blend** | Ensemble | Weights found via non-negative least squares or Optuna on validation |
| 12 | Hybrid (classification) | Two-stage | Regression value binned to class, plus direct classifier |

Classification counterparts: Logistic Regression (multinomial), RandomForestClassifier, ExtraTreesClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier, plus a soft-voting and a stacking ensemble.

> **Do not use deep learning or sequence models (LSTM/Transformer) as the main approach.** With 1,000 rows, boosted trees plus good features are the right tool. An optional small LSTM or GRU baseline may be included for completeness only if time permits, and clearly labeled "experimental".

### 7.4 Hyperparameter tuning
- **Optuna** (TPE sampler), 60 to 100 trials per model, objective = mean CV MAE (regression) or mean CV Macro-F1 (classification), using the expanding-window splitter.
- Set `random_state=42` everywhere and use a Optuna seed.
- Save best params to `models/best_params.json`. Use early stopping for boosting models where supported.
- Keep the search space modest (small data overfits tuning noise). Use a nested check: final test is untouched during tuning.

### 7.5 Class imbalance
Tertile classes are balanced by construction. In binary mode, use `class_weight='balanced'` or `scale_pos_weight`. Optionally test SMOTE **inside** the training fold only. Report macro metrics so minority class performance is visible.

### 7.6 Probability calibration and uncertainty
- Calibrate classifier probabilities with isotonic or Platt scaling (`CalibratedClassifierCV`, time-aware CV). Report Brier score and a reliability curve.
- Regression prediction intervals: **quantile regression** (LightGBM `objective='quantile'`, alpha 0.1/0.9) **and/or split-conformal prediction** using validation residuals. Report empirical coverage of the 80% interval on the test set.

### 7.7 Ensembling rules
- Base models MUST be trained with the same temporal folds. Stacking uses out-of-fold predictions only.
- The ensemble is accepted as final **only if** it beats the best single model on temporal CV MAE and Macro-F1 by a non-trivial margin; otherwise choose the best single model (simplicity wins ties). Print a paired comparison (per-fold scores) and a paired bootstrap confidence interval of the difference.

---

## 8. Mathematics (include in `reports/technical_report.md` and in the Streamlit "Methodology" tab)

Write these out with LaTeX (`st.latex`) in the app.

**8.1 Problem formulation.** For region *r* and year *t*, features `x_{r,t}` ∈ ℝ^d. Learn `f` such that `ŷ_{r,t+1} = f(x_{r,t})`, minimizing expected loss over time-ordered data.

**8.2 Loss ratio identity.**
`loss_pct_{r,t} = 100 · L_{r,t} / A_{r,t}` and `A_{r,t+1} = A_{r,t} − L_{r,t} + G_{r,t}`, where *L* = loss, *G* = gain, *A* = forest area.

**8.3 Residual learning.** `ŷ_{t+1} = y_t + g(x_t)`, with `g` fit to `δ_t = y_{t+1} − y_t`. Explain: this removes the dominant persistence component so that the model concentrates on genuine change.

**8.4 Ridge / ElasticNet.**
`β̂ = argmin_β ‖y − Xβ‖² + λ[(1−α)/2 · ‖β‖₂² + α‖β‖₁]`

**8.5 Gradient boosting (generic).** Additive model `F_m(x) = F_{m−1}(x) + η · h_m(x)`, where `h_m` fits the negative gradient `r_{im} = −∂ℓ(y_i, F(x_i))/∂F(x_i)` at `F = F_{m−1}`.

**8.6 XGBoost regularized objective.**
`𝓛 = Σ_i ℓ(y_i, ŷ_i) + Σ_k Ω(f_k)`, `Ω(f) = γT + ½λ Σ_j w_j²`
Second-order Taylor expansion gives optimal leaf weight `w_j* = − G_j / (H_j + λ)` and split gain
`Gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ`.

**8.7 Random Forest.** `f̂(x) = (1/B) Σ_b T_b(x)`, with bootstrap samples and random feature subsets. Variance of the average: `ρσ² + (1−ρ)σ²/B`.

**8.8 CatBoost note.** Ordered boosting and ordered target statistics to reduce target leakage on categorical features.

**8.9 Stacking.** `ŷ = Σ_m w_m · f_m^{OOF}(x)`, with `w` from a Ridge meta-learner (or NNLS, `w_m ≥ 0`).

**8.10 Prediction intervals.**
- Quantile loss: `ρ_τ(u) = u(τ − 1{u<0})`.
- Split conformal: `q̂ = Quantile_{⌈(n+1)(1−α)⌉/n}(|y_i − ŷ_i|)`, interval `[ŷ − q̂, ŷ + q̂]`.

**8.11 Explainability.** Shapley value `φ_j = Σ_{S⊆F\{j}} |S|!(|F|−|S|−1)!/|F|! · [v(S∪{j}) − v(S)]`, computed with TreeSHAP.

**8.12 Metric formulas** are in Section 10.

**8.13 Risk score** (hotspot ranking), see Section 11.

---

## 9. Forecasting Specification (2025 to 2030)

Goal: estimate **future deforestation trends** per region.

### 9.1 Method: recursive multi-step forecasting with driver projection
For each region and each horizon step h = 1..6 (2025 to 2030):
1. **Project the exogenous drivers** for year t+h:
   - `population_density`: linear extrapolation of the region's own trend (it grows steadily in the data).
   - `road_density`, `protected_area`: held constant (they are static).
   - `policy_active`: hold at its last value in the **baseline** scenario.
   - `agri_expansion_pct`, `rainfall_mm`, `fire_count`: use the region's recent mean (last 5 years), or a damped trend. For the uncertainty band, sample from the historical distribution (Monte Carlo, 200 draws).
2. Predict `loss_pct(t+1)` with the final model.
3. Convert to hectares: `loss_ha = loss_pct x forest_area / 100`. Predict `gain_ha` with a simple second model (or use region's trailing mean gain-to-loss ratio).
4. Update forest area with the mass-balance identity: `A_{t+1} = A_t − L_t + G_t`.
5. Recompute lag and rolling features from the predicted values and repeat.

### 9.2 Scenarios (used by the Streamlit What-If page)
| Scenario | Definition |
|---|---|
| Baseline | Drivers continue as projected |
| Policy-on-everywhere | `policy_active = 1` for all regions from 2025 |
| Agri-surge | `agri_expansion_pct` +30% |
| High-fire | `fire_count` +25% |
| Conservation push | `policy_active = 1` and fire -20% |
Output the difference between scenarios and baseline in ha lost by 2030.

### 9.3 Forecast validation (backtest)
Run the same recursive procedure from 2019 and 2021 as the origin and compare against the known 2020 to 2024 values. Report MAE per horizon step (error grows with h; show a plot). Never claim forecast accuracy beyond what the backtest shows.

### 9.4 Output
`reports/forecast_2025_2030.csv` with columns: `region_id, year, scenario, loss_pct_pred, loss_pct_lo, loss_pct_hi, loss_ha_pred, forest_area_pred, severity_pred`.

---

## 10. Evaluation Metrics

### 10.1 Regression
- **MAE** = `(1/n) Σ|y − ŷ|` (primary)
- **RMSE** = `√((1/n) Σ(y − ŷ)²)`
- **R²** = `1 − Σ(y − ŷ)² / Σ(y − ȳ)²`
- **sMAPE** = `(100/n) Σ 2|y − ŷ| / (|y| + |ŷ|)`
- **MASE** relative to persistence (value < 1 means better than persistence)
- Also report per-region MAE, per-year MAE, and a residual plot.
- Report both on `loss_pct` and `loss_ha`.

### 10.2 Classification (explicitly requested by judges)
For K classes, with per-class TP, FP, FN:
- **Accuracy** = `Σ TP_k / N`
- **Precision_k** = `TP_k / (TP_k + FP_k)`, **Recall_k** = `TP_k / (TP_k + FN_k)`
- **F1_k** = `2 · P_k R_k / (P_k + R_k)`
- **Macro-Precision / Macro-Recall / Macro-F1** = unweighted mean over classes
- **Weighted-F1** (also reported), **Matthews Correlation Coefficient**, **Cohen's kappa** (quadratic weighted, because classes are ordinal)
- **ROC-AUC (one-vs-rest, macro)** and **PR-AUC**
- **Confusion matrix** (counts and row-normalized), full `classification_report`
- **Hotspot Precision@K**: of the top-K regions ranked by predicted risk, what fraction were truly High? (K=5, 10)
- **Adjacent-class accuracy**: prediction within ±1 class (relevant because severity is ordinal)

### 10.3 Reporting rules
- Every table shows: mean ± std over CV folds, plus the single temporal-holdout score.
- Output `reports/model_comparison.csv` (all models, all metrics), `reports/metrics.json` (final model), and figures in `reports/figures/`.
- Include a final **"Beat-the-baseline" table**: Persistence vs Best single vs Ensemble, with the gain in percent and a bootstrap 95% CI.
- With only 100 test rows (2 years x 50 regions), scores are noisy. Always show CIs and avoid over-claiming. If the judges use a different split, the ranking of models should still hold, which is why spatial CV is also reported.

---

## 11. Hotspot / Degradation Detection (`src/hotspots.py`)

Goal: "identify areas experiencing significant forest degradation".

**Region risk score (0 to 100)** combining, with weights configurable in `config.yaml`:
| Component | Default weight |
|---|---|
| Predicted next-year `loss_pct` (percentile rank) | 35% |
| Recent trend: slope of last 5 years of `loss_pct` (percentile rank) | 20% |
| Predicted probability of "High" severity | 20% |
| Fire pressure: `fire_count` percentile | 10% |
| Human pressure index percentile | 10% |
| Not protected and no policy bonus | 5% |

Also output:
- **Tier:** Critical (score ≥ 75), High (50 to 75), Moderate (25 to 50), Low (< 25)
- **Trend label:** Accelerating / Stable / Improving, from the sign and significance of the 5-year slope (Mann-Kendall test or OLS p-value < 0.1)
- **Change-point flag:** detect structural change around policy start (simple before/after mean comparison with a t-test per region, or `ruptures` library if allowed)
- **Optional unsupervised check:** KMeans or DBSCAN on region profiles (mean loss, fire, agri, road, pop) to cluster regions into 3 to 4 archetypes; show cluster profiles. Anomaly flag with Isolation Forest for unusual region-years.
- Output: `reports/hotspots.csv` with `region_id, risk_score, tier, trend_label, predicted_loss_pct_next, p_high, top_drivers` where `top_drivers` comes from SHAP (top 3 features per region).

---

## 12. Explainability (`src/explain.py`)
- Global: SHAP summary (beeswarm) and bar plot for the final tree model; permutation importance on the validation set; partial dependence for `fire_count`, `agri_expansion_pct`, `road_density`, `years_since_policy`.
- Local: per-region SHAP waterfall for any selected region-year.
- Coefficient plot for the Ridge baseline.
- Written interpretation in the report: which drivers push loss up or down, with the caveat that data is synthetic and relationships are associative only.

---

## 13. Streamlit Application Specification

**Goal:** a polished, professional, presentation-ready dashboard that a judge can open and understand in under two minutes.

### 13.1 General
- Run: `streamlit run app/streamlit_app.py`
- `st.set_page_config(layout="wide", page_title="Forest Loss Intelligence", page_icon="🌲")`
- Theme: dark-green accent palette (`#2E7D32`, `#66BB6A`, `#A5D6A7`), risk colors green → amber → red. Define in `.streamlit/config.toml`. Custom CSS in `app/style.css` for KPI cards (rounded, subtle shadow).
- Use **Plotly** for all charts (interactive, consistent theme). Use `@st.cache_data` for data and `@st.cache_resource` for loaded models.
- The app MUST read precomputed artifacts (`models/`, `reports/`) so it starts instantly. If artifacts are missing, show a clear message with the command `python -m src.train`.
- Sidebar (global): region multi-select, year range slider, scenario selector, download button for filtered data.
- Include loading spinners, helpful tooltips (`help=`) on controls, and no stack traces visible to the user (wrap in try/except with friendly errors).

### 13.2 Pages

**Page 1: Overview**
- Title, one-paragraph problem statement, dataset summary (50 regions, 20 years, 1,000 records).
- 5 KPI cards: total forest area (latest year), total loss since 2005 (ha), average annual loss %, number of Critical/High regions, best model's test MAE and Macro-F1.
- Line chart: total forest area over time; bar chart: annual total loss vs gain.
- Short "How it works" pipeline diagram (Data → Features → Models → Ensemble → Forecast → Hotspots).

**Page 2: Data Explorer (EDA)**
- Tabs: *Table* (filterable, with download) | *Distributions* (histograms/box plots per feature) | *Correlations* (heatmap) | *Time Trends* (per-region and national lines) | *Data Quality* (missing values, duplicates, validation checks, the leakage identity check shown as a green tick).
- Region comparison: choose 2 to 5 regions to overlay.
- Static-vs-dynamic feature explainer (protected_area, road_density are constant per region).

**Page 3: Hotspot Analysis**
- Ranked table of regions with risk score bars (`st.column_config.ProgressColumn`), tier badges and trend arrows.
- **Region heatmap** (region x year, color = `loss_pct`), since no coordinates exist. Do NOT fabricate geographic maps; optionally offer a clearly labeled "schematic grid layout" of the 50 regions (10 x 5 tiles).
- Scatter: risk score vs forest area (bubble size = loss_ha), colored by tier.
- Cluster archetype view (if clustering is implemented).
- Download `hotspots.csv`.

**Page 4: Model Performance**
- Leaderboard table of all models (sortable), highlighting the best per metric, with the Persistence baseline row pinned at the top.
- Toggle: Regression | Classification.
- Regression: actual-vs-predicted scatter, residual histogram, error by region and by year, MAE-by-forecast-horizon chart.
- Classification: confusion matrix heatmap (counts and normalized toggle), per-class precision/recall/F1 bar chart, ROC curves (OvR), calibration curve, Macro-F1 / Accuracy / Precision / Recall KPI tiles.
- Single vs ensemble comparison chart with CV error bars.
- "Methodology" expander showing the math (Section 8) via `st.latex`.

**Page 5: Forecast**
- Select region (or "All regions") and scenario. Line chart of historical + forecast `loss_pct` with shaded 80% prediction interval and a vertical divider at 2024.
- Forest area projection chart; table of forecasted values; download CSV.
- National-level aggregate forecast and the Top 10 regions by projected cumulative loss to 2030.
- Backtest panel showing how the forecast method performed historically.

**Page 6: What-If Simulator**
- Pick a region, then sliders: fire count change (%), agricultural expansion change (%), population density change (%), toggle policy active, toggle protected status.
- Instantly show predicted `loss_pct`, severity class with probability bars (gauge or progress), and the delta vs baseline. Include a one-click preset row (Baseline, Policy On, High Fire, Agri Surge, Conservation Push).
- Chart comparing scenarios side by side up to 2030.
- Include the disclaimer: "Model-based estimates on synthetic data; associations, not causal effects."

**Page 7: Explainability**
- SHAP global bar and beeswarm (pre-rendered images or `shap` plots), permutation importance bar.
- Region-year selector for a local SHAP waterfall with plain-language sentence ("Loss predicted high mainly because fire count is above average and no policy active").
- Partial dependence plots for key drivers.

### 13.3 UX quality bar
- No page takes longer than 3 seconds after first load.
- All charts have titles, axis labels with units (ha, %, mm) and readable legends.
- Layout works at 1366px width and full HD.
- Consistent number formatting (thousands separators, 2 decimals for %).
- A final "About / Methodology" section with the technical approach summary, so judges see the approach inside the app.

---

## 14. CLI and Reproducibility
```bash
pip install -r requirements.txt
python -m src.train --config config.yaml          # full pipeline: EDA artifacts, features, tuning, models, ensemble, forecast, hotspots
python -m src.train --fast                        # 10 Optuna trials, for smoke test
streamlit run app/streamlit_app.py
pytest -q
```
- Fixed seeds (`SEED=42`) for numpy, Python, sklearn, Optuna, XGBoost, LightGBM, CatBoost.
- Save `reports/run_metadata.json`: library versions, seed, timestamp, git hash if available, feature list, thresholds.
- Training of the entire pipeline on this dataset SHOULD finish in under 15 minutes on a normal laptop CPU (`--fast` under 2 minutes).
- A `requirements.txt` with pinned versions:
  `pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, catboost, optuna, shap, statsmodels, plotly, streamlit, matplotlib, seaborn, joblib, pyyaml, pyarrow, pytest` (optional: `ruptures`, `pymannkendall`).

---

## 15. Testing Requirements (`tests/`)
1. `test_no_leakage.py`: assert that no feature for row (r, t) uses data from year > t; assert that target year = t + horizon; assert thresholds come from the training split only; assert same-year `loss_ha` and `loss_pct` are not in the feature list when predicting the same year.
2. `test_features.py`: lag/rolling/slope functions on a tiny synthetic frame with known answers; first-year NaN handling; no cross-region bleed (lags never use another region's rows).
3. `test_splits.py`: every validation year is strictly later than all training years; no region-year appears in both sets.
4. `test_forecast.py`: mass-balance identity holds (area never negative, `A_{t+1} = A_t − L + G`); output has 50 regions x 6 years x scenarios; intervals satisfy `lo ≤ pred ≤ hi`.
5. Smoke test: `train --fast` produces all expected artifacts; the Streamlit app imports without errors.

---

## 16. Technical Report (`reports/technical_report.md`, auto-generated and then edited)
Sections: (1) Problem and data; (2) EDA findings and the leakage/identity discovery; (3) Preprocessing and encoding decisions with reasons; (4) Feature engineering with a feature table; (5) Validation strategy and why random CV was rejected; (6) Models, hyperparameters, and ensembling; (7) Mathematics; (8) Results tables and plots (regression and classification); (9) Baseline comparison; (10) Explainability; (11) Forecast and scenarios; (12) Hotspot methodology; (13) Limitations (synthetic data, small sample, 2-year test, noise-only features, associations are not causation); (14) Future work. Keep to 8 to 12 pages equivalent. Plain language, no invented claims.

---

## 17. Non-Goals and Constraints
- No NLP, no external data downloads, no satellite imagery (not provided).
- No fabricated geographic coordinates or maps.
- No claims of causal impact; use associative language.
- Do not tune on the final test set; do not report a score that was selected on it.
- Do not drop rows or silently impute; log every data transformation.
- Keep code typed (type hints), documented (docstrings), PEP8-compliant (`ruff` or `black`), and modular. No notebook-only logic: notebooks only call `src/`.

---

## 18. Implementation Phases (do in this order)

| Phase | Work | Output |
|---|---|---|
| 1 | Project scaffold, `config.yaml`, `data_loader.py` with validation | Validated dataframe, passing schema checks |
| 2 | EDA (`eda.py`) + leakage and identity checks | `reports/figures/eda_*.png`, EDA summary JSON |
| 3 | `targets.py`, `features.py`, `splits.py` + tests | `data/processed/features.parquet`, test_no_leakage passing |
| 4 | Baselines (persistence, region mean, trend, Ridge) | Baseline metrics to beat |
| 5 | Tree and boosting models, residual-target comparison | Model comparison table |
| 6 | Optuna tuning with expanding-window CV | `best_params.json` |
| 7 | Stacking and weighted blend, calibration, intervals | Final models in `models/` |
| 8 | Classification track (direct + regress-then-bin) with full metric suite | Classification report and confusion matrices |
| 9 | Explainability (SHAP, PDP) | Figures and `top_drivers` |
| 10 | Forecast 2025 to 2030, scenarios, backtest | `forecast_2025_2030.csv` |
| 11 | Hotspot scoring and clustering | `hotspots.csv` |
| 12 | Streamlit app (all 7 pages) | Working dashboard |
| 13 | Technical report, README, final tests, cleanup | Submission-ready repo |

---

## 19. Acceptance Criteria (definition of done)

**Correctness**
- [ ] No leakage tests fail; the temporal split is used for all reported scores.
- [ ] Persistence baseline reported, and the final regression model beats it on temporal CV MAE, **or** the report honestly states that it does not and explains why (do not hide this).
- [ ] Classification report includes Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1, MCC, kappa, confusion matrix, ROC-AUC.
- [ ] Single models and ensemble both evaluated and compared with per-fold results.
- [ ] Forecast backtest implemented; forecast mass balance holds.

**Reproducibility**
- [ ] `python -m src.train` runs end to end from a clean clone with only `requirements.txt` installed.
- [ ] All artifacts regenerate identically with the same seed (within floating-point tolerance).

**App**
- [ ] All 7 pages load without errors and use real artifacts.
- [ ] What-If simulator updates predictions interactively.
- [ ] Every chart labeled with units; downloads work.

**Documentation**
- [ ] README with setup, run commands, and screenshots.
- [ ] Technical report covers all items in Section 16.

**Performance targets** (guidance only, since the data is synthetic and the test is small; never tune toward a number at the cost of leakage). Reference points measured on 2022 to 2024 with a quick baseline: Persistence R² ≈ 0.948 / MAE ≈ 0.097; untuned sklearn GBR with lag features R² ≈ 0.940 / MAE ≈ 0.102. The engineered, tuned, residual-learning ensemble should aim to **match or improve on persistence** and reach a Macro-F1 clearly above the 0.33 chance level for ternary classes. Expect the classification Macro-F1 to depend heavily on how close class boundaries are, so also report adjacent-class accuracy.

---

## 20. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Target column unknown in the competition | Config-driven target (Section 5.4), regression + classification + forecast all delivered |
| Leakage via same-year identity (`loss_pct = loss_ha / area`) | Next-year target framing, leakage tests, validation script |
| Models fail to beat persistence | Residual learning, strong lag features, honest reporting, ensemble of diverse models |
| Tiny test set (100 rows) makes scores noisy | Expanding-window CV + spatial CV + bootstrap CIs |
| Overfitting hyperparameter search | Modest search spaces, nested validation, untouched final test |
| Hidden test set has unseen regions or years | Exclude raw `region_id` from the spatial-generalization model; region-baseline features computed from history |
| Synthetic data quirks (noise-only columns) | Feature selection comparison; keep a lean feature set |
| Streamlit slowness | Precomputed artifacts, caching, no training inside the app |

---

*End of PRD. Build phase by phase, run the tests after each phase, and keep the report honest about what the data does and does not support.*
