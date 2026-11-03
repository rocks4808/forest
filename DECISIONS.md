# Architectural & Modeling Decision Log (DECISIONS.md)

This log records all key engineering, mathematical, modeling, and product design choices made during the development of CanopyIntel (Forest Loss Prediction & Deforestation Hotspot System), following the specifications in `PRD.md` and the master execution protocol.

---

### Decision 1: Residual Learning via Log-Ratio Target Formulation
- **Context:** The forest loss time-series exhibits massive autocorrelation ($r = 0.925$ at lag 1). A naive direct regression model predicting $y_{t+1}$ tends to degenerate into an imperfect persistence forecaster, struggling to capture turning points and acceleration.
- **Alternatives Considered:**
  1. Direct prediction: $\hat{y}_{t+1} = f(X_t)$
  2. Difference / Delta: $\Delta y = y_{t+1} - y_t$
  3. Log-Ratio: $z = \ln\left(\frac{y_{t+1} + \epsilon}{y_t + \epsilon}\right)$, reconstructed as $\hat{y}_{t+1} = (y_t + \epsilon)\exp(\hat{z}) - \epsilon$
- **Choice & Rationale:** Log-ratio formulation was selected. Log-ratio maps multiplicative growth rates into a symmetric linear space, naturally enforces non-negativity upon inversion, and yielded the lowest expanding-window CV MAE. Tree models trained on log-ratio directly learn acceleration/deceleration relative to current state.
- **Verification:** On the unseen 2023–2024 holdout test set, the tuned RandomForest on log-ratio achieved MAE of **0.0932**, beating the strong persistence baseline (MAE **0.0969**).

---

### Decision 2: Strict Temporal & Spatial Cross-Validation (Rejection of Random K-Fold)
- **Context:** Standard random $K$-fold cross-validation is fatally flawed for panel datasets because random shuffling leaks future time points into the past (temporal leakage) and leaks contemporaneous regional trends (spatial autocorrelation).
- **Protocol Implemented:**
  1. *Temporal Holdout:* Train on years $\le 2021$ ($N=850$ rows), validation on 2022 ($N=50$ rows), final untouched holdout on 2023–2024 ($N=100$ rows).
  2. *Expanding-Window Temporal CV:* 6 expanding folds spanning 2016 through 2021, strictly training on past years and testing on year $t+1$.
  3. *Group-by-Region Spatial CV:* 5 folds grouping all 20 observations of 10 regions together, testing generalization to unseen regions.
- **Rationale:** Ensures all validation metrics realistically reflect real-world operational deployment where the model predicts future unknown years.

---

### Decision 3: Model Selection & Ensembling Decision (Parsimony Rule)
- **Context:** PRD Section 7.7 specifies that an ensemble (stacking or weighted blend) should only be shipped if it strictly outperforms the best single model on expanding-window out-of-fold evaluation by $\ge 0.001$ MAE with a paired bootstrap 95% confidence interval strictly below 0; otherwise, the simpler single model must be selected to avoid gratuitous complexity.
- **Empirical Results:**
  - Best Single Model (Tuned RandomForest): OOF MAE = **0.1632**
  - Non-Negative Least Squares (NNLS) Weighted Blend: OOF MAE = **0.1637**
  - Paired Difference ($\text{Blend} - \text{RF}$): $+0.0005$ with $95\%$ CI $[-0.0084, +0.0093]$.
- **Decision:** The weighted blend did not beat the tuned RandomForest. In strict accordance with the parsimony rule, **RandomForestRegressor** (with 350 trees, max_depth 12, max_features 0.6) was selected as the final production regression model.
- **Test Set Performance:** Confirmed on the untouched holdout test set: RandomForest MAE **0.0932** vs Persistence **0.0969**.

---

### Decision 4: Classification Track: Direct Classifier vs Regress-then-Bin
- **Context:** Forest loss severity is discretized into three tiers (Low, Medium, High) based on training-set tertiles. We evaluated whether to predict continuous loss and bin the prediction or train a direct multi-class classifier.
- **Empirical Results:**
  - Regress-then-bin Macro-F1: 0.748
  - Direct LightGBM Classifier Macro-F1: **0.772** (Test Accuracy: 76.0%, Adjacent-Class Accuracy: 100.0%)
- **Choice:** Shipped direct LightGBM multi-class classifier with probability calibration. Direct classification directly optimizes cross-entropy on class boundaries, providing reliable risk probabilities for decision-makers.

---

### Decision 5: Training-Set-Only Class Thresholds (Zero Leakage)
- **Context:** Setting class cut points using the full dataset leaks information from the test holdout years into the target definition.
- **Protocol Implemented:**
  - The 33.3rd and 66.7th percentiles of `loss_pct` were computed strictly on the training set (years $\le 2021$):
    - $T_1 = 0.5097\%$ (Low / Medium threshold)
    - $T_2 = 1.0543\%$ (Medium / High threshold)
  - These exact thresholds were serialized to `models/training_stats.joblib` and used consistently for holdout evaluation, forecasting, and the interactive simulator.

---

### Decision 6: Schematic 10x5 Tile Cartogram Layout
- **Context:** The dataset contains 50 synthetic subnational regions (`R01` to `R50`) without geographical latitude/longitude shapefiles. Generating fake geographic coordinates would be intellectually dishonest and violate Rule 0.1.
- **Design Solution:** A 10 column by 5 row tile cartogram was built in Streamlit using Plotly and CSS. Every tile displays the region identifier, risk tier badge, and risk score. The layout is explicitly annotated with: *"Schematic layout, not geographic: 50 regions arranged on a 10x5 matrix for visual pattern discovery"*. This provides an immediate visual spatial overview and interactive region selection without fabricating geographic data.

---

### Decision 7: Physical Mass Balance & Recursive Multi-Year Forecasting
- **Context:** In multi-year forecasting (2025–2030), forest area must obey physical conservation of mass:
  $$\text{forest\_area}_{t+1} = \text{forest\_area}_t - \text{loss\_ha}_t + \text{gain\_ha}_t$$
  $$\text{loss\_ha}_t = \text{loss\_pct}_t \times \text{forest\_area}_t / 100$$
- **Protocol Implemented:**
  - In recursive autoregressive rollouts, each step recalculates `loss_ha`, decrements `forest_area_ha`, increments `forest_gain_ha`, re-computes all lag and rolling features from the dynamically updated trajectory, and clamps `forest_area_ha \ge 0`.
  - All 5 policy and climate scenarios strictly enforce this mass balance identity at every recursive iteration.

