# CanopyIntel: Product Context (PRODUCT.md)

## Product Identity & Purpose
**CanopyIntel** is an environmental intelligence system engineered to predict subnational forest loss rates, classify deforestation severity, project multi-year recursive canopy trajectories (2025–2030), and prioritize high-risk ecological intervention zones across 50 subnational jurisdictions spanning 2005 to 2024.

## Problem Statement & Operational Value
- **Core Challenge:** Environmental agencies face persistent deforestation threats driven by agricultural conversion, fire outbreaks, and governance gaps.
- **Data Reality:** Subnational panel data exhibits extreme 1-year autocorrelation ($r = 0.925$). Naive regression models predict $\hat{y}_{t+1} \approx y_t$, showing deceptive $R^2 > 0.94$ but zero causal foresight ($\text{MASE} \ge 1.0$).
- **The Breakthrough:** CanopyIntel applies **Residual Log-Ratio Learning** ($z = \ln(y_{t+1}/y_t)$), predicting rate growth/decay instead of base levels. This beats naive persistence on untouched 2023–2024 test data with 95% bootstrap confidence ($\text{MASE} = 0.9618$).

## Target Audience
1. **Conservation Agencies & Forest Rangers:** Identifying prioritized jurisdictions for rapid field deployment.
2. **Environmental Policymakers:** Simulating policy and climate counterfactuals through 2030.
3. **Academic Evaluators & Competition Judges:** Reviewing zero-leakage cross-validation, physical mass conservation, and TreeSHAP explainability.

## User Journeys
1. **Executive Screening (1-minute):** View the master KPI strip, computed insight headline, and 10x5 schematic risk cartogram.
2. **What-If Policy Lab (3-minutes):** Adjust regional fire events, agricultural pressure, or policy status to observe immediate model responses.
3. **Multi-Year Scenario Planning (5-minutes):** Compare 2025–2030 recursive canopy trajectories under Status Quo, Universal Policy, and Severe Climate Shock pathways.
4. **Audit & Transparency (10-minutes):** Review model zoo leaderboards, bootstrap error whiskers, confusion matrices, and TreeSHAP beeswarm plots.
