"""Recursive multi-step forecasting module for 2025-2030.

Implements:
- Driver projection (linear population trend, constant roads/protection, policy states, 5-yr climate/fire means).
- Recursive multi-step autoregressive roll-forward with dynamic feature recalculation.
- Mass-balance identity: forest_area(t+1) = max(0, forest_area(t) - loss_ha(t) + gain_ha(t)).
- Five policy/environmental scenarios.
- Prediction intervals (via conformal validation residuals or quantile regression).
- Historical backtesting from 2019 and 2021 origins.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.features import build_features, get_feature_columns

logger = logging.getLogger(__name__)

SCENARIOS: List[str] = [
    "baseline",
    "policy_on_everywhere",
    "agri_surge",
    "high_fire",
    "conservation_push",
]


# ===================================================================
# Exogenous Driver Projections
# ===================================================================

def project_exogenous_drivers(
    df_history: pd.DataFrame,
    region_id: str,
    future_year: int,
    scenario: str = "baseline",
    n_recent: int = 5,
) -> Dict[str, float]:
    """Project environmental and development drivers for a region at future_year."""
    reg_df = df_history[df_history["region_id"] == region_id].sort_values("year")
    last_row = reg_df.iloc[-1]
    recent_df = reg_df.tail(n_recent)

    # 1. Population density: linear extrapolation based on regional growth slope
    if len(reg_df) >= 2:
        years = reg_df["year"].values
        pop = reg_df["population_density"].values
        slope, intercept = np.polyfit(years, pop, 1)
        pop_proj = float(slope * future_year + intercept)
    else:
        pop_proj = float(last_row["population_density"])

    # 2. Static traits
    road_density = float(last_row["road_density_km_per_km2"])
    protected_area = int(last_row["protected_area"])

    # 3. Policy active logic
    if scenario in ["policy_on_everywhere", "conservation_push"]:
        policy_active = 1
    else:
        # baseline: stays at last known state
        policy_active = int(last_row["policy_active"])

    # 4. Agri expansion, rainfall, fire count (5-year regional baseline)
    agri_base = float(recent_df["agri_expansion_pct"].mean())
    rainfall_base = int(round(recent_df["rainfall_mm"].mean()))
    fire_base = int(round(recent_df["fire_count"].mean()))

    # Apply scenario adjustments
    if scenario == "agri_surge":
        agri_base *= 1.30
    elif scenario == "high_fire":
        fire_base = int(round(fire_base * 1.25))
    elif scenario == "conservation_push":
        fire_base = int(round(fire_base * 0.80))

    return {
        "region_id": region_id,
        "year": future_year,
        "population_density": max(0.0, pop_proj),
        "road_density_km_per_km2": road_density,
        "protected_area": protected_area,
        "policy_active": policy_active,
        "agri_expansion_pct": max(0.0, agri_base),
        "rainfall_mm": max(0, rainfall_base),
        "fire_count": max(0, fire_base),
    }


# ===================================================================
# Recursive Multi-Step Forecasting
# ===================================================================

def forecast_recursive(
    model: Any,
    df_raw: pd.DataFrame,
    training_stats: Dict[str, Any],
    feature_cols: List[str],
    thresholds: Dict[str, float],
    target_form: str = "delta",  # 'direct', 'delta', 'log_ratio'
    start_year: int = 2024,
    horizon_years: int = 6,  # 2025 to 2030
    scenario: str = "baseline",
    residual_std: float = 0.08,  # for 80% prediction interval
) -> pd.DataFrame:
    """Run recursive panel roll-forward from start_year to start_year + horizon_years.

    Guarantees mass balance:
    area(t+1) = area(t) - loss(t) + gain(t)
    """
    # Work on a copy of historical records up to start_year
    sim_df = df_raw[df_raw["year"] <= start_year].copy().sort_values(["region_id", "year"]).reset_index(drop=True)

    # Compute trailing gain-to-loss ratio per region to project future gain_ha
    recent_ratios = {}
    for r_id, grp in sim_df.groupby("region_id"):
        tot_loss = grp.tail(5)["loss_ha"].sum()
        tot_gain = grp.tail(5)["gain_ha"].sum()
        recent_ratios[r_id] = float(tot_gain / tot_loss) if tot_loss > 0 else 0.10

    forecast_records = []
    z_80 = 1.282  # 80% two-sided normal quantile

    for step in range(1, horizon_years + 1):
        target_year = start_year + step
        feature_year = target_year - 1

        # 1. Project drivers for all 50 regions for feature_year if step > 1
        # For predicting year t+1, features come from year t.
        # At step 1 (target 2025), features come from 2024 (already in sim_df).
        # At step 2 (target 2026), features come from 2025 (need to project exogenous drivers for 2025).
        if step > 1:
            new_rows = []
            for r_id in sorted(sim_df["region_id"].unique()):
                drivers = project_exogenous_drivers(sim_df, r_id, feature_year, scenario=scenario)
                # The endogenous states (forest_area_ha, loss_ha, gain_ha, loss_pct) were already
                # appended to sim_df at the end of the previous step.
                # Here we update any exogenous driver columns in that row.
                row_idx = sim_df[(sim_df["region_id"] == r_id) & (sim_df["year"] == feature_year)].index
                if len(row_idx) > 0:
                    for k, v in drivers.items():
                        if k not in ["region_id", "year"]:
                            sim_df.loc[row_idx, k] = v

        # 2. Build complete features up to feature_year
        features_df, _ = build_features(sim_df, training_stats=training_stats)

        # Slice features for year == feature_year
        step_feat = features_df[features_df["year"] == feature_year].sort_values("region_id").reset_index(drop=True)

        X_step = step_feat[feature_cols].copy().fillna(0.0)

        # 3. Model inference
        raw_pred = model.predict(X_step)

        # Map back from delta or log-ratio target form
        current_loss_pct = step_feat["loss_pct"].values
        if target_form == "delta":
            pred_loss_pct = current_loss_pct + raw_pred
        elif target_form == "log_ratio":
            pred_loss_pct = current_loss_pct * np.exp(raw_pred)
        else:
            pred_loss_pct = raw_pred

        # Physical constraints: loss_pct >= 0.01%
        pred_loss_pct = np.clip(pred_loss_pct, 0.01, 15.0)

        # Uncertainty intervals (scales gently with sqrt(step))
        band = z_80 * residual_std * np.sqrt(step)
        lo_pct = np.clip(pred_loss_pct - band, 0.01, None)
        hi_pct = pred_loss_pct + band

        # Convert to hectares and update mass balance
        curr_area = step_feat["forest_area_ha"].values
        pred_loss_ha = np.round(pred_loss_pct * curr_area / 100.0).astype(int)

        # Predicted gain
        pred_gain_ha = np.array([
            int(round(pred_loss_ha[i] * recent_ratios.get(step_feat["region_id"].iloc[i], 0.1)))
            for i in range(len(step_feat))
        ])

        # Mass balance: Next year area
        next_area = np.maximum(0, curr_area - pred_loss_ha + pred_gain_ha)

        # Severity classification
        low_thresh = thresholds.get("low_upper", 0.96)
        high_thresh = thresholds.get("high_lower", 1.40)
        severities = np.where(
            pred_loss_pct <= low_thresh,
            0,
            np.where(pred_loss_pct <= high_thresh, 1, 2)
        )

        # Append predicted year to sim_df so next step can compute lags
        for i, r_id in enumerate(step_feat["region_id"].values):
            forecast_records.append({
                "region_id": r_id,
                "year": target_year,
                "scenario": scenario,
                "loss_pct_pred": float(np.round(pred_loss_pct[i], 4)),
                "loss_pct_lo": float(np.round(lo_pct[i], 4)),
                "loss_pct_hi": float(np.round(hi_pct[i], 4)),
                "loss_ha_pred": int(pred_loss_ha[i]),
                "gain_ha_pred": int(pred_gain_ha[i]),
                "forest_area_pred": int(next_area[i]),
                "severity_pred": int(severities[i]),
            })

            # Create new row in sim_df for target_year
            sim_df = pd.concat([sim_df, pd.DataFrame([{
                "region_id": r_id,
                "year": target_year,
                "forest_area_ha": int(next_area[i]),
                "loss_ha": int(pred_loss_ha[i]),
                "gain_ha": int(pred_gain_ha[i]),
                "loss_pct": float(pred_loss_pct[i]),
                "population_density": float(step_feat["population_density"].iloc[i]),
                "road_density_km_per_km2": float(step_feat["road_density_km_per_km2"].iloc[i]),
                "agri_expansion_pct": float(step_feat["agri_expansion_pct"].iloc[i]),
                "rainfall_mm": int(step_feat["rainfall_mm"].iloc[i]),
                "fire_count": int(step_feat["fire_count"].iloc[i]),
                "protected_area": int(step_feat["protected_area"].iloc[i]),
                "policy_active": int(step_feat["policy_active"].iloc[i]),
            }])], ignore_index=True)

    return pd.DataFrame(forecast_records)


def run_all_scenarios(
    model: Any,
    df_raw: pd.DataFrame,
    training_stats: Dict[str, Any],
    feature_cols: List[str],
    thresholds: Dict[str, float],
    target_form: str = "delta",
    residual_std: float = 0.08,
) -> pd.DataFrame:
    """Run recursive forecast for all defined scenarios."""
    all_dfs = []
    for scen in SCENARIOS:
        logger.info("Forecasting scenario: %s...", scen)
        df_scen = forecast_recursive(
            model=model,
            df_raw=df_raw,
            training_stats=training_stats,
            feature_cols=feature_cols,
            thresholds=thresholds,
            target_form=target_form,
            start_year=2024,
            horizon_years=6,
            scenario=scen,
            residual_std=residual_std,
        )
        all_dfs.append(df_scen)
    return pd.concat(all_dfs, ignore_index=True)


# ===================================================================
# Forecast Backtest
# ===================================================================

def run_forecast_backtest(
    model: Any,
    df_raw: pd.DataFrame,
    training_stats: Dict[str, Any],
    feature_cols: List[str],
    thresholds: Dict[str, float],
    target_form: str = "delta",
    origin_years: List[int] = [2019, 2021],
) -> pd.DataFrame:
    """Evaluate multi-step recursive forecasting on historical data."""
    records = []
    max_year = int(df_raw["year"].max())

    for origin in origin_years:
        horizon = max_year - origin
        if horizon <= 0:
            continue

        pred_df = forecast_recursive(
            model=model,
            df_raw=df_raw,
            training_stats=training_stats,
            feature_cols=feature_cols,
            thresholds=thresholds,
            target_form=target_form,
            start_year=origin,
            horizon_years=horizon,
            scenario="baseline",
        )

        # Merge with ground truth
        actuals = df_raw[["region_id", "year", "loss_pct", "loss_ha", "forest_area_ha"]].copy()
        merged = pd.merge(pred_df, actuals, on=["region_id", "year"], suffixes=("_pred", "_true"))

        for h_step, yr in enumerate(range(origin + 1, max_year + 1), start=1):
            sub = merged[merged["year"] == yr]
            mae_pct = float(np.mean(np.abs(sub["loss_pct_pred"] - sub["loss_pct"])))
            records.append({
                "origin_year": origin,
                "horizon_step": h_step,
                "target_year": yr,
                "MAE_loss_pct": mae_pct,
                "n_regions": len(sub),
            })

    return pd.DataFrame(records)

