"""Hotspot identification, degradation detection, and regional risk scoring.

Implements:
- Multidimensional risk score (0-100) combining predicted loss, 5-yr slope, high-severity probability,
  fire pressure, human pressure, and conservation deficit.
- Risk tiers: Critical (>=75), High (50-75), Moderate (25-50), Low (<25).
- Trend classification: Accelerating / Stable / Improving via 5-year OLS slope and p-value.
- Policy change-point detection (two-sample t-test before vs after policy activation).
- Regional clustering into archetypes using KMeans on structural indicators.
- Output generation for reports/hotspots.csv.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "predicted_loss_pct": 0.35,
    "trend_slope": 0.20,
    "prob_high": 0.20,
    "fire_pressure": 0.10,
    "human_pressure": 0.10,
    "no_protection": 0.05,
}


def _percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Compute percentile rank [0, 100] for an array."""
    ranks = stats.rankdata(arr, method="average")
    return (ranks - 1.0) / max(1, len(arr) - 1) * 100.0


def compute_5yr_slope_and_trend(df_reg: pd.DataFrame) -> Tuple[float, float, str]:
    """Compute 5-year OLS slope, p-value, and trend label for loss_pct."""
    sub = df_reg.tail(5)
    if len(sub) < 3:
        return 0.0, 1.0, "Stable"

    x = sub["year"].values
    y = sub["loss_pct"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Classify: Accelerating if slope > 0.02 and p < 0.20
    # Improving if slope < -0.02 and p < 0.20
    # Otherwise Stable
    if slope > 0.02 and p_value < 0.20:
        label = "Accelerating"
    elif slope < -0.02 and p_value < 0.20:
        label = "Improving"
    else:
        label = "Stable"

    return float(slope), float(p_value), label


def detect_policy_change_points(df_raw: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Detect structural change in loss_pct before vs after policy activation."""
    change_points = {}
    for r_id, grp in df_raw.groupby("region_id"):
        grp = grp.sort_values("year")
        policy_active = grp["policy_active"].values
        loss_pct = grp["loss_pct"].values

        # Check if policy changed from 0 to 1
        starts = np.where((policy_active[1:] == 1) & (policy_active[:-1] == 0))[0]
        if len(starts) > 0:
            start_idx = starts[0] + 1
            start_year = int(grp["year"].iloc[start_idx])
            before = loss_pct[:start_idx]
            after = loss_pct[start_idx:]

            if len(before) >= 2 and len(after) >= 2:
                t_stat, p_val = stats.ttest_ind(before, after, equal_var=False)
                mean_diff = float(np.mean(after) - np.mean(before))
                change_points[r_id] = {
                    "policy_start_year": start_year,
                    "mean_before": float(np.mean(before)),
                    "mean_after": float(np.mean(after)),
                    "delta": mean_diff,
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05),
                }
            else:
                change_points[r_id] = {"policy_start_year": start_year, "delta": 0.0, "p_value": 1.0, "significant": False}
        else:
            # Policy never changed
            ever_active = bool(policy_active.max() == 1)
            change_points[r_id] = {
                "policy_start_year": None,
                "delta": 0.0,
                "p_value": 1.0,
                "significant": False,
                "always_active": ever_active,
            }
    return change_points


def cluster_regional_archetypes(df_raw: pd.DataFrame, n_clusters: int = 4, seed: int = 42) -> pd.DataFrame:
    """Cluster regions into archetypes based on multi-year profile averages."""
    profiles = []
    for r_id, grp in df_raw.groupby("region_id"):
        profiles.append({
            "region_id": r_id,
            "mean_loss_pct": float(grp["loss_pct"].mean()),
            "mean_fire_count": float(grp["fire_count"].mean()),
            "mean_agri_exp": float(grp["agri_expansion_pct"].mean()),
            "mean_road_density": float(grp["road_density_km_per_km2"].iloc[0]),
            "mean_pop_density": float(grp["population_density"].mean()),
            "protected": int(grp["protected_area"].iloc[0]),
        })
    prof_df = pd.DataFrame(profiles)
    feature_cols = ["mean_loss_pct", "mean_fire_count", "mean_agri_exp", "mean_road_density", "mean_pop_density"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(prof_df[feature_cols])

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    prof_df["cluster_id"] = km.fit_predict(X_scaled)

    # Name clusters meaningfully based on centroids
    centroids = pd.DataFrame(km.cluster_centers_, columns=feature_cols)
    cluster_names = {}
    for cid in range(n_clusters):
        row = centroids.iloc[cid]
        if row["mean_fire_count"] > 0.5:
            cluster_names[cid] = "High Fire Pressure"
        elif row["mean_road_density"] > 0.5:
            cluster_names[cid] = "Rapid Infrastructure"
        elif row["mean_loss_pct"] < -0.5:
            cluster_names[cid] = "Low-Impact Preserved"
        else:
            cluster_names[cid] = "Agricultural Frontier"

    prof_df["cluster_archetype"] = prof_df["cluster_id"].map(cluster_names)
    return prof_df[["region_id", "cluster_id", "cluster_archetype"]]


def build_hotspots_table(
    df_raw: pd.DataFrame,
    pred_next_loss_pct: Dict[str, float],
    prob_high: Dict[str, float],
    shap_top_drivers: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute composite risk scores, tiers, and build reports/hotspots.csv."""
    w = weights or DEFAULT_WEIGHTS
    
    # 1. Cluster archetypes
    archetypes = cluster_regional_archetypes(df_raw)
    
    # 2. Change point detection
    change_points = detect_policy_change_points(df_raw)

    regions = sorted(df_raw["region_id"].unique())
    records = []

    for r_id in regions:
        reg_df = df_raw[df_raw["region_id"] == r_id].sort_values("year")
        last_row = reg_df.iloc[-1]

        # 5-year trend slope
        slope, p_val, trend_label = compute_5yr_slope_and_trend(reg_df)

        pred_loss = pred_next_loss_pct.get(r_id, float(last_row["loss_pct"]))
        p_hi = prob_high.get(r_id, 0.33)
        recent_fire = float(reg_df.tail(3)["fire_count"].mean())
        
        # Human pressure components
        recent_agri = float(reg_df.tail(3)["agri_expansion_pct"].mean())
        road = float(last_row["road_density_km_per_km2"])
        pop = float(last_row["population_density"])
        human_pressure_raw = (pop / 200.0) + (road / 1.5) + (recent_agri / 2.0)

        # Protection deficit (1 if neither protected nor policy active, 0 otherwise)
        deficit = 1.0 if (last_row["protected_area"] == 0 and last_row["policy_active"] == 0) else 0.0

        cp_info = change_points.get(r_id, {})

        records.append({
            "region_id": r_id,
            "predicted_loss_pct_next": pred_loss,
            "slope_5yr": slope,
            "trend_label": trend_label,
            "p_high": p_hi,
            "fire_pressure_raw": recent_fire,
            "human_pressure_raw": human_pressure_raw,
            "protection_deficit": deficit,
            "forest_area_ha": int(last_row["forest_area_ha"]),
            "policy_active": int(last_row["policy_active"]),
            "protected_area": int(last_row["protected_area"]),
            "policy_start_year": cp_info.get("policy_start_year"),
            "policy_delta": cp_info.get("delta", 0.0),
        })

    hotspot_df = pd.DataFrame(records)

    # Convert components to percentile ranks across the 50 regions
    pct_pred = _percentile_rank(hotspot_df["predicted_loss_pct_next"].values)
    pct_slope = _percentile_rank(hotspot_df["slope_5yr"].values)
    pct_phi = _percentile_rank(hotspot_df["p_high"].values)
    pct_fire = _percentile_rank(hotspot_df["fire_pressure_raw"].values)
    pct_human = _percentile_rank(hotspot_df["human_pressure_raw"].values)
    deficit_bonus = hotspot_df["protection_deficit"].values * 100.0

    # Composite Risk Score (0 - 100)
    risk_score = (
        w["predicted_loss_pct"] * pct_pred
        + w["trend_slope"] * pct_slope
        + w["prob_high"] * pct_phi
        + w["fire_pressure"] * pct_fire
        + w["human_pressure"] * pct_human
        + w["no_protection"] * deficit_bonus
    )
    hotspot_df["risk_score"] = np.round(risk_score, 1)

    # Assign Tiers
    hotspot_df["tier"] = np.where(
        hotspot_df["risk_score"] >= 75.0,
        "Critical",
        np.where(
            hotspot_df["risk_score"] >= 50.0,
            "High",
            np.where(hotspot_df["risk_score"] >= 25.0, "Moderate", "Low"),
        ),
    )

    # Attach archetypes
    hotspot_df = pd.merge(hotspot_df, archetypes, on="region_id", how="left")

    # Attach SHAP top drivers if provided
    if shap_top_drivers:
        hotspot_df["top_drivers"] = hotspot_df["region_id"].map(shap_top_drivers).fillna("Baseline trend")
    else:
        hotspot_df["top_drivers"] = "Fire count; agri expansion; road density"

    # Sort descending by risk score
    hotspot_df = hotspot_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return hotspot_df

