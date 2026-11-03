"""Unit tests for feature engineering logic."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_lag_features,
    compute_rolling_features,
    compute_trend_features,
    compute_policy_features,
    build_features,
)


@pytest.fixture
def synthetic_panel():
    """Small 2-region, 5-year synthetic dataframe with known values."""
    data = []
    for r in ["R01", "R02"]:
        for y in range(2005, 2010):
            data.append({
                "region_id": r,
                "year": y,
                "forest_area_ha": 100000 - (y - 2005) * 1000,
                "loss_ha": 1000 + (y - 2005) * 100,
                "gain_ha": 100,
                "loss_pct": (1000 + (y - 2005) * 100) / (100000 - (y - 2005) * 1000) * 100,
                "population_density": 10.0 + (y - 2005),
                "road_density_km_per_km2": 1.5 if r == "R01" else 2.5,
                "agri_expansion_pct": 1.0,
                "rainfall_mm": 1200,
                "fire_count": 50 + (y - 2005) * 10,
                "protected_area": 0 if r == "R01" else 1,
                "policy_active": 1 if (r == "R01" and y >= 2007) else 0,
            })
    return pd.DataFrame(data)


def test_lag_computation_and_no_bleed(synthetic_panel):
    """Test that lag values match shifted values and never bleed between regions."""
    df_lags = compute_lag_features(synthetic_panel)
    
    # Year 2005 (first year) should have NaN for lag-1
    r1_2005 = df_lags[(df_lags["region_id"] == "R01") & (df_lags["year"] == 2005)]
    assert np.isnan(r1_2005["loss_pct_lag1"].iloc[0])

    # Year 2006 should have Year 2005's loss_pct
    r1_2006 = df_lags[(df_lags["region_id"] == "R01") & (df_lags["year"] == 2006)]
    assert r1_2006["loss_pct_lag1"].iloc[0] == pytest.approx(r1_2005["loss_pct"].iloc[0])

    # R02's first year must also be NaN (no bleed from R01's last year)
    r2_2005 = df_lags[(df_lags["region_id"] == "R02") & (df_lags["year"] == 2005)]
    assert np.isnan(r2_2005["loss_pct_lag1"].iloc[0])


def test_policy_timing_features(synthetic_panel):
    """Test years_since_policy computation."""
    df_pol = compute_policy_features(synthetic_panel)
    r1 = df_pol[df_pol["region_id"] == "R01"].sort_values("year")
    
    # In synthetic panel, R01 policy starts in 2007
    # 2005: policy inactive -> years_since_policy == -1
    assert r1[r1["year"] == 2005]["years_since_policy"].iloc[0] == -1
    # 2007: policy starts -> years_since_policy == 0
    assert r1[r1["year"] == 2007]["years_since_policy"].iloc[0] == 0
    # 2008: 1 year in -> years_since_policy == 1
    assert r1[r1["year"] == 2008]["years_since_policy"].iloc[0] == 1


def test_rolling_and_trend_features(synthetic_panel):
    """Test that rolling mean and slope features compute without errors."""
    df_roll = compute_rolling_features(synthetic_panel)
    assert "loss_pct_roll_mean3" in df_roll.columns
    assert "loss_pct_roll_std3" in df_roll.columns

    df_trend = compute_trend_features(synthetic_panel)
    assert "loss_pct_diff1" in df_trend.columns
    assert "net_loss_ha" in df_trend.columns

