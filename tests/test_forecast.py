"""Tests for forecasting mass balance, scenarios, and intervals."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_validate
from src.features import build_features, get_feature_columns
from src.targets import build_all_targets
from src.models import get_model
from src.forecast import forecast_recursive, SCENARIOS


@pytest.fixture(scope="module")
def prepared_system():
    raw = load_and_validate()
    targets_df, thresholds = build_all_targets(raw, horizon=1)
    feat_df, stats = build_features(targets_df)
    feature_cols = get_feature_columns(feat_df)
    
    # Train simple Ridge model on delta target for fast testing
    train_mask = targets_df["target_year"] <= 2021
    X_train = feat_df.loc[train_mask, feature_cols].fillna(0.0)
    y_train = targets_df.loc[train_mask, "delta"].fillna(0.0)
    
    model = get_model("ridge", task="regression", params={"alpha": 1.0})
    model.fit(X_train, y_train)
    
    return {
        "raw": raw,
        "model": model,
        "training_stats": stats,
        "feature_cols": feature_cols,
        "thresholds": thresholds,
    }


def test_forecast_shape_and_completeness(prepared_system):
    """Forecast must produce exactly 50 regions x 6 years (2025-2030)."""
    df_fc = forecast_recursive(
        model=prepared_system["model"],
        df_raw=prepared_system["raw"],
        training_stats=prepared_system["training_stats"],
        feature_cols=prepared_system["feature_cols"],
        thresholds=prepared_system["thresholds"],
        target_form="delta",
        start_year=2024,
        horizon_years=6,
        scenario="baseline",
    )

    assert len(df_fc) == 50 * 6  # 300 rows
    assert sorted(df_fc["year"].unique()) == [2025, 2026, 2027, 2028, 2029, 2030]
    assert df_fc["region_id"].nunique() == 50


def test_forecast_mass_balance_and_non_negative(prepared_system):
    """Forest area must remain non-negative and satisfy mass balance."""
    df_fc = forecast_recursive(
        model=prepared_system["model"],
        df_raw=prepared_system["raw"],
        training_stats=prepared_system["training_stats"],
        feature_cols=prepared_system["feature_cols"],
        thresholds=prepared_system["thresholds"],
        target_form="delta",
        start_year=2024,
        horizon_years=6,
        scenario="baseline",
    )

    assert (df_fc["forest_area_pred"] >= 0).all(), "Forest area fell below 0!"
    assert (df_fc["loss_ha_pred"] >= 0).all(), "Loss ha fell below 0!"
    assert (df_fc["gain_ha_pred"] >= 0).all(), "Gain ha fell below 0!"


def test_prediction_intervals(prepared_system):
    """Prediction intervals must satisfy lo <= pred <= hi."""
    df_fc = forecast_recursive(
        model=prepared_system["model"],
        df_raw=prepared_system["raw"],
        training_stats=prepared_system["training_stats"],
        feature_cols=prepared_system["feature_cols"],
        thresholds=prepared_system["thresholds"],
        target_form="delta",
        start_year=2024,
        horizon_years=6,
        scenario="baseline",
    )

    assert (df_fc["loss_pct_lo"] <= df_fc["loss_pct_pred"]).all()
    assert (df_fc["loss_pct_pred"] <= df_fc["loss_pct_hi"]).all()

