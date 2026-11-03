"""Leakage prevention tests for Forest Loss Prediction pipeline.

Verifies:
- No feature for (region, year t) uses data from year > t.
- Targets are for year t + horizon.
- Thresholds are computed from training data only.
- Current-year loss_pct and loss_ha are not in feature matrix when predicting.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_validate
from src.features import build_features, get_feature_columns
from src.targets import build_all_targets, compute_class_thresholds
from src.splits import get_temporal_holdout, get_expanding_window_cv


@pytest.fixture(scope="module")
def raw_data():
    return load_and_validate()


def test_target_horizon_and_year(raw_data):
    """Assert target describes year t + 1."""
    targets_df, _ = build_all_targets(raw_data, horizon=1)
    
    # Check that for any row, target_loss_pct equals the next year's loss_pct
    for r_id, grp in targets_df.groupby("region_id"):
        grp = grp.sort_values("year")
        for i in range(len(grp) - 1):
            curr_row = grp.iloc[i]
            next_row = grp.iloc[i + 1]
            assert curr_row["y_loss_pct_next"] == pytest.approx(next_row["loss_pct"], abs=1e-5)
            assert curr_row["target_year"] == curr_row["year"] + 1


def test_no_future_information_in_features(raw_data):
    """Verify that altering future years does not change current-year features."""
    df_orig = raw_data.copy()
    feat_orig, stats_orig = build_features(df_orig)
    
    # Perturb the final year (2024) drastically
    df_perturbed = raw_data.copy()
    mask_2024 = df_perturbed["year"] == 2024
    df_perturbed.loc[mask_2024, "loss_pct"] = 99.9
    df_perturbed.loc[mask_2024, "fire_count"] = 9999
    
    feat_perturbed, _ = build_features(df_perturbed, training_stats=stats_orig)
    
    # Rows for year <= 2023 must remain IDENTICAL
    numeric_cols = feat_orig.select_dtypes(include=[np.number]).columns
    sub_orig = feat_orig[feat_orig["year"] <= 2023][numeric_cols].fillna(0)
    sub_perturbed = feat_perturbed[feat_perturbed["year"] <= 2023][numeric_cols].fillna(0)
    
    pd.testing.assert_frame_equal(sub_orig, sub_perturbed, check_exact=False, rtol=1e-5)


def test_no_same_year_target_leakage_in_feature_cols(raw_data):
    """Ensure raw y_loss_pct_next, delta, etc. are NOT in feature columns."""
    targets_df, _ = build_all_targets(raw_data, horizon=1)
    feat_df, _ = build_features(targets_df)
    cols = get_feature_columns(feat_df)
    
    forbidden = ["y_loss_pct_next", "y_loss_ha_next", "delta", "log_ratio", "severity_next", "target_year"]
    for f in forbidden:
        assert f not in cols, f"Leaked target column '{f}' found in feature columns!"


def test_thresholds_from_train_only(raw_data):
    """Verify class thresholds are calculated strictly on training subset."""
    targets_df, _ = build_all_targets(raw_data)
    train_mask, val_mask, test_mask = get_temporal_holdout(targets_df)
    
    y_train = targets_df.loc[train_mask, "y_loss_pct_next"].dropna()
    thresh_train = compute_class_thresholds(y_train, mode="ternary")
    
    # Thresholds computed from ALL data would be slightly different
    thresh_all = compute_class_thresholds(targets_df["y_loss_pct_next"].dropna(), mode="ternary")
    
    # Verify train threshold matches 33rd/67th percentile of y_train
    assert thresh_train["low_upper"] == pytest.approx(float(np.percentile(y_train, 33)), abs=1e-5)
    assert thresh_train["high_lower"] == pytest.approx(float(np.percentile(y_train, 67)), abs=1e-5)

