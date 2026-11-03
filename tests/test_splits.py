"""Tests for cross-validation and split strategies."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_validate
from src.targets import build_all_targets
from src.splits import (
    get_temporal_holdout,
    get_expanding_window_cv,
    get_spatial_cv,
)


@pytest.fixture(scope="module")
def data_with_targets():
    raw = load_and_validate()
    df, _ = build_all_targets(raw, horizon=1)
    return df


def test_temporal_holdout_strict_ordering(data_with_targets):
    """Verify validation year > train years, test years > val year, no overlap."""
    train_mask, val_mask, test_mask = get_temporal_holdout(data_with_targets)
    
    assert (train_mask & val_mask).sum() == 0, "Train and val overlap!"
    assert (train_mask & test_mask).sum() == 0, "Train and test overlap!"
    assert (val_mask & test_mask).sum() == 0, "Val and test overlap!"

    max_train_yr = data_with_targets.loc[train_mask, "target_year"].max()
    val_yr = data_with_targets.loc[val_mask, "target_year"].unique()
    min_test_yr = data_with_targets.loc[test_mask, "target_year"].min()

    assert val_yr[0] > max_train_yr
    assert min_test_yr > val_yr[0]


def test_expanding_window_cv(data_with_targets):
    """Verify each fold trains on past and validates on the strictly next year."""
    folds = get_expanding_window_cv(data_with_targets, min_train_year=2010, max_val_year=2021)
    assert len(folds) >= 5, "Must have at least 5 folds in expanding CV"

    prev_train_len = 0
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        # No overlap within fold
        assert len(set(train_idx).intersection(set(val_idx))) == 0

        train_years = data_with_targets.iloc[train_idx]["target_year"]
        val_years = data_with_targets.iloc[val_idx]["target_year"]

        assert val_years.nunique() == 1, "Validation fold should be exactly one target year"
        val_year = val_years.iloc[0]
        assert train_years.max() < val_year, f"Fold {fold_i}: train years leak into or exceed val year"

        # Expanding window: train set size strictly increases
        assert len(train_idx) > prev_train_len
        prev_train_len = len(train_idx)


def test_spatial_cv_no_region_bleed(data_with_targets):
    """Verify spatial CV holds out complete regions (no region appears in both train and val)."""
    folds = get_spatial_cv(data_with_targets, n_splits=5)
    assert len(folds) == 5

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        train_regions = set(data_with_targets.iloc[train_idx]["region_id"].unique())
        val_regions = set(data_with_targets.iloc[val_idx]["region_id"].unique())
        assert len(train_regions.intersection(val_regions)) == 0, f"Fold {fold_i}: region leaked across split!"

