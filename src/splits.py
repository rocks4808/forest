"""
splits.py – Temporal and spatial cross-validation utilities for forest-loss prediction.

Panel data layout: 50 regions × 20 years (2005-2024).
The DataFrame is expected to contain a ``target_year`` column (= ``year + horizon``).
If it is missing, the helpers compute it as ``year + 1``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 42


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def _ensure_target_year(df: pd.DataFrame) -> pd.Series:
    """Return the ``target_year`` series, computing it on-the-fly if absent.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``year``.  May contain ``target_year``.

    Returns
    -------
    pd.Series
        ``target_year`` aligned to *df*'s index.
    """
    if "target_year" in df.columns:
        return df["target_year"]
    return df["year"] + 1


# ---------------------------------------------------------------------------
# Temporal holdout split
# ---------------------------------------------------------------------------
def get_temporal_holdout(
    df: pd.DataFrame,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for a fixed temporal train / val / test split.

    All year references are **target years** (``year + horizon``).

    Default scheme (horizon = 1):
    * **Train**: target_year ≤ 2021  (feature years ≤ 2020)
    * **Val**  : target_year == 2022  (feature year 2021)
    * **Test** : target_year ∈ {2023, 2024}  (feature years 2022-2023)

    Rows whose ``y_loss_pct_next`` is NaN (last year per region) are excluded
    from *all* masks so they never enter modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with ``target_year`` (or ``year``).
    val_years : list[int], optional
        Target years for the validation set.  Default ``[2022]``.
    test_years : list[int], optional
        Target years for the test set.  Default ``[2023, 2024]``.

    Returns
    -------
    (train_mask, val_mask, test_mask)
        Three boolean ``np.ndarray`` of length ``len(df)``.
    """
    if val_years is None:
        val_years = [2022]
    if test_years is None:
        test_years = [2023, 2024]

    target_year = _ensure_target_year(df)

    # Rows with a valid target (not NaN)
    has_target = True
    if "y_loss_pct_next" in df.columns:
        has_target = df["y_loss_pct_next"].notna()

    # Exclude test and val years from training
    all_held_out = set(val_years) | set(test_years)
    max_train_year = min(all_held_out) - 1  # everything strictly before the earliest held-out year

    train_mask = (target_year <= max_train_year) & has_target
    val_mask = target_year.isin(val_years) & has_target
    test_mask = target_year.isin(test_years) & has_target

    return (
        train_mask.to_numpy(dtype=bool),
        val_mask.to_numpy(dtype=bool),
        test_mask.to_numpy(dtype=bool),
    )


# ---------------------------------------------------------------------------
# Expanding-window time-series CV
# ---------------------------------------------------------------------------
def get_expanding_window_cv(
    df: pd.DataFrame,
    min_train_year: int = 2010,
    max_val_year: int = 2021,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(train_idx, val_idx)`` pairs for expanding-window CV.

    All year references are **target years**.

    The first fold trains on target_year ≤ ``min_train_year + 5`` and validates
    on the next year.  Each subsequent fold adds one more year to training.

    With defaults (``min_train_year=2010``, ``max_val_year=2021``) the folds are:

    ====  ===============  ============
    Fold  Train targets ≤  Val target
    ====  ===============  ============
    1     2015             2016
    2     2016             2017
    3     2017             2018
    4     2018             2019
    5     2019             2020
    6     2020             2021
    ====  ===============  ============

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with ``target_year`` (or ``year``).
    min_train_year : int
        Earliest target year allowed in training (used to derive the first
        fold's upper bound via ``min_train_year + 5``).
    max_val_year : int
        Latest target year to use as a validation year.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Each element is ``(train_indices, val_indices)`` — integer position
        indices into *df*.
    """
    target_year = _ensure_target_year(df)

    # Rows with a valid target
    has_target_mask = np.ones(len(df), dtype=bool)
    if "y_loss_pct_next" in df.columns:
        has_target_mask = df["y_loss_pct_next"].notna().to_numpy()

    # First fold trains up to min_train_year + 5, validates on min_train_year + 6
    first_train_upper = min_train_year + 5  # 2015 with defaults
    first_val_year = first_train_upper + 1  # 2016

    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    val_year = first_val_year
    while val_year <= max_val_year:
        train_upper = val_year - 1

        train_idx = np.where(
            (target_year.to_numpy() >= min_train_year)
            & (target_year.to_numpy() <= train_upper)
            & has_target_mask
        )[0]

        val_idx = np.where(
            (target_year.to_numpy() == val_year)
            & has_target_mask
        )[0]

        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))

        val_year += 1

    return folds


# ---------------------------------------------------------------------------
# Spatial (region-grouped) CV
# ---------------------------------------------------------------------------
def get_spatial_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(train_idx, val_idx)`` pairs for region-grouped CV.

    Uses :class:`sklearn.model_selection.GroupKFold` with ``region_id`` as
    the group variable so that *all* years of a given region appear in the
    same fold.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``region_id``.  Optionally ``y_loss_pct_next`` (rows
        with NaN target are excluded).
    n_splits : int
        Number of folds (default 5).
    seed : int
        Random seed used to shuffle the groups before splitting so that the
        assignment is reproducible but not alphabetical.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Each element is ``(train_indices, val_indices)`` — integer position
        indices into *df*.
    """
    # Filter to rows with valid targets
    has_target_mask = np.ones(len(df), dtype=bool)
    if "y_loss_pct_next" in df.columns:
        has_target_mask = df["y_loss_pct_next"].notna().to_numpy()

    valid_idx = np.where(has_target_mask)[0]
    df_valid = df.iloc[valid_idx].copy()

    groups = df_valid["region_id"].values

    # Shuffle group labels deterministically so fold assignment isn't order-dependent
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    group_map = {g: i for i, g in enumerate(shuffled)}
    mapped_groups = np.array([group_map[g] for g in groups])

    gkf = GroupKFold(n_splits=n_splits)

    # GroupKFold ignores X and y; we pass dummies
    dummy_X = np.zeros((len(df_valid), 1))
    dummy_y = np.zeros(len(df_valid))

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_local, val_local in gkf.split(dummy_X, dummy_y, groups=mapped_groups):
        # Map local indices back to original df indices
        train_orig = valid_idx[train_local]
        val_orig = valid_idx[val_local]
        folds.append((train_orig, val_orig))

    return folds
