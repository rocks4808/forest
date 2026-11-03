"""Feature engineering module for the Forest Loss Prediction pipeline.

All features are computed per region sorted by year, using ONLY information
available at or before year t.  Never looks into the future.

Feature groups
--------------
1. Lag features
2. Rolling statistics
3. Trend and change features
4. Human-pressure and interaction features
5. Policy-timing features
6. Region-level features
7. Feature selection helper
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED: int = 42

LAG_COLUMNS: List[str] = [
    "loss_pct",
    "loss_ha",
    "fire_count",
    "agri_expansion_pct",
    "rainfall_mm",
    "gain_ha",
]

LAG_ORDERS: List[int] = [1, 2, 3]

ROLLING_COLUMNS: List[str] = [
    "loss_pct",
    "fire_count",
    "agri_expansion_pct",
    "rainfall_mm",
]

ROLLING_WINDOWS: List[int] = [3, 5]

RAW_ID_COLS: List[str] = [
    "region_id",
    "year",
    "forest_area_ha",
    "loss_ha",
    "gain_ha",
    "loss_pct",
    "population_density",
    "road_density_km_per_km2",
    "agri_expansion_pct",
    "rainfall_mm",
    "fire_count",
    "protected_area",
    "policy_active",
]

TARGET_COL: str = "target_loss_pct"

EXCLUDE_FROM_FEATURES: List[str] = [
    "region_id",
    "year",
    "target_year",
    "y_loss_pct_next",
    "y_loss_ha_next",
    "delta",
    "log_ratio",
    "severity_next",
    TARGET_COL,
    "size_bucket_label",
]


# ===================================================================
# 0. Sorting helper
# ===================================================================

def _sort(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted by region_id, year."""
    return df.sort_values(["region_id", "year"]).reset_index(drop=True)


# ===================================================================
# 6.1  Lag features
# ===================================================================

def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag-1, lag-2, lag-3 for selected columns.

    Groups by ``region_id`` (already sorted by year) and uses
    ``shift()`` so only past data is used.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``region_id``, ``year``, and all columns in
        ``LAG_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        Input frame augmented with ``<col>_lag<k>`` columns.
    """
    for col in LAG_COLUMNS:
        for lag in LAG_ORDERS:
            df[f"{col}_lag{lag}"] = df.groupby("region_id")[col].shift(lag)
    return df


# ===================================================================
# 6.2  Rolling statistics
# ===================================================================

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std over windows 3 and 5 for selected columns.

    Uses ``min_periods=1`` and groups by ``region_id``.  The rolling
    window is applied chronologically (data already sorted by year).

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by ``region_id``, ``year``.

    Returns
    -------
    pd.DataFrame
        Input frame augmented with rolling feature columns.
    """
    for col in ROLLING_COLUMNS:
        grouped = df.groupby("region_id")[col]
        for win in ROLLING_WINDOWS:
            df[f"{col}_roll_mean{win}"] = grouped.transform(
                lambda s: s.rolling(window=win, min_periods=1).mean()
            )
            df[f"{col}_roll_std{win}"] = grouped.transform(
                lambda s: s.rolling(window=win, min_periods=1).std()
            )
    return df


# ===================================================================
# 6.3  Trend and change features
# ===================================================================

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of a least-squares linear fit over a rolling window.

    Uses ``np.polyfit`` degree-1.  Returns NaN where fewer than 2
    observations are available.
    """
    def _slope(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.nan
        x = np.arange(len(valid), dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(x, valid, deg=1)
        return coeffs[0]

    return series.rolling(window=window, min_periods=2).apply(_slope, raw=True)


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend, difference, and change-rate features.

    All computations use only current and past data within each region.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by ``region_id``, ``year``.

    Returns
    -------
    pd.DataFrame
        Input frame with new trend columns.
    """
    g = df.groupby("region_id")

    # First and second differences of loss_pct
    df["loss_pct_diff1"] = g["loss_pct"].diff(1)
    df["loss_pct_diff2"] = g["loss_pct_diff1"].diff(1)

    # Slope of linear fit over last 3 and 5 years
    df["loss_pct_slope3"] = g["loss_pct"].transform(
        lambda s: _rolling_slope(s, window=3)
    )
    df["loss_pct_slope5"] = g["loss_pct"].transform(
        lambda s: _rolling_slope(s, window=5)
    )

    # Year-over-year percentage change of fire_count
    df["fire_count_pct_change"] = g["fire_count"].pct_change()

    # Population density growth (yoy)
    df["pop_density_growth"] = g["population_density"].pct_change()

    # Forest area change rate
    prev_forest = g["forest_area_ha"].shift(1)
    df["forest_area_change_rate"] = (
        (df["forest_area_ha"] - prev_forest) / prev_forest.replace(0, np.nan)
    )

    # Net loss
    df["net_loss_ha"] = df["loss_ha"] - df["gain_ha"]

    # Gain-to-loss ratio (clip denominator to avoid div-by-zero)
    loss_clipped = df["loss_ha"].clip(lower=1e-6)
    df["gain_to_loss_ratio"] = df["gain_ha"] / loss_clipped

    # Net loss percentage of forest area
    forest_safe = df["forest_area_ha"].replace(0, np.nan)
    df["net_loss_pct"] = df["net_loss_ha"] / forest_safe * 100.0

    return df


# ===================================================================
# 6.4  Human-pressure and interaction features
# ===================================================================

def compute_training_stats(df: pd.DataFrame) -> dict:
    """Compute statistics from training data for feature engineering.

    Stores mean/std of pressure columns, the 20th-percentile of
    rainfall, and per-region mean rainfall.  These are later applied
    unchanged to validation/test data to prevent leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Training split (must contain raw columns).

    Returns
    -------
    dict
        Dictionary of computed statistics keyed by stat name.
    """
    stats: Dict[str, Any] = {}

    # Pressure z-score components
    for col in ["population_density", "road_density_km_per_km2", "agri_expansion_pct"]:
        stats[f"{col}_mean"] = float(df[col].mean())
        stats[f"{col}_std"] = float(df[col].std())
        # Protect against zero std
        if stats[f"{col}_std"] == 0:
            stats[f"{col}_std"] = 1.0

    # Drought threshold (20th percentile of training rainfall)
    stats["rainfall_20pct"] = float(df["rainfall_mm"].quantile(0.20))

    # Region-level mean rainfall (training only)
    region_rain_mean = df.groupby("region_id")["rainfall_mm"].mean().to_dict()
    stats["region_rainfall_mean"] = region_rain_mean

    # Initial forest area quantile boundaries for size_bucket
    first_year_mask = df.groupby("region_id")["year"].transform("min") == df["year"]
    initial_areas = df.loc[first_year_mask, "forest_area_ha"]
    stats["forest_area_q33"] = float(initial_areas.quantile(1 / 3))
    stats["forest_area_q66"] = float(initial_areas.quantile(2 / 3))

    return stats


def compute_pressure_features(
    df: pd.DataFrame,
    training_stats: dict,
) -> pd.DataFrame:
    """Compute human-pressure and interaction features.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by ``region_id``, ``year``.
    training_stats : dict
        Statistics dictionary returned by ``compute_training_stats``.

    Returns
    -------
    pd.DataFrame
        Input frame with new pressure/interaction columns.
    """
    # Simple interactions
    df["fire_x_agri"] = df["fire_count"] * df["agri_expansion_pct"]
    df["road_x_pop"] = df["road_density_km_per_km2"] * df["population_density"]

    # Pressure index (z-scores from training stats)
    z_pop = (
        (df["population_density"] - training_stats["population_density_mean"])
        / training_stats["population_density_std"]
    )
    z_road = (
        (df["road_density_km_per_km2"] - training_stats["road_density_km_per_km2_mean"])
        / training_stats["road_density_km_per_km2_std"]
    )
    z_agri = (
        (df["agri_expansion_pct"] - training_stats["agri_expansion_pct_mean"])
        / training_stats["agri_expansion_pct_std"]
    )
    df["pressure_index"] = z_pop + z_road + z_agri

    # Fire per 100 k ha
    forest_safe = df["forest_area_ha"].replace(0, np.nan)
    df["fire_per_100k_ha"] = df["fire_count"] / forest_safe * 100_000.0

    # Drought flag
    df["drought_flag"] = (
        df["rainfall_mm"] < training_stats["rainfall_20pct"]
    ).astype(np.int8)

    # Rainfall anomaly (region mean from training data)
    region_rain_map = training_stats["region_rainfall_mean"]
    df["rainfall_anomaly"] = df["rainfall_mm"] - df["region_id"].map(region_rain_map)
    # For unseen regions fall back to global training mean
    global_rain_mean = np.mean(list(region_rain_map.values()))
    df["rainfall_anomaly"] = df["rainfall_anomaly"].fillna(
        df["rainfall_mm"] - global_rain_mean
    )

    # Protected × policy interaction
    df["protected_x_policy"] = df["protected_area"] * df["policy_active"]

    return df


# ===================================================================
# 6.5  Policy-timing features
# ===================================================================

def compute_policy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute policy-timing features per region.

    Only uses information from years ≤ t within each region.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by ``region_id``, ``year``.

    Returns
    -------
    pd.DataFrame
        Input frame with new policy columns.
    """
    # ── policy_ever_active (expanding) ─────────────────────────────
    df["policy_ever_active"] = (
        df.groupby("region_id")["policy_active"]
        .transform(lambda s: s.expanding().max())
        .astype(np.int8)
    )

    # ── years_since_policy ─────────────────────────────────────────
    # For each row we find the first year where policy_active == 1
    # within the same region, considering only years <= current year.
    # If policy never active up to year t → -1.
    # If policy starts this year → 0.
    years_since = pd.Series(-1, index=df.index, dtype=np.int64)
    for _, group in df.groupby("region_id"):
        first_active_year: Optional[int] = None
        for idx, row in group.iterrows():
            if first_active_year is None and row["policy_active"] == 1:
                first_active_year = int(row["year"])
            if first_active_year is not None:
                years_since.at[idx] = int(row["year"]) - first_active_year
    df["years_since_policy"] = years_since

    # ── policy_just_started ────────────────────────────────────────
    df["policy_just_started"] = (df["years_since_policy"] == 0).astype(np.int8)

    return df


# ===================================================================
# 6.6  Region-level features
# ===================================================================

def compute_region_features(
    df: pd.DataFrame,
    training_stats: dict,
) -> pd.DataFrame:
    """Compute region-level / structural features.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by ``region_id``, ``year``.
    training_stats : dict
        Training statistics for size-bucket boundaries.

    Returns
    -------
    pd.DataFrame
        Input frame with region-level feature columns.
    """
    g = df.groupby("region_id")

    # Expanding mean / std of loss_pct (only past+present)
    df["region_mean_loss_pct"] = g["loss_pct"].transform(
        lambda s: s.expanding().mean()
    )
    df["region_std_loss_pct"] = g["loss_pct"].transform(
        lambda s: s.expanding().std()
    )
    df["region_mean_fire"] = g["fire_count"].transform(
        lambda s: s.expanding().mean()
    )

    # Forest area rank across all regions within the same year
    df["forest_area_rank"] = df.groupby("year")["forest_area_ha"].rank(
        method="dense", ascending=False
    )

    # Size bucket based on training quantiles of initial forest area
    q33 = training_stats["forest_area_q33"]
    q66 = training_stats["forest_area_q66"]
    initial_area = g["forest_area_ha"].transform("first")
    df["size_bucket"] = np.where(
        initial_area <= q33,
        0,
        np.where(initial_area <= q66, 1, 2),
    ).astype(np.int8)

    # Cumulative loss within region up to year t
    df["cum_loss_ha"] = g["loss_ha"].transform(lambda s: s.expanding().sum())

    # Cumulative loss share = cum_loss / first-year forest area
    first_forest = g["forest_area_ha"].transform("first").replace(0, np.nan)
    df["cum_loss_share"] = df["cum_loss_ha"] / first_forest

    # Year index
    df["year_idx"] = df["year"] - 2005

    return df


# ===================================================================
# 6.8  Feature selection helper
# ===================================================================

def select_features(
    df: pd.DataFrame,
    target_col: str,
    method: str = "correlation",
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Drop one of any highly-correlated feature pair.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that includes both features and *target_col*.
    target_col : str
        Name of the target column (kept regardless).
    method : str
        Currently only ``'correlation'`` is supported.
    threshold : float
        Absolute Pearson correlation above which one feature is
        dropped.  The feature with *lower* correlation to the target
        is dropped.

    Returns
    -------
    pd.DataFrame
        DataFrame with redundant columns removed.
    """
    if method != "correlation":
        raise ValueError(f"Unsupported method: {method!r}")

    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_FROM_FEATURES and c != target_col
    ]

    if len(feature_cols) < 2:
        return df

    corr_matrix = df[feature_cols].corr().abs()

    # Correlation of each feature with the target
    if target_col in df.columns:
        target_corr = df[feature_cols].corrwith(df[target_col]).abs().fillna(0)
    else:
        target_corr = pd.Series(0.0, index=feature_cols)

    to_drop: set = set()
    for i in range(len(feature_cols)):
        if feature_cols[i] in to_drop:
            continue
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the one less correlated with the target
                if target_corr[feature_cols[i]] >= target_corr[feature_cols[j]]:
                    to_drop.add(feature_cols[j])
                else:
                    to_drop.add(feature_cols[i])

    return df.drop(columns=list(to_drop))


# ===================================================================
# Master builder
# ===================================================================

def build_features(
    df: pd.DataFrame,
    training_stats: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Build every feature group and return the augmented frame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with all original columns.
    training_stats : dict or None
        If ``None`` (training mode), statistics are computed from
        *df* itself.  For validation / test, pass the dict returned
        during training to avoid leakage.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(df_with_features, training_stats_used)``
    """
    df = _sort(df.copy())

    # Compute training statistics if not provided (training mode)
    if training_stats is None:
        training_stats = compute_training_stats(df)

    # 6.1 Lag features
    df = compute_lag_features(df)

    # 6.2 Rolling statistics
    df = compute_rolling_features(df)

    # 6.3 Trend & change features
    df = compute_trend_features(df)

    # 6.4 Pressure & interaction features
    df = compute_pressure_features(df, training_stats)

    # 6.5 Policy-timing features
    df = compute_policy_features(df)

    # 6.6 Region-level features
    df = compute_region_features(df, training_stats)

    # Replace infinities introduced by division with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df, training_stats


# ===================================================================
# Feature column helper
# ===================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of feature column names (excluding target, IDs).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by ``build_features``.

    Returns
    -------
    list[str]
        Sorted list of numeric feature column names.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in numeric_cols
        if c not in EXCLUDE_FROM_FEATURES
    ]
    return sorted(feature_cols)
